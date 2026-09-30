from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import random
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from WFlib import models
from WFlib.tools import data_processor, evaluator

try:
    from log import log_epoch
except ImportError:  # python -m exp.proteus
    from exp.log import log_epoch

LOGGER = logging.getLogger("proteus.training")


def freeze(module):
    module.requires_grad_(False)
    module.eval()  # Also freeze BN running statistics and disable dropout.


class RFAdapter(nn.Module):
    """Expose F and g without changing the original model/checkpoint keys.

    RF's last three features layers are Conv1d(512, C), BN(C), ReLU.
    The split is BEFORE these layers, not immediately before GAP (which
    would give C channels). GAP itself has no trainable parameters.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        tail = list(model.features.children())[-3:]
        if (
            len(tail) != 3
            or not isinstance(tail[0], nn.Conv1d)
            or tail[0].in_channels != 512
            or not isinstance(tail[1], nn.BatchNorm1d)
            or not isinstance(tail[2], nn.ReLU)
            or not isinstance(model.classifier, nn.AdaptiveAvgPool1d)
        ):
            raise ValueError("Expected RF/DMX tail: Conv1d(512, C) + BN + ReLU + GAP")
        self.cut = len(model.features) - 3
        freeze(self)

    def features(self, inputs):
        x = self.model.first_layer(inputs)
        x = x.reshape(x.shape[0], self.model.first_layer_out_channel, -1)
        for layer in list(self.model.features.children())[: self.cut]:
            x = layer(x)
        return x  # [B, 512, L], no flattening or global pooling.

    def logits_from_features(self, features):
        x = features
        for layer in list(self.model.features.children())[self.cut :]:
            x = layer(x)
        return self.model.classifier(x).flatten(1)

    def forward(self, inputs):
        return self.logits_from_features(self.features(inputs))

    def train(self, mode=True):
        # Every train() call leaves the entire backbone (including BN) in eval.
        super().train(False)
        self.training = mode
        for layer in list(self.model.features.children())[self.cut :]:
            layer.train(mode)
        self.model.classifier.train(mode)
        return self

    def unfreeze_tail(self):
        freeze(self)
        for layer in list(self.model.features.children())[self.cut :]:
            layer.requires_grad_(True)
        self.train()


@dataclass
class TargetBank:
    """CPU storage: complete descriptor sets, positions, and cached support maps."""

    descriptors: list
    positions: list
    support_maps: torch.Tensor
    support_labels: torch.Tensor
    variances: torch.Tensor
    weights: torch.Tensor


def local_descriptors(maps):
    """Flatten only the sample/time axes, preserving normalized time u in [0, 1]."""
    batch, channels, length = maps.shape
    descriptors = maps.transpose(1, 2).reshape(batch * length, channels)
    positions = torch.linspace(0, 1, length, device=maps.device).repeat(batch)
    return descriptors, positions


def reliability_weights(class_maps, strength=5.0, singleton_weight=0.5):
    """w_c = 1 / (1 + strength * v_c), using support samples ONLY.

    v_c is the mean squared deviation of channel-normalized maps across
    samples at the SAME time position (sum channels, then average time).
    This measures sample variability rather than normal temporal structure.
    One-shot variance is unidentifiable, so use an explicit fallback weight.
    """
    variances, weights = [], []
    for maps in class_maps:
        normalized = F.normalize(maps, dim=1, eps=1e-8)
        variance = normalized.var(dim=0, unbiased=False).sum(0).mean()
        variances.append(variance)
        weights.append(
            variance.new_tensor(singleton_weight)
            if len(maps) == 1
            else 1 / (1 + strength * variance)
        )
    return torch.stack(variances), torch.stack(weights)


@torch.no_grad()
def build_target_bank(
    adapter,
    support_loader,
    num_classes,
    device,
    reliability_strength=5.0,
    singleton_weight=0.5,
):
    """Stage 1 has NO optimizer and never modifies the pretrained model."""
    freeze(adapter)
    maps, labels = [], []
    for inputs, batch_labels in support_loader:
        maps.append(adapter.features(inputs.to(device)).cpu())
        labels.append(batch_labels.long().cpu())
    if not maps:
        raise ValueError("Empty target support set")
    maps, labels = torch.cat(maps), torch.cat(labels)
    by_class = [maps[labels == c] for c in range(num_classes)]
    if any(len(values) == 0 for values in by_class):
        raise ValueError("Every class must have target support samples")
    descriptors, positions = zip(*(local_descriptors(values) for values in by_class))
    variances, weights = reliability_weights(
        by_class,
        reliability_strength,
        singleton_weight,
    )
    return TargetBank(
        list(descriptors), list(positions), maps, labels, variances, weights
    )


class ClassConditionalAffine(nn.Module):
    """Only 2 * 512 parameters per class; identity initialization.

    a is a scale DELTA, so regularizing a toward zero preserves the identity.
    The same channel affine map is applied to every time slice of a class.
    """

    def __init__(self, num_classes, channels=512):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(num_classes, channels))
        self.b = nn.Parameter(torch.zeros(num_classes, channels))

    def forward(self, maps, labels):
        return maps * (1 + self.a[labels, :, None]) + self.b[labels, :, None]

    def descriptors(self, values, class_id):
        return values * (1 + self.a[class_id]) + self.b[class_id]

    def regularization(self):
        return self.a.square().mean() + self.b.square().mean()


def subsample_descriptors(values, positions, limit):
    """Sample only for OT computation; Stage 1's stored bank stays complete."""
    if limit > 0 and len(values) > limit:
        indices = torch.randperm(len(values), device=values.device)[:limit]
        return values[indices], positions[indices]
    return values, positions


def local_ot_loss(
    source, target, source_u, target_u, eta=0.1, epsilon=0.05, iterations=50
):
    """Balanced, entropy-regularized OT with uniform descriptor marginals.

    d_ij = 1 - cos(h_i, h_j) + eta * |u_i - u_j|.
    Log-domain Sinkhorn approximates the optimal coupling. The coupling is
    detached (alternating OT optimization / envelope gradient); gradients
    flow through the live cost into the affine mapper. KL(P || uniform)
    regularizes the transport PLAN; affine regularization is separate.
    """
    if epsilon <= 0 or iterations < 1 or eta < 0:
        raise ValueError("OT requires epsilon > 0, iterations >= 1 and eta >= 0")
    source = F.normalize(source, dim=1, eps=1e-8)
    target = F.normalize(target, dim=1, eps=1e-8)
    cost = (1 - source @ target.T).clamp_min(0)
    cost = cost + eta * (source_u[:, None] - target_u[None, :]).abs()
    rows, cols = cost.shape
    with torch.no_grad():
        kernel = -cost / epsilon
        log_v = torch.zeros(cols, device=cost.device, dtype=cost.dtype)
        for _ in range(iterations):
            log_u = -math.log(rows) - torch.logsumexp(kernel + log_v[None, :], dim=1)
            log_v = -math.log(cols) - torch.logsumexp(kernel + log_u[:, None], dim=0)
        log_plan = kernel + log_u[:, None] + log_v[None, :]
        plan = log_plan.exp()
        kl = (plan * (log_plan + math.log(rows * cols))).sum()
    return (plan * cost).sum() + epsilon * kl


def train_translator(
    adapter,
    translator,
    source_loader,
    bank,
    device,
    epochs=50,
    lr=1e-3,
    eta=0.1,
    epsilon=0.05,
    iterations=50,
    max_descriptors=128,
    regularization_weight=0.01,
):
    """Stage 2: class-matched local OT + identity regularization; only T learns."""
    freeze(adapter)
    translator.to(device).train()
    optimizer = torch.optim.Adam(translator.parameters(), lr=lr)
    for epoch in range(epochs):
        sums = dict(loss=0.0, ot=0.0, regularization=0.0)
        steps = 0
        for inputs, labels in source_loader:
            labels = labels.to(device).long()
            with torch.no_grad():
                maps = adapter.features(inputs.to(device))
            losses = []
            for class_id in labels.unique().tolist():
                source, source_u = local_descriptors(maps[labels == class_id])
                source, source_u = subsample_descriptors(
                    source, source_u, max_descriptors
                )
                target, target_u = subsample_descriptors(
                    bank.descriptors[class_id],
                    bank.positions[class_id],
                    max_descriptors,
                )
                losses.append(
                    local_ot_loss(
                        translator.descriptors(source, class_id),
                        target.to(device),
                        source_u,
                        target_u.to(device),
                        eta,
                        epsilon,
                        iterations,
                    )
                )
            # Average classes present in this minibatch, not individual slices.
            ot = torch.stack(losses).mean()
            regularization = translator.regularization()
            loss = ot + regularization_weight * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for name, value in zip(sums, (loss, ot, regularization)):
                sums[name] += value.detach().item()
            steps += 1
        log_epoch(2, epoch + 1, epochs, {k: v / max(steps, 1) for k, v in sums.items()})
    freeze(translator)


def weighted_source_ce(logits, labels, weights):
    # Do NOT divide by sum(weights): unreliable classes must reduce the loss scale.
    return (weights[labels] * F.cross_entropy(logits, labels, reduction="none")).mean()


def train_stage3(
    adapter,
    translator,
    source_loader,
    bank,
    device,
    epochs=50,
    lr=1e-4,
    source_weight=1.0,
    batch_size=128,
    eval_loader=None,
):
    """Stage 3: CE(target) + lambda * mean(w_y * CE(translated source)).

    Stream the large source set instead of caching every [512, L] feature map.
    Target support maps are cached safely because F never changes. A joint
    tail forward updates tail BN on the mixed target/source feature batch.
    """
    freeze(translator)
    adapter.unfreeze_tail()
    weights = bank.weights.to(device)
    support_loader = make_loader(
        bank.support_maps, bank.support_labels, batch_size, True, 0
    )
    optimizer = torch.optim.Adam(
        (p for p in adapter.parameters() if p.requires_grad), lr=lr
    )
    LOGGER.info(
        "Stage 3 trainable parameters: %s",
        [name for name, p in adapter.named_parameters() if p.requires_grad],
    )
    for epoch in range(epochs):
        adapter.train()  # Keeps backbone BN frozen, enables only tail BN.
        support_iter = iter(support_loader)
        sums = dict(loss=0.0, target=0.0, source_weighted=0.0)
        count = 0
        # lambda=0 is a target-only ablation; source cannot affect BN in this case.
        batches = source_loader if source_weight > 0 else support_loader
        for inputs, labels in batches:
            if source_weight > 0:
                try:
                    target_maps, target_labels = next(support_iter)
                except StopIteration:
                    support_iter = iter(support_loader)
                    target_maps, target_labels = next(support_iter)
            else:
                target_maps, target_labels = inputs, labels
            target_maps, target_labels = (
                target_maps.to(device),
                target_labels.to(device).long(),
            )
            n_target = len(target_labels)
            if source_weight > 0:
                labels = labels.to(device).long()
                with torch.no_grad():
                    translated = translator(adapter.features(inputs.to(device)), labels)
                logits = adapter.logits_from_features(
                    torch.cat([target_maps, translated])
                )
                source_loss = weighted_source_ce(logits[n_target:], labels, weights)
            else:
                logits = adapter.logits_from_features(target_maps)
                source_loss = logits.new_zeros(())
            target_loss = F.cross_entropy(logits[:n_target], target_labels)
            loss = target_loss + source_weight * source_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = len(labels)
            count += n
            for name, value in zip(sums, (loss, target_loss, source_loss)):
                sums[name] += value.detach().item() * n
        accuracy = (
            evaluate_model(adapter, eval_loader, device)["Accuracy"]
            if eval_loader
            else None
        )
        log_epoch(
            3,
            epoch + 1,
            epochs,
            {k: v / max(count, 1) for k, v in sums.items()},
            accuracy,
            acc_scope="held-out target (never used for training)",
        )


@torch.no_grad()
def evaluate_model(adapter, loader, device):
    adapter.eval()
    predictions, labels = [], []
    for inputs, batch_labels in loader:
        predictions.append(adapter(inputs.to(device)).argmax(1).cpu().numpy())
        labels.append(batch_labels.numpy())
    return evaluator.measurement(
        np.concatenate(labels), np.concatenate(predictions), ["Accuracy"]
    )


def select_support_indices(labels, shot, seed=2024):
    if shot < 1:
        raise ValueError("shot must be at least 1")
    values = labels.cpu().numpy()
    rng = np.random.default_rng(seed)
    support = []
    for label in np.unique(values):
        candidates = np.flatnonzero(values == label)
        if len(candidates) <= shot:
            raise ValueError(
                f"Class {label} needs at least shot + 1 samples for support/evaluation"
            )
        support.extend(rng.choice(candidates, shot, replace=False).tolist())
    support = np.asarray(sorted(support), dtype=np.int64)
    mask = np.ones(len(values), dtype=bool)
    mask[support] = False
    return support, np.flatnonzero(mask)


def make_loader(data, labels, batch_size, shuffle, num_workers):
    return DataLoader(
        TensorDataset(data, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
    )


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_checkpoint(model, path):
    state = load_checkpoint(path)
    if "classifier.weight" in state:
        raise ValueError(
            f"{path} uses the legacy convolution-only classifier. This protocol "
            "requires the RF Conv1d + BN + ReLU + GAP tail. Retrain the source "
            "model with the current RF/DMX definition (RUN_BASE_TRAIN=1 in DMX.sh)."
        )
    model.load_state_dict(state)


def stage2_checkpoint_path(args):
    return (
        Path(args.checkpoints)
        / args.dataset
        / args.model
        / "stage2"
        / f"{args.train_file}_to_{args.test_file}_shot{args.shot}"
        f"_seed{args.support_seed}.pth"
    )


def load_stage2_checkpoint(translator, path, args, support_indices):
    state = load_checkpoint(path)
    # Stage 3 settings may change, but the translator must match its data and F.
    keys = (
        "dataset",
        "model",
        "train_file",
        "test_file",
        "shot",
        "support_seed",
        "feature",
        "seq_len",
        "load_name",
    )
    saved_config = state.get("config", {})
    mismatches = [key for key in keys if saved_config.get(key) != getattr(args, key)]
    if mismatches:
        raise ValueError(
            f"Stage 2 checkpoint {path} has incompatible config: {', '.join(mismatches)}"
        )
    if state.get("support_indices") != support_indices.tolist():
        raise ValueError(f"Stage 2 checkpoint {path} has incompatible support indices")
    translator.load_state_dict(state["translator"])
    freeze(translator)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("Data and checkpoint")
    data.add_argument("--dataset", required=True)
    data.add_argument("--model", choices=("RF", "DMX"), default="DMX")
    data.add_argument("--device", default="cpu")
    data.add_argument("--train_file", default="tam_train")
    data.add_argument("--test_file", default="tam_day270")
    data.add_argument("--feature", default="TAM")
    data.add_argument("--seq_len", type=int, default=1800)
    data.add_argument("--batch_size", type=int, default=128)
    data.add_argument("--num_workers", type=int, default=4)
    data.add_argument("--seed", type=int, default=2024)
    data.add_argument("--shot", type=int, default=10)
    data.add_argument("--support_seed", type=int, default=20070206)
    data.add_argument("--log_path", default="./logs")
    data.add_argument("--checkpoints", default="./checkpoints")
    data.add_argument("--load_name", default="max_f1")
    data.add_argument("--model_save_name", default="proteus")
    data.add_argument("--result_file", default="proteus")
    stage2 = parser.add_argument_group("Stage 2: class-conditional affine + local OT")
    stage2.add_argument(
        "--load_stage2",
        nargs="?",
        const="auto",
        metavar="PATH",
        help="Load a translator and skip Stage 2; omit PATH to use the dataset/shot default",
    )
    stage2.add_argument("--stage2_epochs", type=int, default=50)
    stage2.add_argument("--map_lr", type=float, default=1e-3)
    stage2.add_argument(
        "--ot_eta", type=float, default=0.1, help="Time-position cost coefficient"
    )
    stage2.add_argument(
        "--ot_epsilon", type=float, default=0.05, help="Sinkhorn entropy strength"
    )
    stage2.add_argument("--ot_iterations", type=int, default=50)
    stage2.add_argument(
        "--ot_max_descriptors",
        type=int,
        default=128,
        help="Per class/domain/minibatch OT slice limit; 0 uses all slices",
    )
    stage2.add_argument("--map_reg_weight", type=float, default=0.01)
    stage3 = parser.add_argument_group(
        "Stage 3: frozen backbone + trainable classification tail"
    )
    stage3.add_argument("--stage3_epochs", type=int, default=50)
    stage3.add_argument("--adapt_lr", type=float, default=1e-4)
    stage3.add_argument(
        "--augmented_loss_weight",
        type=float,
        default=1.0,
        help="Source CE coefficient lambda",
    )
    stage3.add_argument(
        "--reliability_strength",
        type=float,
        default=5.0,
        help="w_c = 1 / (1 + strength * variance_c)",
    )
    stage3.add_argument(
        "--singleton_weight",
        type=float,
        default=0.5,
        help="Fallback reliability for one-shot classes",
    )
    return parser


def validate_args(args):
    for name in ("batch_size", "seq_len", "shot", "ot_iterations"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be >= 1")
    for name in ("map_lr", "adapt_lr", "ot_epsilon"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and > 0")
    for name in (
        "stage2_epochs",
        "stage3_epochs",
        "num_workers",
        "ot_eta",
        "ot_max_descriptors",
        "map_reg_weight",
        "augmented_loss_weight",
        "reliability_strength",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and >= 0")
    if not 0 <= args.singleton_weight <= 1:
        raise ValueError("singleton_weight must lie in [0, 1]")


def main(argv: Optional[Sequence[str]] = None):
    args = make_parser().parse_args(argv)
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cpu"
        if args.device.startswith("cuda") and not torch.cuda.is_available()
        else args.device
    )
    dataset_path = Path("datasets") / args.dataset
    log_path = Path(args.log_path) / args.dataset / args.model
    checkpoint_path = Path(args.checkpoints) / args.dataset / args.model
    log_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    LOGGER.addHandler(
        logging.FileHandler(
            log_path / f"{args.result_file}.log", mode="a", encoding="utf-8"
        )
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.info(
        "\nLocal OT adaptation | device=%s\n%s",
        device,
        json.dumps(vars(args), indent=2),
    )
    source_data, source_labels = data_processor.load_data(
        str(dataset_path / f"{args.train_file}.npz"),
        args.feature,
        args.seq_len,
    )
    target_data, target_labels = data_processor.load_data(
        str(dataset_path / f"{args.test_file}.npz"),
        args.feature,
        args.seq_len,
    )
    classes = torch.unique(target_labels)
    num_classes = len(classes)
    if not num_classes or not torch.equal(classes, torch.arange(num_classes)):
        raise ValueError("Target labels must be contiguous integers starting at zero")
    if not torch.equal(torch.unique(source_labels), classes):
        raise ValueError("Source and target must contain the same classes/label IDs")
    support_indices, eval_indices = select_support_indices(
        target_labels, args.shot, args.support_seed
    )
    source_loader = make_loader(
        source_data, source_labels, args.batch_size, True, args.num_workers
    )
    support_loader = make_loader(
        target_data[support_indices],
        target_labels[support_indices],
        args.batch_size,
        False,
        args.num_workers,
    )
    eval_loader = make_loader(
        target_data[eval_indices],
        target_labels[eval_indices],
        args.batch_size,
        False,
        args.num_workers,
    )
    model = getattr(models, args.model)(num_classes)
    load_model_checkpoint(model, checkpoint_path / f"{args.load_name}.pth")
    adapter = RFAdapter(model).to(device)
    baseline = evaluate_model(adapter, eval_loader, device)
    log_epoch(
        1, 0, 0, {}, baseline["Accuracy"], acc_scope="held-out target before adaptation"
    )

    bank = build_target_bank(
        adapter,
        support_loader,
        num_classes,
        device,
        args.reliability_strength,
        args.singleton_weight,
    )
    print(
        f"[Stage 1] {num_classes} classes; support maps {tuple(bank.support_maps.shape)}; "
        f"reliability range [{bank.weights.min():.4f}, {bank.weights.max():.4f}]",
        flush=True,
    )
    translator = ClassConditionalAffine(num_classes).to(device)
    stage2_path = stage2_checkpoint_path(args)
    if args.load_stage2:
        if args.load_stage2 != "auto":
            stage2_path = Path(args.load_stage2)
        load_stage2_checkpoint(translator, stage2_path, args, support_indices)
        message = f"[Stage 2] Loaded translator from {stage2_path}; training skipped"
    else:
        train_translator(
            adapter,
            translator,
            source_loader,
            bank,
            device,
            epochs=args.stage2_epochs,
            lr=args.map_lr,
            eta=args.ot_eta,
            epsilon=args.ot_epsilon,
            iterations=args.ot_iterations,
            max_descriptors=args.ot_max_descriptors,
            regularization_weight=args.map_reg_weight,
        )
        stage2_path.parent.mkdir(parents=True, exist_ok=True)
        # Save before Stage 3, without moving the live translator off its device.
        torch.save(
            {
                "translator": {
                    k: v.detach().cpu() for k, v in translator.state_dict().items()
                },
                "weights": bank.weights,
                "variances": bank.variances,
                "support_indices": support_indices.tolist(),
                "config": vars(args),
            },
            stage2_path,
        )
        message = f"[Stage 2] Saved translator to {stage2_path}"
    print(message, flush=True)
    LOGGER.info(message)
    train_stage3(
        adapter,
        translator,
        source_loader,
        bank,
        device,
        epochs=args.stage3_epochs,
        lr=args.adapt_lr,
        source_weight=args.augmented_loss_weight,
        batch_size=args.batch_size,
        eval_loader=eval_loader,
    )
    result = evaluate_model(adapter, eval_loader, device)
    log_epoch("Final", None, None, {}, result["Accuracy"], acc_scope="held-out target")
    # Original RF/DMX keys: directly usable by exp/test.py; no mapper at inference.
    torch.save(model.state_dict(), checkpoint_path / f"{args.model_save_name}.pth")
    torch.save(
        {
            "translator": translator.cpu().state_dict(),
            "weights": bank.weights,
            "variances": bank.variances,
            "config": vars(args),
        },
        checkpoint_path / f"{args.model_save_name}_adaptation.pth",
    )
    report = dict(
        result,
        baseline_accuracy=baseline["Accuracy"],
        stage2_checkpoint=str(stage2_path),
        support_indices=support_indices.tolist(),
        class_variances=bank.variances.tolist(),
        class_weights=bank.weights.tolist(),
        descriptor_counts=[len(values) for values in bank.descriptors],
        config=vars(args),
    )
    with (log_path / f"{args.result_file}.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return result


if __name__ == "__main__":
    main()
