"""Few-shot feature translation adaptation (three-stage protocol)."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from WFlib import models
from WFlib.tools import data_processor, evaluator
try:
    # ``python exp/proteus.py`` puts exp/ on sys.path, while
    # ``python -m exp.proteus`` resolves this as a package import.
    from log import (configure_logging, log_epoch, log_target_diagnostics,
                     log_prototype_geometry, log_trainable_parameters, Stage2Metrics)
except ImportError:  # pragma: no cover - depends on invocation style
    from exp.log import (configure_logging, log_epoch, log_target_diagnostics,
                         log_prototype_geometry, log_trainable_parameters, Stage2Metrics)


def unpack_output(output):
    if isinstance(output, (tuple, list)):
        return output[0], output[1] if len(output) > 1 else output[0]
    return output, output


def flatten_features(features):
    return features.reshape(features.shape[0], -1)


def embedding_features(features):
    """Return the metric-learning representation used by Proteus.

    DMX now returns a pooled 512-D embedding.  L2 normalization makes
    prototype distances comparable across samples and prevents a few samples
    with large activation norms from dominating the class mean.
    """
    return F.normalize(flatten_features(features), p=2, dim=1, eps=1e-8)


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def evaluate_model(model, data_loader, metrics, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for inputs, batch_labels in data_loader:
            logits, _ = unpack_output(model(inputs.to(device)))
            predictions.append(logits.argmax(1).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    if not predictions:
        return {metric: float("nan") for metric in metrics}
    return evaluator.measurement(
        np.concatenate(labels), np.concatenate(predictions), metrics
    )



def load_checkpoint(path):
    """Load a state dict without the pickle warning on recent PyTorch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        return torch.load(path, map_location="cpu")


def select_support_indices(labels, shot, seed=2024):
    if shot < 1:
        raise ValueError("shot must be at least 1")
    labels_np = labels.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    support = []
    for label in np.unique(labels_np):
        candidates = np.flatnonzero(labels_np == label)
        rng.shuffle(candidates)
        support.extend(candidates[:shot].tolist())
    support = np.asarray(sorted(support), dtype=np.int64)
    all_indices = np.arange(labels_np.shape[0])
    mask = np.ones(labels_np.shape[0], dtype=bool)
    mask[support] = False
    return support, all_indices[mask]


@torch.no_grad()
def compute_target_prototypes(target_encoder, support_loader, num_classes, device):
    target_encoder.eval()
    vectors = [[] for _ in range(num_classes)]
    for inputs, labels in support_loader:
        _, features = unpack_output(target_encoder(inputs.to(device)))
        for feature, label in zip(embedding_features(features), labels.tolist()):
            if 0 <= int(label) < num_classes:
                vectors[int(label)].append(feature)
    if any(not values for values in vectors):
        missing = [str(i) for i, values in enumerate(vectors) if not values]
        raise ValueError(
            "support set has no example for class(es): " + ", ".join(missing)
        )
    prototypes = torch.stack([torch.stack(values).mean(0) for values in vectors])
    return F.normalize(prototypes, p=2, dim=1, eps=1e-8)


class FeatureTranslator(nn.Module):
    """2-3 layer MLP used as the asymmetric source-to-target mapper."""

    def __init__(self, feature_dim, hidden_dim=None, layers=2):
        super().__init__()
        if layers not in (2, 3):
            raise ValueError("translator layers must be 2 or 3")
        hidden_dim = hidden_dim or max(128, min(1024, feature_dim))
        blocks = [nn.Linear(feature_dim, hidden_dim), nn.GELU()]
        if layers == 3:
            blocks.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        blocks.append(nn.Linear(hidden_dim, feature_dim))
        self.network = nn.Sequential(*blocks)

    def forward(self, features):
        return F.normalize(self.network(flatten_features(features)), p=2, dim=1, eps=1e-8)


def attractive_alignment_loss(translated, prototypes, labels):
    """Pull each translated source feature to its target-class prototype."""
    return (translated - prototypes[labels]).pow(2).sum(1).mean()


def repulsive_contrast_loss(translated, prototypes, labels, alpha=1.0):
    """Keep translated features at least ``alpha`` away from other classes."""
    distances = torch.cdist(translated, prototypes)
    other = torch.ones_like(distances, dtype=torch.bool)
    other[torch.arange(distances.shape[0], device=distances.device), labels] = False
    return (
        F.relu(alpha - distances[other]).mean()
        if other.any()
        else translated.new_zeros(())
    )


def translator_loss(translated, prototypes, labels, alpha=1.0, lambda_contrast=1.0):
    """Return ``L_align + lambda * L_contrast`` from the report."""
    align = attractive_alignment_loss(translated, prototypes, labels)
    contrast = repulsive_contrast_loss(translated, prototypes, labels, alpha)
    return align + lambda_contrast * contrast, align, contrast


def train_epoch_target(
    target_encoder, support_loader, device, optimizer,
):
    """Stage 1: fine-tune only on target support set (no source CE drift)."""
    target_encoder.train()
    criterion = nn.CrossEntropyLoss()
    sums = {"loss": 0.0}
    count = 0
    for support_inputs, support_labels in support_loader:
        support_inputs, support_labels = (
            support_inputs.to(device),
            support_labels.to(device).long(),
        )
        optimizer.zero_grad()
        logits, _ = unpack_output(target_encoder(support_inputs))
        loss = criterion(logits, support_labels)
        loss.backward()
        optimizer.step()
        n = support_inputs.shape[0]
        count += n
        sums["loss"] += float(loss.detach()) * n
    return {name: value / max(count, 1) for name, value in sums.items()}


def train_translator(
    translator,
    feature_encoder,  # same encoder used to extract source features & compute prototypes
    source_loader,
    prototypes,
    device,
    epochs=20,
    lr=1e-3,
    alpha=1.0,
    contrast_weight=1.0,
):
    """Stage 2: L_align + lambda * L_contrast; only T is updated."""
    feature_encoder.eval()
    freeze(feature_encoder)
    translator.to(device)
    prototypes = prototypes.to(device)
    optimizer = torch.optim.Adam(translator.parameters(), lr=lr)
    for epoch in range(epochs):
        translator.train()
        metrics = Stage2Metrics()
        for inputs, labels in source_loader:
            with torch.no_grad():
                source_features = embedding_features(
                    unpack_output(feature_encoder(inputs.to(device)))[1]
                )
            labels = labels.to(device).long()
            valid = (labels >= 0) & (labels < prototypes.shape[0])
            if not valid.any():
                continue
            translated = translator(source_features[valid])
            chosen_labels = labels[valid]
            loss, align_loss, contrast_loss = translator_loss(
                translated, prototypes, chosen_labels, alpha, contrast_weight
            )
            optimizer.zero_grad()
            metrics.measure_gradient_conflict(
                align_loss, contrast_loss, translator, contrast_weight,
            )
            loss.backward()
            metrics.update(
                translated, prototypes, chosen_labels, translator, alpha,
                loss=loss, align=align_loss, contrast=contrast_loss,
            )
            optimizer.step()
        metrics.log_epoch(epoch + 1, epochs)
    return translator


def train_stage3(
    target_encoder,
    translator,
    source_loader,
    support_loader,
    device,
    epochs,
    lr,
    target_loss_weight=1.0,
    augmented_loss_weight=1.0,
    max_pseudo=0,
    eval_loader=None,
    eval_device=None,
    source_loss_weight=0.0,
):
    """Stage 3: train on target support (full forward) + raw source (full forward)
    + translated features (via logits_from_embedding for head only).

    All feature extraction uses the *same* target_encoder (frozen at Stage 3 start),
    so the translator's input space matches Stage 2's learned mapping.
    """
    freeze(translator)
    translator.eval()
    target_encoder.to(device)
    # Stage 2 froze this same object; train()/to() do not undo that freeze.
    target_encoder.requires_grad_(True)
    log_trainable_parameters(target_encoder, translator)
    optimizer = torch.optim.Adam(target_encoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Pre-compute translated features using the current target_encoder's embedding space
    target_encoder.eval()  # temporarily eval for feature extraction
    augmented_features, augmented_labels = [], []
    seen = 0
    with torch.no_grad():
        for inputs, labels in source_loader:
            if max_pseudo > 0 and seen >= max_pseudo:
                break
            features = embedding_features(
                unpack_output(target_encoder(inputs.to(device)))[1]
            )
            translated = translator(features).cpu()
            labels = labels.long().cpu()
            if max_pseudo > 0:
                take = min(translated.shape[0], max_pseudo - seen)
                translated, labels = translated[:take], labels[:take]
            augmented_features.append(translated)
            augmented_labels.append(labels)
            seen += translated.shape[0]
    if not augmented_features:
        raise ValueError("Stage 3 received no source samples for augmentation")
    augmented_x = torch.cat(augmented_features)
    augmented_y = torch.cat(augmented_labels)
    augmented_loader = data_processor.load_iter(
        augmented_x, augmented_y, batch_size=min(256, len(augmented_y)),
        is_train=False, num_workers=0
    )
    target_encoder.train()

    source_iter = iter(source_loader)

    for epoch in range(epochs):
        target_encoder.train()
        support_iter = iter(support_loader)
        sums = {"loss": 0.0, "augmented": 0.0, "target": 0.0, "source_raw": 0.0}
        count = 0
        for augmented_batch, augmented_labels_batch in augmented_loader:
            # --- Target support (full forward) ---
            try:
                support_inputs, support_labels = next(support_iter)
            except StopIteration:
                support_iter = iter(support_loader)
                support_inputs, support_labels = next(support_iter)
            support_inputs = support_inputs.to(device)
            support_labels = support_labels.to(device).long()

            # --- Raw source (full forward) ---
            try:
                src_inputs, src_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_inputs, src_labels = next(source_iter)
            src_inputs, src_labels = src_inputs.to(device), src_labels.to(device).long()

            optimizer.zero_grad()

            # Target support: full forward → backbone + head updated
            target_logits, _ = unpack_output(target_encoder(support_inputs))
            target_loss = criterion(target_logits, support_labels)

            # Raw source: full forward → backbone + head updated
            source_logits, _ = unpack_output(target_encoder(src_inputs))
            source_raw_loss = criterion(source_logits, src_labels)

            # Translated features: head only (via logits_from_embedding)
            augmented_batch = augmented_batch.to(device)
            augmented_labels_batch = augmented_labels_batch.to(device).long()
            augmented_logits = target_encoder.logits_from_embedding(augmented_batch)
            augmented_loss = criterion(augmented_logits, augmented_labels_batch)

            total = (
                augmented_loss_weight * augmented_loss
                + target_loss_weight * target_loss
                + source_loss_weight * source_raw_loss
            )
            total.backward()
            optimizer.step()

            n = int(augmented_batch.shape[0])
            count += n
            sums["loss"] += float(total.detach()) * n
            sums["augmented"] += float(augmented_loss.detach()) * n
            sums["target"] += float(target_loss.detach()) * n
            sums["source_raw"] += float(source_raw_loss.detach()) * n

        values = {name: value / max(count, 1) for name, value in sums.items()}
        accuracy = None
        if eval_loader is not None:
            accuracy = evaluate_model(
                target_encoder, eval_loader, ["Accuracy"], eval_device or device,
            )["Accuracy"]
        log_epoch(3, epoch + 1, epochs, values, accuracy, acc_scope="target evaluation")


def build_model(model_name, num_classes, num_tabs):
    return (
        getattr(models, model_name)(num_classes, num_tabs)
        if model_name in ("BAPM", "TMWF")
        else getattr(models, model_name)(num_classes)
    )


def make_parser():
    parser = argparse.ArgumentParser(
        description="Few-shot feature translation adaptation"
    )
    parser.add_argument("--dataset", required=True, default="CW")
    parser.add_argument("--model", required=True, default="DF")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_tabs", type=int, default=1)
    parser.add_argument("--train_file", default="train")
    parser.add_argument("--test_file", default="test")
    parser.add_argument("--feature", default="DIR")
    parser.add_argument("--seq_len", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_method", default="common")
    parser.add_argument(
        "--eval_metrics", nargs="+", default=["Accuracy"],
        help="Deprecated compatibility option; evaluation always reports Accuracy only",
    )
    parser.add_argument("--log_path", default="./logs/")
    parser.add_argument("--checkpoints", default="./checkpoints/")
    parser.add_argument("--load_name", default="base")
    parser.add_argument("--result_file", default="result")
    parser.add_argument("--model_save_name", default="fewshot")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--support_seed", type=int, default=20070206)
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--stage2_epochs", type=int, default=100)
    parser.add_argument("--stage3_epochs", type=int, default=50)
    parser.add_argument("--adapt_lr", type=float, default=1e-4)
    parser.add_argument("--map_lr", type=float, default=1e-3)
    parser.add_argument("--map_hidden", type=int, default=0)
    parser.add_argument("--map_layers", type=int, default=2, choices=(2, 3))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda_contrast", type=float, default=1.0)
    parser.add_argument("--target_loss_weight", type=float, default=1.0)
    parser.add_argument("--augmented_loss_weight", type=float, default=1.0)
    parser.add_argument("--source_raw_weight", type=float, default=0.0,
                      help="Weight for raw source CE (full forward) in Stage 3")
    parser.add_argument("--max_pseudo", type=int, default=0)
    parser.add_argument(
        "--enable_log",
        action="store_true",
        help="Write additional target prototype diagnostics to the log file",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = make_parser().parse_args(argv)
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    device = torch.device(
        "cpu"
        if args.device.startswith("cuda") and not torch.cuda.is_available()
        else args.device
    )
    dataset_path = os.path.join("./datasets", args.dataset)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"The dataset path does not exist: {dataset_path}")
    log_path = os.path.join(args.log_path, args.dataset, args.model)
    checkpoint_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    configure_logging(os.path.join(log_path, f"{args.result_file}.log"), vars(args), device)
    source_data, source_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.train_file}.npz"),
        args.feature,
        args.seq_len,
        args.num_tabs,
    )
    target_data, target_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.test_file}.npz"),
        args.feature,
        args.seq_len,
        args.num_tabs,
    )
    if args.num_tabs != 1:
        raise ValueError(
            "Few-shot adaptation requires integer single-class labels (--num_tabs 1)"
        )
    num_classes = int(torch.unique(target_labels).numel())
    if int(target_labels.max()) + 1 != num_classes:
        raise ValueError("target labels must be contiguous integers starting at 0")
    shot = args.shot
    support_indices, eval_indices = select_support_indices(
        target_labels, shot, args.support_seed
    )
    if len(eval_indices) == 0:
        raise ValueError("target set must contain samples outside support set")
    support_data, support_labels = (
        target_data[support_indices],
        target_labels[support_indices],
    )
    eval_data, eval_labels = target_data[eval_indices], target_labels[eval_indices]
    # Keep the final partial source batch; dropping it can make a small
    # few-shot smoke test (or a small dataset) produce an empty loader.
    source_loader = data_processor.load_iter(
        source_data, source_labels, args.batch_size, True, args.num_workers
    )
    # Never drop the support set: with 1-shot data it is normally smaller than
    # the training batch size.  Shuffling is unnecessary because the support
    # set was sampled with a deterministic seed above.
    support_loader = data_processor.load_iter(
        support_data, support_labels, args.batch_size, True, args.num_workers
    )
    eval_loader = data_processor.load_iter(
        eval_data, eval_labels, args.batch_size, False, args.num_workers
    )
    target_encoder = build_model(args.model, num_classes, args.num_tabs)
    target_encoder.load_state_dict(
        load_checkpoint(os.path.join(checkpoint_path, f"{args.load_name}.pth"))
    )
    target_encoder.to(device)
    optimizer = torch.optim.Adam(target_encoder.parameters(), lr=args.adapt_lr)
    with torch.no_grad():
        feature_dim = embedding_features(
            unpack_output(target_encoder(next(iter(source_loader))[0].to(device)))[1]
        ).shape[1]
    for epoch in range(args.stage1_epochs):
        losses = train_epoch_target(
            target_encoder, support_loader, device, optimizer,
        )
        accuracy = evaluate_model(target_encoder, eval_loader, ["Accuracy"], device)
        log_epoch(1, epoch + 1, args.stage1_epochs, losses,
                  accuracy["Accuracy"], acc_scope="target evaluation")
    prototypes = compute_target_prototypes(
        target_encoder, support_loader, num_classes, device
    )
    log_prototype_geometry(prototypes, source_labels, args.alpha, args.lambda_contrast)
    if args.enable_log:
        log_target_diagnostics(target_encoder, support_loader, eval_loader,
                               prototypes, device, prefix="After Stage 1 / Before Stage 2")
    translator = FeatureTranslator(
        feature_dim, args.map_hidden or None, args.map_layers
    )
    train_translator(
        translator,
        target_encoder,
        source_loader,
        prototypes,
        device,
        args.stage2_epochs,
        args.map_lr,
        args.alpha,
        args.lambda_contrast,
    )
    train_stage3(
        target_encoder,
        translator,
        source_loader,
        support_loader,
        device,
        args.stage3_epochs,
        args.adapt_lr,
        args.target_loss_weight,
        args.augmented_loss_weight,
        args.max_pseudo,
        eval_loader,
        device,
        source_loss_weight=args.source_raw_weight,
    )
    if args.enable_log:
        # Recompute prototypes because Stage 1/3 may have changed the target
        # encoder.  This makes the before/after diagnostics comparable.
        adapted_prototypes = compute_target_prototypes(
            target_encoder, support_loader, num_classes, device
        )
        log_target_diagnostics(target_encoder, support_loader, eval_loader,
                               adapted_prototypes, device, prefix="After Stage 3")
    result = evaluate_model(target_encoder, eval_loader, ["Accuracy"], device)
    log_epoch("Final", None, None, {}, result["Accuracy"], acc_scope="target evaluation")
    torch.save(
        target_encoder.state_dict(),
        os.path.join(checkpoint_path, f"{args.model_save_name}.pth"),
    )
    with open(
        os.path.join(log_path, f"{args.result_file}.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    return result


if __name__ == "__main__":
    main()
