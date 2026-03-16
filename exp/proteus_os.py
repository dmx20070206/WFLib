"""
Open-set adaptation experiment with energy-based unknown detection.

Pipeline:
  1. Load a pretrained source backbone (classes 0..101).
  2. Split target samples into known/unknown by energy score.
  3. Adapt on source labels and target known samples.
  4. Refresh the target split and pseudo labels periodically.
  5. Evaluate by assigning detected unknowns to class 102.
"""

import os
import sys
import json
import random
import warnings
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

from WFlib import models
from WFlib.tools import data_processor, evaluator

sys.path.append(os.path.dirname(__file__))
from utils.debug import *

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, required=True, default="CW")
parser.add_argument("--model", type=str, required=True, default="DF")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--num_tabs", type=int, default=1)
parser.add_argument("--scenario", type=str, default="Closed-world")

parser.add_argument("--train_file", type=str, default="train")
parser.add_argument("--valid_file", type=str, default="valid")
parser.add_argument("--test_file", type=str, default="test")
parser.add_argument("--feature", type=str, default="DIR")
parser.add_argument("--seq_len", type=int, default=5000)

parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--eval_method", type=str, default="common")
parser.add_argument("--eval_metrics", nargs="+", required=True, type=str)
parser.add_argument("--log_path", type=str, default="./logs/")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
parser.add_argument("--load_name", type=str, default="base")
parser.add_argument("--result_file", type=str, default="result")
parser.add_argument("--gmm_threshold", type=float, default=0.6)
parser.add_argument("--model_save_name", type=str, default="proteus")
parser.add_argument("--adapt_epochs", type=int, default=100)
parser.add_argument("--adapt_lr", type=float, default=1e-4)
parser.add_argument("--gmm_refresh", type=int, default=5)
parser.add_argument("--unknown_refresh", type=int, default=5)
parser.add_argument("--energy_weight", type=float, default=0.1)
parser.add_argument("--energy_margin_in", type=float, default=-12.0)
parser.add_argument("--energy_margin_out", type=float, default=-6.0)
parser.add_argument("--energy_temperature", type=float, default=1.0)
parser.add_argument("--fix_seed", type=int, default=20070206)
parser.add_argument("--unknown_label", type=int, default=102)
parser.add_argument("--unknown_posterior_threshold", type=float, default=0.6)
parser.add_argument("--min_known_ratio", type=float, default=0.0)

args = parser.parse_args()

random.seed(args.fix_seed)
torch.manual_seed(args.fix_seed)
np.random.seed(args.fix_seed)


def compute_gaussian_kernel(source, target):
    """Gaussian kernel matrix for MMD computation."""
    n = int(source.size(0)) + int(target.size(0))
    combined = torch.cat([source, target], dim=0)
    l2 = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(2)
    bw = torch.clamp(torch.sum(l2) / (n**2 - n), min=1e-5)
    return torch.exp(-l2 / (bw + 1e-5))


def calculate_mmd_loss(src_feat, tgt_feat):
    """Maximum Mean Discrepancy between source and target features."""
    bs = min(src_feat.size(0), tgt_feat.size(0))
    src_feat, tgt_feat = src_feat[:bs], tgt_feat[:bs]
    kernel = compute_gaussian_kernel(src_feat, tgt_feat)
    xx = kernel[:bs, :bs]
    yy = kernel[bs:, bs:]
    xy = kernel[:bs, bs:]
    yx = kernel[bs:, :bs]
    return torch.mean(xx + yy - xy - yx)


def compute_softmax_entropy(logits):
    """Per-sample softmax entropy."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def compute_energy(logits, temperature=1.0):
    """Energy score: E(x) = -T * logsumexp(logits / T)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def compute_energy_threshold_with_gmm(energies):
    """Fit a 2-component GMM on energies and use the component intersection as threshold."""
    energies = np.asarray(energies, dtype=np.float64).reshape(-1)

    gmm = GaussianMixture(n_components=2, covariance_type="full", tol=1e-6, random_state=args.fix_seed)
    gmm.fit(energies.reshape(-1, 1))

    means = gmm.means_.reshape(-1)
    variances = gmm.covariances_.reshape(2, -1)[:, 0]
    variances = np.maximum(variances, 1e-8)
    stds = np.sqrt(variances)
    weights = np.maximum(gmm.weights_.reshape(-1), 1e-12)

    low_idx = int(np.argmin(means))
    high_idx = 1 - low_idx
    m1, s1, w1 = float(means[low_idx]), float(stds[low_idx]), float(weights[low_idx])
    m2, s2, w2 = float(means[high_idx]), float(stds[high_idx]), float(weights[high_idx])

    # Solve w1*N(x|m1,s1^2) = w2*N(x|m2,s2^2), then pick the root between two means.
    a = 1.0 / (2.0 * s2 * s2) - 1.0 / (2.0 * s1 * s1)
    b = m1 / (s1 * s1) - m2 / (s2 * s2)
    c = (m2 * m2) / (2.0 * s2 * s2) - (m1 * m1) / (2.0 * s1 * s1) + np.log((w2 * s1) / (w1 * s2))

    roots = []
    if abs(a) < 1e-10:
        if abs(b) > 1e-10:
            roots = [(-c) / b]
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            roots = [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]

    lo, hi = min(m1, m2), max(m1, m2)
    between = [r for r in roots if lo <= r <= hi]
    if between:
        return float(between[0])
    if roots:
        mid = 0.5 * (m1 + m2)
        return float(min(roots, key=lambda r: abs(r - mid)))

    # Last resort: midpoint between two component means.
    return float(0.5 * (m1 + m2))


def detect_unknowns_with_energy(model, data, batch_size, device, temperature):
    """Mark samples as unknown based on GMM posterior P(high-energy component | energy).

    Unknown condition: posterior >= args.unknown_posterior_threshold.
    If the fraction of known samples would fall below args.min_known_ratio, the
    threshold is relaxed so that only the top (1 - min_known_ratio) fraction by
    energy are kept unknown.
    """
    model.eval()
    all_energy = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.as_tensor(data[i : i + batch_size]).to(device)
            logits, _ = model(batch)
            energy = compute_energy(logits, temperature=temperature)
            all_energy.append(energy.cpu().numpy())

    energies = np.concatenate(all_energy)

    # Fit 2-component GMM and identify the high-energy (unknown) component.
    gmm = GaussianMixture(n_components=2, covariance_type="full", tol=1e-6, random_state=args.fix_seed)
    gmm.fit(energies.reshape(-1, 1))
    high_idx = int(np.argmax(gmm.means_.reshape(-1)))  # component with larger mean = unknown

    # Posterior P(unknown | energy) for every sample.
    posteriors = gmm.predict_proba(energies.reshape(-1, 1))[:, high_idx]
    unknown_mask = posteriors >= args.unknown_posterior_threshold

    # Enforce min_known_ratio: if too few samples remain as known, fall back to
    # keeping only the top-(1-min_known_ratio) highest-energy samples as unknown.
    if args.min_known_ratio > 0.0:
        min_known = int(np.ceil(args.min_known_ratio * len(energies)))
        if int((~unknown_mask).sum()) < min_known:
            n_unknown_max = max(0, len(energies) - min_known)
            sorted_asc = np.argsort(energies)
            unknown_mask = np.zeros(len(energies), dtype=bool)
            if n_unknown_max > 0:
                unknown_mask[sorted_asc[-n_unknown_max:]] = True

    known_mask = ~unknown_mask
    return known_mask, unknown_mask, energies


def evaluate_open_set_metrics(model, data, labels, unknown_mask, unknown_label, device, eval_metrics):
    """Evaluate metrics requested by --eval_metrics with open-set unknown assignment."""
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            batch = torch.as_tensor(data[i : i + args.batch_size]).to(device)
            outputs, _ = model(batch)
            preds = outputs.argmax(dim=1)
            preds_list.append(preds.cpu().numpy())

    raw_preds = np.concatenate(preds_list)
    all_labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)

    full_preds = raw_preds.copy()
    full_preds[np.asarray(unknown_mask, dtype=bool)] = int(unknown_label)

    metrics = {}
    known_gt_mask = all_labels != int(unknown_label)
    for metric_name in eval_metrics:
        if metric_name == "Closed-F1":
            if known_gt_mask.any():
                metrics[metric_name] = float(
                    evaluator.measurement(
                        all_labels[known_gt_mask],
                        raw_preds[known_gt_mask],
                        [metric_name],
                    ).get(metric_name, float("nan"))
                )
            else:
                metrics[metric_name] = float("nan")
        else:
            metrics[metric_name] = float(
                evaluator.measurement(all_labels, full_preds, [metric_name]).get(metric_name, float("nan"))
            )

    return metrics


def compute_gmm_probabilities(model, loader, device):
    """Use a two-component GMM on entropy to find clean pseudo labels."""
    entropies, predictions = [], []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            preds = outputs.argmax(dim=1)
            entropies.append(compute_softmax_entropy(outputs).cpu().numpy())
            predictions.append(preds.cpu().numpy())

    if len(entropies) == 0:
        return np.array([]), torch.tensor([], dtype=torch.int64)

    entropies = np.concatenate(entropies).flatten()
    predictions = np.concatenate(predictions).flatten()

    e_min, e_max = entropies.min(), entropies.max()
    if e_max - e_min > 1e-8:
        entropies = (entropies - e_min) / (e_max - e_min)
    entropies = entropies.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, tol=1e-6)
    gmm.fit(entropies)
    probs = gmm.predict_proba(entropies)
    low_idx = np.argmin(gmm.means_.flatten())
    return probs[:, low_idx], torch.tensor(predictions, dtype=torch.int64)


def make_pseudo_loader(clean_probs, data, labels, threshold):
    """Build a loader from confident pseudo-labelled target-known samples."""
    if len(clean_probs) == 0:
        return None

    mask = clean_probs >= threshold
    if int(mask.sum()) == 0:
        return None

    selected_x = torch.as_tensor(data[mask], dtype=torch.float32)
    selected_y = torch.as_tensor(labels[mask], dtype=torch.int64)
    return data_processor.load_iter(selected_x, selected_y, args.batch_size, True, args.num_workers)


def adapt_model(backbone, train_data, train_labels, all_test_data, all_test_labels, device):
    """Adapt the backbone with source supervision, pseudo labels, MMD and unknown energy margin."""
    optimizer = torch.optim.Adam(backbone.parameters(), lr=args.adapt_lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    origin_loader = data_processor.load_iter(train_data, train_labels, args.batch_size, True, args.num_workers)
    origin_iter = iter(origin_loader)

    if isinstance(all_test_labels, torch.Tensor):
        all_test_labels_np = all_test_labels.detach().cpu().numpy()
    else:
        all_test_labels_np = np.asarray(all_test_labels)

    def rebuild_target_loaders():
        """Rebuild target loaders from the current known/unknown split."""
        nonlocal known_mask, unknown_mask

        known_test_data = all_test_data[known_mask]
        known_test_labels = all_test_labels_np[known_mask]
        unknown_test_data = all_test_data[unknown_mask]

        print_energy_detection_stats(
            known_mask, all_test_labels_np, args.unknown_label, known_test_data.shape[0], unknown_test_data.shape[0]
        )

        if known_test_data.shape[0] > 0:
            known_test_x = torch.as_tensor(known_test_data, dtype=torch.float32)
            known_test_y = torch.as_tensor(known_test_labels, dtype=torch.int64)
            adapt_loader = data_processor.load_iter(
                known_test_x,
                torch.zeros(known_test_x.shape[0], dtype=torch.int64),
                args.batch_size,
                True,
                args.num_workers,
            )
            known_test_loader = data_processor.load_iter(
                known_test_x,
                known_test_y,
                args.batch_size,
                False,
                args.num_workers,
            )
        else:
            adapt_loader = None
            known_test_loader = None

        if unknown_test_data.shape[0] > 1:
            unk_tensor = torch.as_tensor(unknown_test_data, dtype=torch.float32)
            unk_dataset = torch.utils.data.TensorDataset(unk_tensor)
            unk_loader = torch.utils.data.DataLoader(
                unk_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers,
            )
        else:
            unk_loader = None

        return adapt_loader, known_test_loader, unk_loader

    # evaluation metric
    monitor_metric = args.eval_metrics[0]
    best_f1, best_epoch = 0.0, 0

    for epoch in range(args.adapt_epochs):
        # Refresh known/unknown split by energy and rebuild loaders periodically.
        if epoch % args.unknown_refresh == 0:
            known_mask, unknown_mask, _ = detect_unknowns_with_energy(
                backbone,
                all_test_data,
                args.batch_size,
                device,
                args.energy_temperature,
            )
            adapt_loader, known_test_loader, unk_loader = rebuild_target_loaders()
            unk_iter = iter(unk_loader) if unk_loader is not None else None
        # Refresh pseudo labels by GMM on known samples periodically.
        if epoch % args.gmm_refresh == 0:
            known_test_data = all_test_data[known_mask]
            known_test_labels = all_test_labels_np[known_mask]
            clean_probs, pseudo_labels = compute_gmm_probabilities(backbone, known_test_loader, device)
            print_gmm_pseudo_stats(
                clean_probs,
                pseudo_labels,
                known_test_labels,
                args.gmm_threshold,
            )
            pseudo_loader = make_pseudo_loader(clean_probs, known_test_data, pseudo_labels, args.gmm_threshold)
            pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None

        backbone.train()
        loss_cls, loss_mmd, loss_pse, loss_ent, loss_energy, total_n = 0, 0, 0, 0, 0, 0

        if adapt_loader is not None:
            batch_bar = tqdm(
                adapt_loader,
                desc=f"Epoch {epoch + 1:03d}/{args.adapt_epochs:03d}",
                unit="batch",
                leave=False,
            )
            for adapt_batch in batch_bar:
                try:
                    origin_batch = next(origin_iter)
                except StopIteration:
                    origin_iter = iter(origin_loader)
                    origin_batch = next(origin_iter)

                adapt_x = adapt_batch[0].to(device)
                origin_x = origin_batch[0].to(device)
                origin_y = origin_batch[1].to(device)

                if pseudo_iter is not None:
                    try:
                        pseudo_batch = next(pseudo_iter)
                    except StopIteration:
                        pseudo_iter = iter(pseudo_loader)
                        pseudo_batch = next(pseudo_iter)
                    pseudo_x = pseudo_batch[0].to(device)
                    pseudo_y = pseudo_batch[1].to(device)
                else:
                    pseudo_x, pseudo_y = None, None

                if unk_iter is not None:
                    try:
                        unk_batch = next(unk_iter)[0].to(device)
                    except StopIteration:
                        unk_iter = iter(unk_loader)
                        unk_batch = next(unk_iter)[0].to(device)
                    if unk_batch.size(0) <= 1:
                        unk_batch = None
                else:
                    unk_batch = None

                optimizer.zero_grad()
                origin_out, origin_feat = backbone(origin_x)
                adapt_out, adapt_feat = backbone(adapt_x)

                # classification loss
                known_mask_b = origin_y != args.unknown_label
                unknown_mask_b = origin_y == args.unknown_label
                classification_loss = ce_loss_fn(origin_out[known_mask_b], origin_y[known_mask_b])

                # mmd loss: align only known-class source features with target known features
                known_src_feat = origin_feat[known_mask_b]
                if known_src_feat.size(0) > 0:
                    mmd_loss = calculate_mmd_loss(known_src_feat, adapt_feat)
                else:
                    mmd_loss = torch.tensor(0.0, device=device)

                # entropy loss
                softmax_out = F.softmax(adapt_out, dim=-1)
                mean_softmax = softmax_out.mean(dim=0)
                entropy_loss = compute_softmax_entropy(adapt_out).mean(0) + 0 * torch.sum(
                    mean_softmax * torch.log(mean_softmax + 1e-5)
                )

                # pseudo label loss
                pseudo_out, _ = backbone(pseudo_x)
                pseudo_loss = ce_loss_fn(pseudo_out, pseudo_y)

                # energy loss
                origin_energy = compute_energy(origin_out, temperature=1)
                m_in = args.energy_margin_in
                m_out = args.energy_margin_out
                energy_loss_in = torch.relu(origin_energy[known_mask_b] - m_in).mean()
                if unknown_mask_b.any():
                    # Keep source-domain unknowns (if any) above m_out.
                    energy_loss_out_src = torch.relu(m_out - origin_energy[unknown_mask_b]).mean()
                else:
                    energy_loss_out_src = torch.tensor(0.0, device=device)

                if unk_batch is not None:
                    # Force target-domain unknown samples from unk_loader to have high energy.
                    unk_out, _ = backbone(unk_batch)
                    unk_energy = compute_energy(unk_out, temperature=1)
                    energy_loss_out_tgt = torch.relu(m_out - unk_energy).mean()
                else:
                    energy_loss_out_tgt = torch.tensor(0.0, device=device)

                energy_loss = energy_loss_in + energy_loss_out_src + 0.1 * energy_loss_out_tgt

                # total loss
                total_loss = classification_loss + pseudo_loss + 10 * mmd_loss + entropy_loss + 0.5 * energy_loss
                total_loss.backward()
                optimizer.step()

                bs = origin_out.size(0)
                loss_cls += classification_loss.item() * bs
                loss_mmd += mmd_loss.item() * bs
                loss_ent += entropy_loss.item() * bs
                loss_pse += pseudo_loss.item() * bs
                loss_energy += energy_loss.item() * bs
                total_n += bs

                batch_bar.set_postfix(
                    cls=f"{classification_loss.item():.3f}",
                    pse=f"{pseudo_loss.item():.3f}",
                    mmd=f"{mmd_loss.item():.3f}",
                    ent=f"{entropy_loss.item():.3f}",
                    eng=f"{energy_loss.item():.3f}",
                )

        ep_metrics = evaluate_open_set_metrics(
            backbone,
            all_test_data,
            all_test_labels_np,
            unknown_mask,
            args.unknown_label,
            device,
            args.eval_metrics,
        )
        monitored_score = float(ep_metrics.get(monitor_metric, float("nan")))

        loss_cls_ep = loss_cls / max(total_n, 1)
        loss_pse_ep = loss_pse / max(total_n, 1)
        loss_ent_ep = loss_ent / max(total_n, 1)
        loss_mmd_ep = loss_mmd / max(total_n, 1)
        loss_energy_ep = loss_energy / max(total_n, 1)

        metric_text = ", ".join([f"{m}: {float(ep_metrics.get(m, float('nan'))):.4f}" for m in args.eval_metrics])
        loss_text = ", ".join(
            [
                f"cls: {loss_cls_ep:.4f}",
                f"pse: {loss_pse_ep:.4f}",
                f"ent: {loss_ent_ep:.4f}",
                f"mmd: {loss_mmd_ep:.4f}",
                f"energy: {loss_energy_ep:.4f}",
            ]
        )

        print(f" Epoch {epoch + 1:03d} | {metric_text} | {loss_text}")

        if monitored_score > best_f1:
            best_f1 = monitored_score
            best_epoch = epoch

    final_n_known = int(known_mask.sum())
    final_n_unknown = int(unknown_mask.sum())
    return best_f1, best_epoch, unknown_mask, final_n_known, final_n_unknown


def main():
    # Fall back to CPU if CUDA is requested but unavailable.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset_path = os.path.join("./datasets", args.dataset)
    assert os.path.exists(dataset_path), f"Dataset path not found: {dataset_path}"
    log_path = os.path.join(args.log_path, args.dataset, args.model)
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    # Load source and target data once for the full experiment.
    train_data, train_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.train_file}.npz"), args.feature, args.seq_len, args.num_tabs
    )
    valid_data, valid_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.valid_file}.npz"), args.feature, args.seq_len, args.num_tabs
    )
    test_data, test_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.test_file}.npz"), args.feature, args.seq_len, args.num_tabs
    )

    train_labels_np = train_labels.cpu().numpy() if isinstance(train_labels, torch.Tensor) else np.asarray(train_labels)
    # Compute num_source_classes from known labels only; train_data keeps all samples (including unknowns).
    known_train_labels = train_labels_np[train_labels_np != int(args.unknown_label)]
    if known_train_labels.size == 0:
        raise ValueError(
            f"No known-class samples left after excluding unknown label {args.unknown_label} in train set."
        )

    num_source_classes = int(known_train_labels.max()) + 1
    assert num_source_classes == len(np.unique(known_train_labels)), "Known train labels are not continuous."
    unknown_label = int(args.unknown_label)

    print(f"\n{'=' * 20} Configuration {'=' * 20}")
    print(f"Dataset: {args.dataset}, Model: {args.model}, Device: {device}")
    print(f"Train: {train_data.shape}, Valid: {valid_data.shape}, Test: {test_data.shape}")
    print(f"Source classes: {num_source_classes}, Unknown label: {unknown_label}")
    print(f"Unknown refresh every {args.unknown_refresh} epochs")
    print(f"{'=' * 55}\n")

    # Start adaptation from the source-pretrained checkpoint.
    backbone = models.DF(num_source_classes) if args.model == "DF" else eval(f"models.{args.model}")(num_source_classes)
    ckp_file = os.path.join(ckp_path, f"{args.load_name}.pth")
    if os.path.exists(ckp_file):
        backbone.load_state_dict(torch.load(ckp_file, map_location="cpu"))
        print(f"Loaded pretrained backbone from {ckp_file}")
    backbone.to(device)

    test_labels_np = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else np.asarray(test_labels)
    gt_unknown_mask = test_labels_np == unknown_label
    gt_known_mask = ~gt_unknown_mask

    known_energy_info = summarize_energy_distribution(
        backbone,
        test_data[gt_known_mask],
        args.batch_size,
        device,
        temperature=args.energy_temperature,
    )
    unknown_energy_info = summarize_energy_distribution(
        backbone,
        test_data[gt_unknown_mask],
        args.batch_size,
        device,
        temperature=args.energy_temperature,
    )

    print(f"\n{'=' * 20} Energy Distribution (GT Labels) {'=' * 20}")
    print(
        f"[GT-Known] n={known_energy_info['count']}, mean={known_energy_info['mean']:.4f}, "
        f"std={known_energy_info['std']:.4f}, min={known_energy_info['min']:.4f}, "
        f"max={known_energy_info['max']:.4f}"
    )
    print(
        f"[GT-Unknown] n={unknown_energy_info['count']}, mean={unknown_energy_info['mean']:.4f}, "
        f"std={unknown_energy_info['std']:.4f}, min={unknown_energy_info['min']:.4f}, "
        f"max={unknown_energy_info['max']:.4f}"
    )
    print(f"{'=' * 66}\n")

    print(f"\n{'=' * 20} Step 1: Adapt Backbone {'=' * 20}")
    best_f1, best_epoch, unknown_mask, n_known, n_unknown = adapt_model(
        backbone,
        train_data,
        train_labels,
        test_data,
        test_labels,
        device,
    )
    print(f"Adaptation done. Best F1: {best_f1:.4f} at epoch {best_epoch + 1}")

    print(f"\n{'=' * 20} Step 2: Final Evaluation {'=' * 20}")
    monitor_metric = args.eval_metrics[0]
    final_metrics = evaluate_open_set_metrics(
        backbone,
        test_data,
        test_labels,
        unknown_mask,
        unknown_label,
        device,
        args.eval_metrics,
    )

    final_result = {metric: float(final_metrics.get(metric, float("nan"))) for metric in args.eval_metrics}
    final_result["best_adapt_metric"] = monitor_metric
    final_result["best_adapt_score"] = float(best_f1)
    final_result["best_adapt_epoch"] = int(best_epoch)
    final_result["n_known"] = int(n_known)
    final_result["n_unknown"] = int(n_unknown)

    final_metric_text = ", ".join([f"{m}: {float(final_result[m]):.4f}" for m in args.eval_metrics])
    print(f"Final metrics | {final_metric_text}")
    print(f"{'=' * 55}\n")

    model_path = os.path.join(ckp_path, f"{args.model_save_name}.pth")
    torch.save(backbone.state_dict(), model_path)
    print(f"Adapted model saved to {model_path}")

    output_file = os.path.join(log_path, f"{args.result_file}.json")
    with open(output_file, "w") as f:
        json.dump(final_result, f, indent=4)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
