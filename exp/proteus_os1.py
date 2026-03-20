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
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
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
parser.add_argument("--model_save_name", type=str, default="proteus")

parser.add_argument("--adapt_epochs", type=int, default=100)
parser.add_argument("--adapt_lr", type=float, default=1e-4)
parser.add_argument("--split_refresh", type=int, default=5, help="Re-split target every N epochs")
parser.add_argument("--pseudo_refresh", type=int, default=5, help="Refresh pseudo labels every N epochs")
parser.add_argument("--pseudo_threshold", type=float, default=0.6, help="Min softmax confidence for pseudo labeling")

parser.add_argument("--energy_temperature", type=float, default=1.0)
parser.add_argument("--tau_low_pct", type=float, default=99.0, help="Percentile of energy distribution for tau_low")
parser.add_argument("--tau_high_pct", type=float, default=99.0, help="Percentile of energy distribution for tau_high")
parser.add_argument(
    "--tau_from_source",
    type=bool,
    default=True,
    help="True: compute tau from source-domain energies; False: compute tau from target-domain energies",
)
parser.add_argument(
    "--energy_gamma",
    type=float,
    default=1.0,
    help="Temperature for energy-to-weight sigmoid mapping; larger = smoother transition",
)

parser.add_argument("--energy_margin_in", type=float, default=-12.0)
parser.add_argument("--energy_margin_out", type=float, default=-2.0)

parser.add_argument("--fix_seed", type=int, default=20070206)
parser.add_argument("--unknown_label", type=int, default=102)

args = parser.parse_args()

random.seed(args.fix_seed)
torch.manual_seed(args.fix_seed)
np.random.seed(args.fix_seed)


# --- Core Utilities -----------------------------------------------------------


def compute_gaussian_kernel(source, target):
    """Gaussian kernel matrix for MMD computation."""
    n = source.size(0) + target.size(0)
    combined = torch.cat([source, target], dim=0)
    l2 = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(2)
    bw = (l2.sum() / (n**2 - n)).clamp(min=1e-5)
    return torch.exp(-l2 / bw)


def calculate_mmd_loss(src_feat, tgt_feat):
    """Maximum Mean Discrepancy between source and target feature distributions."""
    bs = min(src_feat.size(0), tgt_feat.size(0))
    src_feat, tgt_feat = src_feat[:bs], tgt_feat[:bs]
    k = compute_gaussian_kernel(src_feat, tgt_feat)
    return (k[:bs, :bs] + k[bs:, bs:] - k[:bs, bs:] - k[bs:, :bs]).mean()


def compute_softmax_entropy(logits):
    """Per-sample softmax entropy H(p)."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def compute_energy(logits, temperature=1.0):
    """Energy score E(x) = -T * logsumexp(logits / T)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def compute_energy_weights(energies, tau_center, gamma):
    """Map energy scores to known-class confidence weights via sigmoid.

    w_i = 1 / (1 + exp((E_i - tau_center) / gamma))

    Low energy (known-like)  -> w near 1.
    High energy (unknown-like) -> w near 0.
    gamma controls sharpness: larger = smoother, smaller = closer to hard threshold.
    """
    return torch.sigmoid(-(energies - tau_center) / gamma)


def calculate_weighted_mmd_loss(src_feat, tgt_feat, tgt_weights):
    """MMD loss with per-sample weights on the target side.

    Replaces the uniform target empirical average with a weighted average
    guided by tgt_weights (known-class confidence), so unknown-like samples
    contribute less to the alignment objective.
    """
    bs_s = src_feat.size(0)
    w = tgt_weights / (tgt_weights.sum() + 1e-10)  # normalised target weights
    combined = torch.cat([src_feat, tgt_feat], dim=0)
    l2 = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(2)
    n = combined.size(0)
    bw = (l2.sum() / (n**2 - n)).clamp(min=1e-5)
    K = torch.exp(-l2 / bw)
    K_ss = K[:bs_s, :bs_s].mean()
    K_tt = (w.unsqueeze(0) * K[bs_s:, bs_s:] * w.unsqueeze(1)).sum()
    K_st = (K[:bs_s, bs_s:] * w.unsqueeze(0)).mean()
    return K_ss + K_tt - 2 * K_st


def infinite_iter(loader):
    """Infinite iterator that cycles through a DataLoader indefinitely."""
    while True:
        yield from loader


# --- Energy-Based Three-Way Split ---------------------------------------------


def compute_source_energy_thresholds(model, source_data, device):
    """Compute tau_low and tau_high as percentiles of the source energy distribution.
    Uses args.tau_low_pct and args.tau_high_pct.
    """
    model.eval()
    energies = []
    with torch.no_grad():
        for i in range(0, len(source_data), args.batch_size):
            batch = torch.as_tensor(source_data[i : i + args.batch_size]).to(device)
            logits, _ = model(batch)
            energies.append(compute_energy(logits, args.energy_temperature).cpu().numpy())
    energies = np.concatenate(energies)
    tau_low = float(np.percentile(energies, args.tau_low_pct))
    tau_high = float(np.percentile(energies, args.tau_high_pct))
    return tau_low, tau_high


def split_target_by_energy(model, target_data, tau_low, tau_high, device):
    """Split target samples into three partitions using fixed energy thresholds.

    Returns:
        known_mask:   E(x) < tau_low           -- D_t,known
        unknown_mask: E(x) > tau_high          -- D_t,unknown
        gray_mask:    tau_low <= E(x) <= tau_high -- D_t,gray (unused for now)
        energies:     raw energy score per sample
    """
    model.eval()
    energies = []
    with torch.no_grad():
        for i in range(0, len(target_data), args.batch_size):
            batch = torch.as_tensor(target_data[i : i + args.batch_size]).to(device)
            logits, _ = model(batch)
            energies.append(compute_energy(logits, args.energy_temperature).cpu().numpy())
    energies = np.concatenate(energies)
    known_mask = energies < tau_low
    unknown_mask = energies > tau_high
    gray_mask = ~known_mask & ~unknown_mask
    return known_mask, unknown_mask, gray_mask, energies


# --- Pseudo Label Selection ---------------------------------------------------


def compute_gmm_probabilities(model, data_loader, device):
    """Use GMM on entropy distribution to estimate per-sample clean probability."""
    entropies = []
    predictions = []
    with torch.no_grad():
        model.eval()
        for batch in data_loader:
            inputs = batch[0].to(device)
            outputs, _ = model(inputs)
            preds = torch.argsort(outputs, dim=1, descending=True)[:, 0]
            entropies.append(compute_softmax_entropy(outputs).cpu().numpy())
            predictions.append(preds.cpu().numpy())
    entropies = np.concatenate(entropies).flatten()
    predictions = np.concatenate(predictions).flatten()
    entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-10)
    entropies = entropies.reshape(-1, 1)
    predictions = torch.tensor(predictions, dtype=torch.int64)
    gmm = GaussianMixture(n_components=2, tol=1e-6)
    gmm.fit(entropies)
    probabilities = gmm.predict_proba(entropies)
    low_uncertainty_index = np.argmin(gmm.means_.flatten())
    return probabilities[:, low_uncertainty_index], predictions


def create_pseudo_labels(clean_probs, known_data, predictions, threshold, batch_size, num_workers):
    """Build a DataLoader for samples whose GMM clean probability >= threshold."""
    clean_indices = clean_probs >= threshold
    pseudo_inputs = torch.as_tensor(known_data[clean_indices], dtype=torch.float32)
    pseudo_labels = predictions[clean_indices]
    return data_processor.load_iter(pseudo_inputs, pseudo_labels, batch_size, True, num_workers)


# --- Evaluation ---------------------------------------------------------------


def evaluate_open_set(model, test_data, test_labels, unknown_mask, device):
    """Run model inference; assign args.unknown_label to detected-unknown samples."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_data), args.batch_size):
            batch = torch.as_tensor(test_data[i : i + args.batch_size]).to(device)
            preds.append(model(batch)[0].argmax(dim=1).cpu().numpy())
    raw_preds = np.concatenate(preds)
    full_preds = raw_preds.copy()
    full_preds[unknown_mask] = args.unknown_label

    known_gt = test_labels != args.unknown_label
    metrics = {}
    for metric in args.eval_metrics:
        if metric == "Closed-F1":
            metrics[metric] = float(
                evaluator.measurement(test_labels[known_gt], raw_preds[known_gt], [metric]).get(metric, float("nan"))
            )
        else:
            metrics[metric] = float(evaluator.measurement(test_labels, full_preds, [metric]).get(metric, float("nan")))
    return metrics


# --- Adaptation Loop ----------------------------------------------------------


def adapt_model(backbone, train_data, train_labels, test_data, test_labels, tau_low, tau_high, device):
    """Adapt backbone with source CE, pseudo labels, weighted MMD, weighted entropy, and weighted energy margin."""
    optimizer = torch.optim.Adam(backbone.parameters(), lr=args.adapt_lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    tau_center = (tau_low + tau_high) / 2.0

    test_labels_np = np.asarray(test_labels)

    origin_loader = data_processor.load_iter(train_data, train_labels, args.batch_size, True, args.num_workers)
    origin_iter = infinite_iter(origin_loader)

    # Full target loader — all samples; soft weights w_i route each sample's loss contribution.
    adapt_loader = data_processor.load_iter(
        torch.as_tensor(test_data, dtype=torch.float32),
        torch.zeros(len(test_data), dtype=torch.int64),
        args.batch_size,
        True,
        args.num_workers,
    )

    # Hard-threshold split retained only for: (a) eval unknown_mask, (b) GMM pseudo-label source.
    known_mask = unknown_mask = None
    pseudo_loader = pseudo_iter = None
    known_loader = known_iter = None
    known_data = None

    best_score, best_epoch = 0.0, 0
    monitor = args.eval_metrics[0]

    for epoch in range(args.adapt_epochs):

        # Re-split target by hard energy thresholds for eval / pseudo-label source.
        if epoch % args.split_refresh == 0:
            known_mask, unknown_mask, _, _ = split_target_by_energy(backbone, test_data, tau_low, tau_high, device)
            print_energy_detection_stats(
                known_mask, test_labels_np, args.unknown_label, known_mask.sum(), unknown_mask.sum()
            )
            known_data = test_data[known_mask]
            known_x_all = torch.as_tensor(known_data, dtype=torch.float32)
            known_loader = data_processor.load_iter(
                known_x_all,
                torch.zeros(len(known_x_all), dtype=torch.int64),
                args.batch_size,
                True,
                args.num_workers,
            )
            known_iter = infinite_iter(known_loader)

        # Refresh pseudo labels on D_t,known using GMM entropy (hard-selected only, unchanged).
        if epoch % args.pseudo_refresh == 0:
            known_tmp_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.as_tensor(known_data, dtype=torch.float32),
                    torch.zeros(len(known_data), dtype=torch.int64),
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            confidences, predictions = compute_gmm_probabilities(backbone, known_tmp_loader, device)
            pseudo_loader = create_pseudo_labels(
                confidences, known_data, predictions, args.pseudo_threshold, args.batch_size, args.num_workers
            )
            print_gmm_pseudo_stats(confidences, predictions.numpy(), test_labels_np[known_mask], args.pseudo_threshold)
            pseudo_iter = infinite_iter(pseudo_loader)

        # Train one epoch.
        backbone.train()
        loss_cls = loss_mmd = loss_pse = loss_ent = loss_eng = n = 0

        for adapt_batch in tqdm(
            adapt_loader,
            desc=f"Epoch {epoch+1:03d}/{args.adapt_epochs}",
            unit="batch",
            leave=False,
        ):
            origin_batch = next(origin_iter)
            known_batch = next(known_iter)
            adapt_x = adapt_batch[0].to(device)  # all target — for energy margin
            known_x_b = known_batch[0].to(device)  # known target — for MMD + entropy
            origin_x, origin_y = origin_batch[0].to(device), origin_batch[1].to(device)

            optimizer.zero_grad()
            origin_out, origin_feat = backbone(origin_x)
            adapt_out, _ = backbone(adapt_x)  # full target, only logits needed for energy
            known_out, known_feat = backbone(known_x_b)  # known target features + logits

            # Soft weights computed on the full target batch for energy margin routing.
            with torch.no_grad():
                w = compute_energy_weights(
                    compute_energy(adapt_out.detach(), args.energy_temperature), tau_center, args.energy_gamma
                )

            # Classification loss on known source samples.
            src_known = origin_y != args.unknown_label
            src_unknown = ~src_known
            cls_loss = ce_loss_fn(origin_out[src_known], origin_y[src_known])

            # MMD: align source-known features with target-known features only.
            mmd_loss = calculate_mmd_loss(origin_feat[src_known], known_feat)

            # Entropy minimisation on target-known samples only.
            softmax_out = F.softmax(known_out, dim=-1)
            mean_softmax = softmax_out.mean(dim=0)
            ent_loss = compute_softmax_entropy(known_out).mean(0) + torch.sum(
                mean_softmax * torch.log(mean_softmax + 1e-5)
            )

            # Pseudo-label loss on GMM-filtered high-confidence samples (hard threshold preserved).
            pseudo_x, pseudo_y = [t.to(device) for t in next(pseudo_iter)]
            pse_loss = ce_loss_fn(backbone(pseudo_x)[0], pseudo_y)

            # Source energy margin.
            eng_loss_sk = torch.relu(compute_energy(origin_out[src_known]) - args.energy_margin_in).mean()
            eng_loss_suk = (
                torch.relu(args.energy_margin_out - compute_energy(origin_out[src_unknown])).mean()
                if src_unknown.any()
                else torch.tensor(0.0, device=device)
            )

            # Unified weighted energy margin on target.
            # w_i -> 1: push energy down (known margin); w_i -> 0: push energy up (unknown margin).
            adapt_energies = compute_energy(adapt_out, args.energy_temperature)
            eng_loss_target = (
                w * torch.relu(adapt_energies - args.energy_margin_in)
                + (1 - w) * torch.relu(args.energy_margin_out - adapt_energies)
            ).mean()

            eng_loss = 0.5 * eng_loss_sk + 0.5 * eng_loss_suk + 0.1 * eng_loss_target

            w_cls, w_pse, w_mmd, w_ent, w_eng = 1.0, 1.0, 1.0, 1.0, 1.0
            total_loss = w_cls * cls_loss + w_pse * pse_loss + w_mmd * mmd_loss + w_ent * ent_loss + w_eng * eng_loss
            total_loss.backward()
            optimizer.step()

            bs = origin_out.size(0)
            loss_cls += cls_loss.item() * bs
            loss_mmd += mmd_loss.item() * bs
            loss_ent += ent_loss.item() * bs
            loss_pse += pse_loss.item() * bs
            loss_eng += eng_loss.item() * bs
            n += bs

        metrics = evaluate_open_set(backbone, test_data, test_labels_np, unknown_mask, device)
        score = float(metrics.get(monitor, float("nan")))
        if score > best_score:
            best_score, best_epoch = score, epoch

        n = max(n, 1)
        m_str = ", ".join(f"{m}: {metrics.get(m, float('nan')):.4f}" for m in args.eval_metrics)
        l_str = (
            f"cls: {loss_cls/n:.4f}, pse: {loss_pse/n:.4f}, "
            f"ent: {loss_ent/n:.4f}, "
            f"mmd: {loss_mmd/n:.4f}, eng: {loss_eng/n:.4f}"
        )
        print(f" Epoch {epoch+1:03d} | {m_str} | {l_str}")

    return best_score, best_epoch, unknown_mask, int(known_mask.sum()), int(unknown_mask.sum())


# --- Entry Point --------------------------------------------------------------


def main():
    device = torch.device(args.device)

    dataset_path = os.path.join("./datasets", args.dataset)
    log_path = os.path.join(args.log_path, args.dataset, args.model)
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    train_data, train_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.train_file}.npz"),
        args.feature,
        args.seq_len,
        args.num_tabs,
    )
    valid_data, valid_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.valid_file}.npz"),
        args.feature,
        args.seq_len,
        args.num_tabs,
    )
    test_data, test_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.test_file}.npz"),
        args.feature,
        args.seq_len,
        args.num_tabs,
    )

    train_labels_np = np.asarray(train_labels)
    known_src_labels = train_labels_np[train_labels_np != args.unknown_label]
    num_classes = int(known_src_labels.max()) + 1

    print(f"\n{'='*20} Configuration {'='*20}")
    print(f"Dataset: {args.dataset},  Model: {args.model},  Device: {device}")
    print(f"Train: {train_data.shape},  Valid: {valid_data.shape},  Test: {test_data.shape}")
    print(f"Source classes: {num_classes},  Unknown label: {args.unknown_label}")
    print(f"Energy tau percentiles: low={args.tau_low_pct}%,  high={args.tau_high_pct}%")
    print(f"{'='*55}\n")

    # Build and load pretrained backbone.
    backbone = models.DF(num_classes) if args.model == "DF" else eval(f"models.{args.model}")(num_classes)
    ckp_file = os.path.join(ckp_path, f"{args.load_name}.pth")
    if os.path.exists(ckp_file):
        backbone.load_state_dict(torch.load(ckp_file, map_location="cpu"))
        print(f"Loaded pretrained backbone from {ckp_file}")
    backbone.to(device)

    # Summarise pre-adaptation energy distributions using ground-truth labels.
    test_labels_np = np.asarray(test_labels)
    gt_known_mask = test_labels_np != args.unknown_label
    gt_unknown_mask = ~gt_known_mask

    known_info = summarize_energy_distribution(
        backbone, test_data[gt_known_mask], args.batch_size, device, args.energy_temperature
    )
    unknown_info = summarize_energy_distribution(
        backbone, test_data[gt_unknown_mask], args.batch_size, device, args.energy_temperature
    )
    print(f"\n{'='*20} Pre-Adaptation Energy Distribution (GT) {'='*20}")
    print(f"[GT-Known]   n={known_info['count']},  mean={known_info['mean']:.4f},  std={known_info['std']:.4f}")
    print(f"[GT-Unknown] n={unknown_info['count']}, mean={unknown_info['mean']:.4f},  std={unknown_info['std']:.4f}")

    # Compute tau_low and tau_high from source or target domain depending on args.tau_from_source.
    src_known_mask = train_labels_np != args.unknown_label
    _tau_data = train_data[src_known_mask] if args.tau_from_source else test_data
    tau_low, tau_high = compute_source_energy_thresholds(backbone, _tau_data, device)
    _tau_domain = "source" if args.tau_from_source else "target"
    print(f"\nEnergy thresholds from {_tau_domain}: tau_low={tau_low:.4f},  tau_high={tau_high:.4f}")
    print(f"{'='*65}\n")

    print(f"\n{'='*20} Adaptation {'='*20}")
    best_score, best_epoch, unknown_mask, n_known, n_unknown = adapt_model(
        backbone,
        train_data,
        train_labels,
        test_data,
        test_labels,
        tau_low,
        tau_high,
        device,
    )
    print(f"Done.  Best {args.eval_metrics[0]}: {best_score:.4f}  (epoch {best_epoch + 1})")

    print(f"\n{'='*20} Final Evaluation {'='*20}")
    final_metrics = evaluate_open_set(backbone, test_data, test_labels_np, unknown_mask, device)
    final_result = {m: float(final_metrics.get(m, float("nan"))) for m in args.eval_metrics}
    final_result.update(
        {
            "best_adapt_metric": args.eval_metrics[0],
            "best_adapt_score": float(best_score),
            "best_adapt_epoch": int(best_epoch),
            "n_known": int(n_known),
            "n_unknown": int(n_unknown),
        }
    )
    m_str = ", ".join(f"{m}: {final_result[m]:.4f}" for m in args.eval_metrics)
    print(f"Final | {m_str}")
    print(f"{'='*55}\n")

    model_path = os.path.join(ckp_path, f"{args.model_save_name}.pth")
    torch.save(backbone.state_dict(), model_path)
    print(f"Adapted model saved -> {model_path}")

    output_file = os.path.join(log_path, f"{args.result_file}.json")
    with open(output_file, "w") as f:
        json.dump(final_result, f, indent=4)
    print(f"Results saved     -> {output_file}")


if __name__ == "__main__":
    main()
