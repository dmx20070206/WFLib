import os
import sys
import json
import random
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm
from WFlib import models
from WFlib.tools import data_processor, model_utils
from WFlib.tools.debug import *

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


# --- Arguments ----------------------------------------------------------------

parser = argparse.ArgumentParser(description="Proteus Open-Set Adaptation")

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--num_tabs", type=int, default=1)

parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--extra_train_file", type=str, default=None)
parser.add_argument("--tune_file", type=str, default=None)
parser.add_argument("--extra_tune_file", type=str, default=None)
parser.add_argument("--test_file", type=str, required=True)
parser.add_argument("--extra_test_file", type=str, default=None)
parser.add_argument("--feature", type=str, default="DIR")
parser.add_argument("--seq_len", type=int, default=5000)

parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--adapt_epochs", type=int, default=100)
parser.add_argument("--adapt_lr", type=float, default=1e-4)
parser.add_argument("--split_refresh", type=int, default=5)
parser.add_argument("--pseudo_refresh", type=int, default=5)
parser.add_argument("--pseudo_threshold", type=float, default=0.6)

parser.add_argument("--energy_temperature", type=float, default=1.0)
parser.add_argument("--tau_pct", type=float, default=99.0)
parser.add_argument("--tau_ema", type=float, default=0.99)
parser.add_argument("--energy_m_in", type=float, default=-12.0)
parser.add_argument("--energy_m_out", type=float, default=-2.0)
parser.add_argument("--energy_loss_weight", type=float, default=0.2)

parser.add_argument("--eval_metrics", nargs="+", required=True, type=str)
parser.add_argument("--log_path", type=str, default="./logs/")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
parser.add_argument("--load_name", type=str, default="base")
parser.add_argument("--result_file", type=str, default="result")
parser.add_argument("--model_save_name", type=str, default="proteus")
parser.add_argument("--tune_unknown_ratio", type=float, default=2.0)
parser.add_argument("--tune_known_keep_ratio", type=float, default=1.0)
parser.add_argument("--fix_seed", type=int, default=20070206)

args = parser.parse_args()

random.seed(args.fix_seed)
torch.manual_seed(args.fix_seed)
np.random.seed(args.fix_seed)


# --- Utilities ----------------------------------------------------------------


def gaussian_kernel(source, target):
    n = source.size(0) + target.size(0)
    combined = torch.cat([source, target], dim=0)
    l2 = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(2)
    bw = l2.sum() / (n**2 - n)
    return torch.exp(-l2 / bw)


def mmd_loss(src_feat, tgt_feat):
    bs = min(src_feat.size(0), tgt_feat.size(0))
    src_feat, tgt_feat = src_feat[:bs], tgt_feat[:bs]
    k = gaussian_kernel(src_feat, tgt_feat)
    return (k[:bs, :bs] + k[bs:, bs:] - k[:bs, bs:] - k[bs:, :bs]).mean()


def softmax_entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def energy_score(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=1)


def energy_weights(energies, tau_center, gamma):
    return torch.sigmoid(-(energies - tau_center) / gamma)


def infinite_iter(loader):
    while True:
        yield from loader


def load_and_merge(file_list):
    all_data, all_labels = [], []
    for f in file_list:
        name = f if f.endswith(".npz") else f"{f}.npz"
        data, labels = data_processor.load_data(name, args.feature, args.seq_len, args.num_tabs)
        all_data.append(data)
        all_labels.append(labels)
    return torch.cat(all_data), torch.cat(all_labels)


def infer_unknown_label(labels):
    return int(labels.max().item())


def rebalance_tune_data(data, labels, unknown_label):
    if args.tune_unknown_ratio <= 0:
        raise ValueError("tune_unknown_ratio must be positive")

    if not (0 < args.tune_known_keep_ratio <= 1):
        raise ValueError("tune_known_keep_ratio must be in the range (0, 1]")

    if labels.ndim != 1:
        raise ValueError("tune_unknown_ratio only supports 1D labels")

    known_idx = torch.where(labels != unknown_label)[0]
    unknown_idx = torch.where(labels == unknown_label)[0]
    known_count = int(known_idx.numel())
    unknown_count = int(unknown_idx.numel())

    if known_count == 0:
        print(
            f"Tune ratio skipped: known={known_count}, unknown={unknown_count}, "
            f"requested 1:{args.tune_unknown_ratio:g}"
        )
        return data, labels

    if unknown_count == 0:
        known_selected_idx = known_idx
        unknown_selected_idx = unknown_idx
        ratio_known = known_count
        ratio_unknown = unknown_count
        print(
            f"Tune ratio skipped: known={known_count}, unknown={unknown_count}, "
            f"requested 1:{args.tune_unknown_ratio:g}"
        )
    else:
        keep_known = min(known_count, int(unknown_count / args.tune_unknown_ratio))
        keep_unknown = min(unknown_count, int(round(keep_known * args.tune_unknown_ratio)))

        if keep_known == 0 or keep_unknown == 0:
            keep_known = min(known_count, 1)
            keep_unknown = min(unknown_count, max(1, int(round(keep_known * args.tune_unknown_ratio))))

        known_perm = torch.randperm(known_count)[:keep_known]
        unknown_perm = torch.randperm(unknown_count)[:keep_unknown]
        known_selected_idx = known_idx[known_perm]
        unknown_selected_idx = unknown_idx[unknown_perm]
        ratio_known = int(known_selected_idx.numel())
        ratio_unknown = int(unknown_selected_idx.numel())

    final_known = ratio_known
    final_unknown = ratio_unknown
    if args.tune_known_keep_ratio < 1.0:
        if ratio_known > 0:
            final_known = max(1, int(ratio_known * args.tune_known_keep_ratio))
            known_selected_idx = known_selected_idx[torch.randperm(ratio_known)[:final_known]]
        if ratio_unknown > 0:
            final_unknown = max(1, int(ratio_unknown * args.tune_known_keep_ratio))
            unknown_selected_idx = unknown_selected_idx[torch.randperm(ratio_unknown)[:final_unknown]]

    selected_idx = torch.cat([known_selected_idx, unknown_selected_idx])
    selected_idx = selected_idx[torch.randperm(selected_idx.numel())]

    final_known = int(known_selected_idx.numel())
    final_unknown = int(unknown_selected_idx.numel())
    actual_ratio = final_unknown / final_known if final_known > 0 else float("inf")
    print(
        f"Tune sampling: requested ratio 1:{args.tune_unknown_ratio:g}, "
        f"known keep={args.tune_known_keep_ratio:g}, "
        f"ratio-stage known={ratio_known}, unknown={ratio_unknown}, "
        f"final known={final_known}, unknown={final_unknown} (actual 1:{actual_ratio:.2f})"
    )
    return data[selected_idx], labels[selected_idx]


# --- Energy Splitting ---------------------------------------------------------


def collect_energies(model, data, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            batch = torch.as_tensor(data[i : i + args.batch_size]).to(device)
            logits, _ = model(batch)
            chunks.append(energy_score(logits, args.energy_temperature).cpu().numpy())
    return np.concatenate(chunks)


def compute_energy_threshold(model, source_data, device):
    energies = collect_energies(model, source_data, device)
    return float(np.percentile(energies, args.tau_pct))


def predict_unknown_mask(model, eval_data, tau, device, eval_labels=None):
    known_mask = collect_energies(model, eval_data, device) < tau
    if eval_labels is not None:
        eval_labels_np = np.asarray(eval_labels)
        print_energy_detection_stats(known_mask, eval_labels_np, int(eval_labels_np.max()), tau)
    return ~known_mask


# --- Pseudo Labels ------------------------------------------------------------


def gmm_clean_probs(model, data_loader, device):
    all_ent, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits, _ = model(batch[0].to(device))
            all_ent.append(softmax_entropy(logits).cpu().numpy())
            all_pred.append(logits.argmax(1).cpu().numpy())
    ent = np.concatenate(all_ent).flatten()
    pred = np.concatenate(all_pred).flatten()
    ent = (ent - ent.min()) / (ent.max() - ent.min())
    gmm = GaussianMixture(n_components=2, tol=1e-6)
    gmm.fit(ent.reshape(-1, 1))
    probs = gmm.predict_proba(ent.reshape(-1, 1))
    clean_idx = np.argmin(gmm.means_.flatten())
    return probs[:, clean_idx], torch.tensor(pred, dtype=torch.int64)


def make_pseudo_loader(probs, data, predictions, threshold):
    mask = probs >= threshold
    inputs = torch.as_tensor(data[mask], dtype=torch.float32)
    labels = predictions[mask]
    return data_processor.load_iter(inputs, labels, args.batch_size, True, args.num_workers)


# --- Evaluation ---------------------------------------------------------------


def evaluate(model, test_data, test_labels, unknown_mask, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_data), args.batch_size):
            batch = torch.as_tensor(test_data[i : i + args.batch_size]).to(device)
            preds.append(model(batch)[0].argmax(1).cpu().numpy())
    raw_preds = np.concatenate(preds)
    full_preds = raw_preds.copy()
    full_preds[unknown_mask] = infer_unknown_label(test_labels)
    return model_utils.measure_open_set(test_labels, full_preds, args.eval_metrics, args.num_tabs, open_set=True)


def run_single_test(model, test_name, eval_data, eval_labels, source_known_data, device):
    tau = compute_energy_threshold(model, source_known_data, device)
    print(f"\n[{test_name}] Energy threshold: tau={tau:.4f}")
    unknown_mask = predict_unknown_mask(model, eval_data, tau, device, eval_labels=eval_labels)
    metrics = evaluate(model, eval_data, np.asarray(eval_labels), unknown_mask, device)
    print(f"[{test_name}] {', '.join(f'{k}: {v:.4f}' for k, v in metrics.items())}")
    return metrics, tau


# --- Adaptation ---------------------------------------------------------------


def adapt(backbone, train_data, train_labels, tune_data, tune_labels, tau, device):
    optimizer = torch.optim.Adam(backbone.parameters(), lr=args.adapt_lr)
    ce = nn.CrossEntropyLoss()
    tau_center = tau
    tune_labels_np = np.asarray(tune_labels)
    train_unknown_label = infer_unknown_label(train_labels)
    tune_unknown_label = infer_unknown_label(tune_labels)

    src_iter = infinite_iter(
        data_processor.load_iter(train_data, train_labels, args.batch_size, True, args.num_workers)
    )
    # tune_data: unlabeled target domain — used for adaptation only
    tgt_loader = data_processor.load_iter(
        torch.as_tensor(tune_data, dtype=torch.float32),
        torch.zeros(len(tune_data), dtype=torch.int64),
        args.batch_size,
        True,
        args.num_workers,
    )

    known_mask = unknown_mask = None
    known_iter = pseudo_iter = None
    known_data = None
    best_metrics = {}

    for epoch in range(args.adapt_epochs):

        # Refresh energy split
        if epoch % args.split_refresh == 0:
            src_known_mask = train_labels != train_unknown_label
            tau_new = compute_energy_threshold(backbone, train_data[src_known_mask], device)
            tau = args.tau_ema * tau + (1.0 - args.tau_ema) * tau_new

            known_mask = collect_energies(backbone, tune_data, device) < tau
            unknown_mask = ~known_mask

            tau_center = tau

            print_energy_detection_stats(known_mask, tune_labels_np, tune_unknown_label, tau)
            known_data = tune_data[known_mask]
            known_iter = infinite_iter(
                data_processor.load_iter(
                    torch.as_tensor(known_data, dtype=torch.float32),
                    torch.zeros(len(known_data), dtype=torch.int64),
                    args.batch_size,
                    True,
                    args.num_workers,
                )
            )

        # Refresh pseudo labels
        if epoch % args.pseudo_refresh == 0:
            tmp_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.as_tensor(known_data, dtype=torch.float32),
                    torch.zeros(len(known_data), dtype=torch.int64),
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            confidences, predictions = gmm_clean_probs(backbone, tmp_loader, device)
            pseudo_loader = make_pseudo_loader(confidences, known_data, predictions, args.pseudo_threshold)
            print_gmm_pseudo_stats(confidences, predictions.numpy(), tune_labels_np[known_mask], args.pseudo_threshold)
            pseudo_iter = infinite_iter(pseudo_loader)

        # Train one epoch
        backbone.train()
        losses = {"cls": 0, "mmd": 0, "pse": 0, "ent": 0, "eng": 0}
        n = 0

        for tgt_batch in tqdm(
            tgt_loader, desc=f"Epoch {epoch+1:03d}/{args.adapt_epochs}", leave=False, dynamic_ncols=True
        ):
            src_batch = next(src_iter)
            src_x, src_y = src_batch[0].to(device), src_batch[1].to(device)
            kn_x = next(known_iter)[0].to(device)
            tgt_x = tgt_batch[0].to(device)

            optimizer.zero_grad()
            src_out, src_feat = backbone(src_x)
            tgt_out, _ = backbone(tgt_x)
            kn_out, kn_feat = backbone(kn_x)

            with torch.no_grad():
                w = energy_weights(energy_score(tgt_out.detach(), args.energy_temperature), tau_center, 3)

            is_known = src_y != train_unknown_label
            is_unknown = ~is_known

            cls = ce(src_out[is_known], src_y[is_known])
            mmd = mmd_loss(src_feat[is_known], kn_feat)

            p = F.softmax(kn_out, dim=-1)
            ent = softmax_entropy(kn_out).mean() + (p.mean(0) * torch.log(p.mean(0) + 1e-5)).sum()

            ps_x, ps_y = [t.to(device) for t in next(pseudo_iter)]
            pse = ce(backbone(ps_x)[0], ps_y)

            eng_k = torch.relu(energy_score(src_out[is_known]) - args.energy_m_in).mean()
            eng_u = (
                torch.relu(args.energy_m_out - energy_score(src_out[is_unknown])).mean()
                if is_unknown.any()
                else torch.tensor(0.0, device=device)
            )
            tgt_e = energy_score(tgt_out, args.energy_temperature)
            eng_t = (w * torch.relu(tgt_e - args.energy_m_in) + (1 - w) * torch.relu(args.energy_m_out - tgt_e)).mean()
            eng = eng_k + eng_u + 0.2 * eng_t

            w_cls = 1.0
            w_eng = args.energy_loss_weight
            w_mmd = 1.0

            if epoch < args.adapt_epochs // 10:
                w_pse = 0.1
                w_ent = 0.1
            else:
                w_pse = 1.0
                w_ent = 1.0

            total = w_cls * cls + w_mmd * mmd + w_ent * ent + w_pse * pse + w_eng * eng
            total.backward()
            optimizer.step()

            bs = src_out.size(0)
            losses["cls"] += cls.item() * bs
            losses["mmd"] += mmd.item() * bs
            losses["ent"] += ent.item() * bs
            losses["pse"] += pse.item() * bs
            losses["eng"] += eng.item() * bs
            n += bs

        metrics = evaluate(backbone, tune_data, tune_labels_np, unknown_mask, device)
        for k, v in metrics.items():
            if k not in best_metrics or v > best_metrics[k]:
                best_metrics[k] = v
        m_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        l_str = ", ".join(f"{k}: {v / n:.4f}" for k, v in losses.items())
        print(f"  Epoch {epoch+1:03d} | {m_str} | {l_str}")

    return unknown_mask, best_metrics


# --- Entry Point --------------------------------------------------------------


def main():
    device = torch.device(args.device)
    dataset_path = os.path.join("./datasets", args.dataset)
    log_path = os.path.join(args.log_path, args.dataset, args.model)
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    # Load data
    train_files = [os.path.join(dataset_path, args.train_file)] + (
        [os.path.join("./datasets", args.extra_train_file)] if args.extra_train_file else []
    )
    tune_files = [os.path.join(dataset_path, args.tune_file)] + (
        [os.path.join("./datasets", args.extra_tune_file)] if args.extra_tune_file else []
    )
    test_files = [os.path.join(dataset_path, args.test_file)] + (
        [os.path.join("./datasets", args.extra_test_file)] if args.extra_test_file else []
    )

    train_data, train_labels = load_and_merge(train_files)
    tune_data, tune_labels = load_and_merge(tune_files)
    test_data, test_labels = load_and_merge(test_files)
    tune_unknown_label = infer_unknown_label(tune_labels)
    tune_data, tune_labels = rebalance_tune_data(tune_data, tune_labels, tune_unknown_label)

    train_unknown_label = infer_unknown_label(train_labels)
    num_classes = int(train_labels[train_labels != train_unknown_label].max()) + 1
    src_known = train_labels != train_unknown_label
    source_known_data = train_data[src_known]

    print(f"\nDataset: {args.dataset}, Model: {args.model}, Device: {device}")
    print(f"Train: {train_data.shape}, Tune: {tune_data.shape}, Test: {test_data.shape}, Classes: {num_classes}")

    # Load model
    backbone = getattr(models, args.model)(num_classes)
    ckp_file = os.path.join(ckp_path, f"{args.load_name}.pth")
    backbone.load_state_dict(torch.load(ckp_file, map_location="cpu"))
    backbone.to(device)
    print(f"Loaded: {ckp_file}")

    # Test before adaptation
    pre_test_metrics, tau = run_single_test(backbone, "Pre-Adapt Test", test_data, test_labels, source_known_data, device)
    print()

    # Adapt
    _, best_metrics = adapt(backbone, train_data, train_labels, tune_data, tune_labels, tau, device)

    # Test after adaptation
    post_test_metrics, _ = run_single_test(backbone, "Post-Adapt Test", test_data, test_labels, source_known_data, device)

    metrics = {f"pre_{k}": v for k, v in pre_test_metrics.items()}
    metrics.update(post_test_metrics)
    metrics.update({f"post_{k}": v for k, v in post_test_metrics.items()})
    for k, v in best_metrics.items():
        metrics[f"best_{k}"] = v
    print(f"\nFinal: {', '.join(f'{k}: {v:.4f}' for k, v in post_test_metrics.items())}")

    # Save
    torch.save(backbone.state_dict(), os.path.join(ckp_path, f"{args.model_save_name}.pth"))
    with open(os.path.join(log_path, f"{args.result_file}.json"), "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
