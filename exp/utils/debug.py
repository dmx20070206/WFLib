import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def print_gmm_pseudo_stats(clean_probs, pseudo_labels, gt_labels, threshold, prefix=""):
    clean_probs = _to_numpy(clean_probs).reshape(-1)
    pseudo_labels = _to_numpy(pseudo_labels).reshape(-1)
    gt_labels = _to_numpy(gt_labels).reshape(-1)

    if gt_labels.size == 0:
        print(f"[GMM]{prefix} no known-target samples")
        return

    overall_acc = float((pseudo_labels == gt_labels).mean())
    selected_mask = clean_probs >= float(threshold)
    selected_count = int(selected_mask.sum())
    total_count = int(clean_probs.shape[0])
    selected_ratio = selected_count / max(total_count, 1)

    if selected_count > 0:
        selected_acc = float((pseudo_labels[selected_mask] == gt_labels[selected_mask]).mean())
    else:
        selected_acc = 0.0

    print(
        f"[GMM]{prefix} overall_acc={overall_acc:.4f}, "
        f"selected={selected_count}/{total_count} ({selected_ratio:.4f}), "
        f"selected_acc={selected_acc:.4f}, threshold={threshold:.4f}"
    )


def summarize_energy_distribution(
    model,
    samples,
    batch_size,
    device,
    temperature=1.0,
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    hist_bins=10,
):
    """
    Compute energy-score distribution summary for a sample set.

    Energy is defined as: E(x) = -T * logsumexp(logits / T).
    """
    samples = np.asarray(samples)
    q_levels = tuple(float(q) for q in quantiles)

    if samples.shape[0] == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "quantiles": {f"q{int(q * 100):02d}": float("nan") for q in q_levels},
            "histogram": {"counts": [], "bin_edges": []},
        }

    model.eval()
    energy_chunks = []
    with torch.no_grad():
        for i in range(0, samples.shape[0], batch_size):
            batch = torch.as_tensor(samples[i : i + batch_size], dtype=torch.float32, device=device)
            logits, _ = model(batch)
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
            energy_chunks.append(energy.detach().cpu().numpy())

    energies = np.concatenate(energy_chunks, axis=0)
    hist_counts, hist_edges = np.histogram(energies, bins=int(hist_bins))

    return {
        "count": int(energies.shape[0]),
        "mean": float(np.mean(energies)),
        "std": float(np.std(energies)),
        "min": float(np.min(energies)),
        "max": float(np.max(energies)),
        "quantiles": {f"q{int(q * 100):02d}": float(np.quantile(energies, q)) for q in q_levels},
        "histogram": {
            "counts": hist_counts.astype(int).tolist(),
            "bin_edges": hist_edges.astype(float).tolist(),
        },
    }


def print_energy_detection_stats(known_mask, labels, unknown_label, ksize, unksize):
    """Print binary known-vs-unknown detection metrics."""
    known_mask = np.asarray(known_mask, dtype=bool)
    labels = np.asarray(labels).reshape(-1)

    pred_unknown = ~known_mask
    gt_unknown = labels == int(unknown_label)

    unknown_recall = float(pred_unknown[gt_unknown].mean()) if gt_unknown.any() else 0.0
    known_recall = float((~pred_unknown)[~gt_unknown].mean()) if (~gt_unknown).any() else 0.0

    print(f"[Unknown] ({ksize}) known_recall={known_recall:.4f}, ({unksize}) unknown_recall={unknown_recall:.4f}")
