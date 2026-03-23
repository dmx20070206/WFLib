import os
import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


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

    energies = np.concatenate(energy_chunks, axis=0).reshape(-1)
    energies = energies[np.isfinite(energies)]

    if energies.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "quantiles": {f"q{int(q * 100):02d}": float("nan") for q in q_levels},
            "histogram": {"counts": [], "bin_edges": []},
        }

    bins = max(int(hist_bins), 1)

    try:
        hist_counts, hist_edges = np.histogram(energies, bins=bins)
    except ValueError:
        # Fallback for rare NumPy histogram edge-case broadcasting issues.
        e_min = float(np.min(energies))
        e_max = float(np.max(energies))
        if not np.isfinite(e_min) or not np.isfinite(e_max):
            hist_counts = np.array([], dtype=int)
            hist_edges = np.array([], dtype=float)
        elif e_min == e_max:
            hist_counts = np.zeros(bins, dtype=int)
            hist_counts[0] = int(energies.size)
            hist_edges = np.linspace(e_min - 1e-6, e_max + 1e-6, bins + 1)
        else:
            hist_edges = np.linspace(e_min, e_max, bins + 1)
            indices = np.digitize(energies, hist_edges[1:-1], right=False)
            indices = np.clip(indices, 0, bins - 1)
            hist_counts = np.bincount(indices, minlength=bins).astype(int)

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




def plot_pre_adapt_energy_distribution(
    model,
    known_samples,
    unknown_samples,
    batch_size,
    device,
    save_path,
    temperature=1.0,
    bins=60,
    clip_percentiles=(1.0, 99.0),
):
    """Plot and save pre-adaptation energy distributions with robust x-range clipping.

    clip_percentiles controls visual x-axis span to avoid rare extreme values
    stretching the figure. Values outside the span are clipped to the edges for
    display purposes only.
    """
    if plt is None:
        print("matplotlib is not available; skip pre-adaptation energy plot.")
        return

    def _collect(samples):
        arr = np.asarray(samples)
        if arr.shape[0] == 0:
            return np.array([], dtype=np.float32)

        model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, arr.shape[0], int(batch_size)):
                batch = torch.as_tensor(arr[i : i + int(batch_size)], dtype=torch.float32, device=device)
                logits, _ = model(batch)
                energy = -float(temperature) * torch.logsumexp(logits / float(temperature), dim=1)
                chunks.append(energy.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    known_energies = _collect(known_samples)
    unknown_energies = _collect(unknown_samples)

    all_energies = np.concatenate([known_energies, unknown_energies]) if (known_energies.size + unknown_energies.size) > 0 else np.array([], dtype=np.float32)
    all_energies = all_energies[np.isfinite(all_energies)]

    if all_energies.size == 0:
        print("No valid energies for plotting; skip pre-adaptation energy plot.")
        return

    low_p, high_p = clip_percentiles
    low = float(np.percentile(all_energies, low_p))
    high = float(np.percentile(all_energies, high_p))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(np.min(all_energies))
        high = float(np.max(all_energies) + 1e-6)

    known_plot = np.clip(known_energies, low, high) if known_energies.size > 0 else known_energies
    unknown_plot = np.clip(unknown_energies, low, high) if unknown_energies.size > 0 else unknown_energies

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    if known_plot.size > 0:
        plt.hist(known_plot, bins=int(bins), alpha=0.55, density=True, label="GT-Known", color="#1f77b4")
    if unknown_plot.size > 0:
        plt.hist(unknown_plot, bins=int(bins), alpha=0.55, density=True, label="GT-Unknown", color="#d62728")

    plt.xlim(low, high)
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.title(f"Pre-Adaptation Energy Distribution (GT, clipped {low_p:.1f}-{high_p:.1f} pct)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def print_energy_detection_stats(known_mask, labels, unknown_label, ksize, unksize):
    """Print binary known-vs-unknown detection metrics."""
    known_mask = np.asarray(known_mask, dtype=bool)
    labels = np.asarray(labels).reshape(-1)

    pred_unknown = ~known_mask
    gt_unknown = labels == int(unknown_label)

    unknown_recall = float(pred_unknown[gt_unknown].mean()) if gt_unknown.any() else 0.0
    known_recall = float((~pred_unknown)[~gt_unknown].mean()) if (~gt_unknown).any() else 0.0

    print(f"[Unknown] ({ksize}) known_recall={known_recall:.4f}, ({unksize}) unknown_recall={unknown_recall:.4f}")
