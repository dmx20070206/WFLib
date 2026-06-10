import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from WFlib import models
from WFlib.tools import data_processor

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def parse_common_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--model_file", type=str, required=True, help="Checkpoint file path")
    parser.add_argument("--test_file", type=str, required=True, help="Known-class test file")
    parser.add_argument("--extra_test_file", type=str, required=True, help="Unknown-class test file")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--feature", type=str, default="DIR", help="Feature type")
    parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Data loader workers")
    parser.add_argument("--num_tabs", type=int, default=1, help="Number of tabs")
    parser.add_argument("--energy_temperature", type=float, default=1.0, help="Temperature used by energy score")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_npz_suffix(path_str):
    return path_str if path_str.endswith(".npz") else f"{path_str}.npz"


def resolve_path(path_str):
    candidate = ensure_npz_suffix(path_str)
    if os.path.exists(candidate):
        return candidate

    dataset_candidate = os.path.join("./datasets", candidate)
    if os.path.exists(dataset_candidate):
        return dataset_candidate

    raise FileNotFoundError(f"File not found: {path_str}")


def load_split(path_str, feature, seq_len, num_tabs):
    return data_processor.load_data(resolve_path(path_str), feature, seq_len, num_tabs)


def merge_test_sets(test_file, extra_test_file, feature, seq_len, num_tabs):
    test_x, test_y = load_split(test_file, feature, seq_len, num_tabs)
    extra_x, extra_y = load_split(extra_test_file, feature, seq_len, num_tabs)

    if num_tabs != 1:
        raise ValueError("OOD scripts only support num_tabs=1 labels.")

    num_known_classes = int(test_y.max().item()) + 1
    unknown_label = num_known_classes
    extra_y = torch.full_like(extra_y, unknown_label)

    merged_x = torch.cat([test_x, extra_x], dim=0)
    merged_y = torch.cat([test_y, extra_y], dim=0)
    y_true_binary = np.concatenate(
        [
            np.ones(len(test_y), dtype=np.int64),
            np.zeros(len(extra_y), dtype=np.int64),
        ]
    )
    return merged_x, merged_y, y_true_binary, num_known_classes, unknown_label


def build_model(model_name, num_classes, num_tabs):
    if model_name in ["BAPM", "TMWF"]:
        return getattr(models, model_name)(num_classes, num_tabs)
    return getattr(models, model_name)(num_classes)


def load_model(model_name, model_file, num_classes, num_tabs, device):
    model = build_model(model_name, num_classes, num_tabs)
    state_dict = torch.load(model_file, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def collect_scores(model, data_loader, device, score_fn, unknown_label, plot_score_fn=None):
    all_scores = []
    all_plot_scores = []
    all_binary_labels = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            logits = get_logits(model(batch_x.to(device)))
            probs = F.softmax(logits, dim=1)
            scores = score_fn(logits, probs, unknown_label)
            all_scores.append(scores.cpu())
            if plot_score_fn is None:
                plot_scores = scores
            else:
                plot_scores = plot_score_fn(logits, probs, unknown_label)
            all_plot_scores.append(plot_scores.cpu())
            all_binary_labels.append((batch_y != unknown_label).to(torch.int64).cpu())
    return torch.cat(all_scores).numpy(), torch.cat(all_plot_scores).numpy(), torch.cat(all_binary_labels).numpy()


def infer_dataset_name(test_file):
    normalized = ensure_npz_suffix(test_file).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return parts[0]
    return os.path.splitext(os.path.basename(normalized))[0]


def plot_ood_histogram(scores, y_true_binary, dataset_name, score_tag):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skip OOD histogram plotting.")
        return None

    scores = np.asarray(scores)
    y_true_binary = np.asarray(y_true_binary)
    known_scores = scores[y_true_binary == 1]
    unknown_scores = scores[y_true_binary == 0]

    if known_scores.size == 0 or unknown_scores.size == 0:
        print("Skip OOD histogram plotting because known or unknown scores are empty.")
        return None

    all_scores = np.concatenate([known_scores, unknown_scores])
    low = float(np.percentile(all_scores, 1.0))
    high = float(np.percentile(all_scores, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(np.min(all_scores))
        high = float(np.max(all_scores) + 1e-6)

    bins = 60
    known_plot = np.clip(known_scores, low, high)
    unknown_plot = np.clip(unknown_scores, low, high)

    plot_dir = os.path.join("plots", "ood_detection")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{dataset_name}_{score_tag}.png")

    plt.figure(figsize=(10, 6))
    plt.hist(known_plot, bins=int(bins), alpha=0.55, density=True, label="GT-Known", color="#1f77b4")
    plt.hist(unknown_plot, bins=int(bins), alpha=0.55, density=True, label="GT-Unknown", color="#d62728")

    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.xlim(low, high)

    ticks = np.linspace(low, high, 6)
    tick_labels = [f"{tick:.2f}" for tick in ticks]
    tick_labels[0] = f"< {low:.2f}"
    tick_labels[-1] = f"> {high:.2f}"
    plt.xticks(ticks, tick_labels)

    plt.title(f"{dataset_name}_{score_tag}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def format_table(headers, rows):
    headers = [str(header) for header in headers]
    normalized_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in normalized_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row):
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [border, format_row(headers), border]
    lines.extend(format_row(row) for row in normalized_rows)
    lines.append(border)
    return "\n".join(lines)


def compute_fpr90(y_true_binary, scores):
    y_true_binary = np.asarray(y_true_binary)
    scores = np.asarray(scores)

    known_scores = scores[y_true_binary == 1]
    unknown_scores = scores[y_true_binary == 0]

    if known_scores.size == 0 or unknown_scores.size == 0:
        raise ValueError("FPR@90 requires both known and unknown samples.")

    threshold = float(np.quantile(known_scores, 0.2))
    return float(np.mean(unknown_scores >= threshold))


def compute_eer(y_true_binary, scores):
    y_true_binary = np.asarray(y_true_binary)
    scores = np.asarray(scores)

    if np.unique(y_true_binary).size != 2:
        raise ValueError("EER requires both known and unknown samples.")

    fpr, tpr, _ = roc_curve(y_true_binary, scores)
    fnr = 1.0 - tpr
    diff = fpr - fnr

    exact_matches = np.where(diff == 0)[0]
    if exact_matches.size > 0:
        return float(fpr[exact_matches[0]])

    sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if sign_changes.size > 0:
        idx = int(sign_changes[0])
        diff_left = diff[idx]
        diff_right = diff[idx + 1]
        ratio = diff_left / (diff_left - diff_right)
        return float(fpr[idx] + ratio * (fpr[idx + 1] - fpr[idx]))

    best_idx = int(np.argmin(np.abs(diff)))
    return float((fpr[best_idx] + fnr[best_idx]) / 2.0)


def run_ood_eval(args, score_name, score_fn, score_tag, model_class_offset=0, plot_score_fn=None, print_summary=True):
    set_seed(args.seed)
    device = torch.device(args.device)

    merged_x, merged_y, y_true_binary, num_known_classes, unknown_label = merge_test_sets(
        args.test_file,
        args.extra_test_file,
        args.feature,
        args.seq_len,
        args.num_tabs,
    )
    data_loader = data_processor.load_iter(
        merged_x,
        merged_y,
        batch_size=args.batch_size,
        is_train=False,
        num_workers=args.num_workers,
    )
    model = load_model(
        args.model,
        args.model_file,
        num_known_classes + model_class_offset,
        args.num_tabs,
        device,
    )
    scores, plot_scores, collected_binary = collect_scores(
        model,
        data_loader,
        device,
        score_fn,
        unknown_label,
        plot_score_fn=plot_score_fn,
    )

    if not np.array_equal(y_true_binary, collected_binary):
        print("Warning: detected label order mismatch, using labels collected from DataLoader.")
    y_true_binary = collected_binary

    auroc = roc_auc_score(y_true_binary, scores)
    fpr90 = compute_fpr90(y_true_binary, scores)
    eer = compute_eer(y_true_binary, scores)
    dataset_name = infer_dataset_name(args.test_file)
    known_test_file = resolve_path(args.test_file)
    unknown_test_file = resolve_path(args.extra_test_file)
    plot_path = plot_ood_histogram(plot_scores, y_true_binary, dataset_name, score_tag)

    result = {
        "auroc": float(auroc),
        "fpr90": float(fpr90),
        "eer": float(eer),
        "score_name": score_name,
        "dataset_name": dataset_name,
        "model": args.model,
        "model_file": args.model_file,
        "known_test_file": known_test_file,
        "unknown_test_file": unknown_test_file,
        "merged_shape": f"X={tuple(merged_x.shape)}, y={tuple(merged_y.shape)}",
        "known_classes": int(num_known_classes),
        "unknown_label": int(unknown_label),
        "plot_path": plot_path,
    }

    if print_summary:
        summary_rows = [
            ["Dataset", result["dataset_name"]],
            ["Model", result["model"]],
            ["Checkpoint", result["model_file"]],
            ["Known test", result["known_test_file"]],
            ["Unknown test", result["unknown_test_file"]],
            ["Merged shape", result["merged_shape"]],
            ["Known classes", result["known_classes"]],
            ["Unknown label", result["unknown_label"]],
            ["Score", result["score_name"]],
            ["AUROC", f"{result['auroc']:.6f}"],
            ["FPR@90", f"{result['fpr90']:.6f}"],
            ["EER", f"{result['eer']:.6f}"],
            ["Histogram", result["plot_path"] if result["plot_path"] is not None else "-"],
        ]
        print(format_table(["Field", "Value"], summary_rows))

    return result