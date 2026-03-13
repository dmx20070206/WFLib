import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from WFlib.tools import data_processor
from lmmd import compute_lmmd
from academic_style import apply_style, COLORS, MARKERS

TEST_FILES = ["day14", "day30", "day90", "day150", "day270"]


def _as_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "detach") and hasattr(arr, "cpu") and hasattr(arr, "numpy"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def load_lmmd(dataset_path, train_file, test_file, feature, seq_len, num_tabs):
    train_data, train_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{train_file}.npz"), feature, seq_len, num_tabs
    )
    test_data, test_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{test_file}.npz"), feature, seq_len, num_tabs
    )
    train_x = _as_numpy(train_data)
    test_x = _as_numpy(test_data)
    train_y = _as_numpy(train_labels)
    test_y = _as_numpy(test_labels)
    if train_y.ndim > 1:
        train_y = np.argmax(train_y, axis=1)
    if test_y.ndim > 1:
        test_y = np.argmax(test_y, axis=1)
    data = np.concatenate([train_x, test_x], axis=0)
    cls = np.concatenate([train_y.reshape(-1), test_y.reshape(-1)], axis=0)
    dom = np.concatenate([np.zeros(train_x.shape[0], dtype=np.int64), np.ones(test_x.shape[0], dtype=np.int64)], axis=0)
    return compute_lmmd(data, (cls, dom), source_domain=0, target_domain=1)


def main():
    parser = argparse.ArgumentParser(description="Compute LMMD curves for multiple test sets")
    parser.add_argument("--dataset", type=str, default="TemporalDrift", help="Dataset name")
    parser.add_argument("--train_file", type=str, default="train", help="Train file name without .npz")
    parser.add_argument("--feature", type=str, default="DIR", help="Feature type")
    parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
    parser.add_argument("--num_tabs", type=int, default=1, help="Maximum number of tabs")
    args = parser.parse_args()

    apply_style()

    dataset_path = os.path.join("./datasets", args.dataset)
    output_dir = "results/key_observation"
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, test_file in enumerate(TEST_FILES):
        test_path = os.path.join(dataset_path, f"{test_file}.npz")
        if not os.path.exists(test_path):
            print(f"Skipping {test_file}: file not found")
            continue
        lmmd_result = load_lmmd(dataset_path, args.train_file, test_file, args.feature, args.seq_len, args.num_tabs)
        x = np.arange(lmmd_result.shape[0])
        ax.plot(
            x,
            lmmd_result,
            label=test_file,
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            markevery=max(1, len(x) // 20),
        )
        print(f"{test_file}: {lmmd_result}")

    ax.set_xlabel("Class Index")
    ax.set_ylabel("LMMD")
    ax.set_title(f"LMMD per Class — {args.dataset}")
    ax.legend(title="Test set", frameon=True)

    save_path = os.path.join(output_dir, "lmmd_curves.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    main()
