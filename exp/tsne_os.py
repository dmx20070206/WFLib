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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
parser.add_argument("--split_refresh", type=int, default=2, help="Re-split target every N epochs")
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

parser.add_argument("--energy_margin_in", type=float, default=-12.0)
parser.add_argument("--energy_margin_out", type=float, default=-2.0)

parser.add_argument("--fix_seed", type=int, default=20070206)
parser.add_argument("--unknown_label", type=int, default=102)

args = parser.parse_args()

random.seed(args.fix_seed)
torch.manual_seed(args.fix_seed)
np.random.seed(args.fix_seed)


# --- Feature Extraction -------------------------------------------------------


def extract_features(model, data, device):
    """Extract feature vectors from the backbone for all samples.

    Runs on CPU to avoid GPU OOM when the test set is large.
    The model is temporarily moved to CPU and restored to device afterwards.
    """
    model.eval()
    model.cpu()
    features = []
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            batch = torch.as_tensor(data[i : i + args.batch_size])
            _, feat = model(batch)
            features.append(feat.numpy())
    model.to(device)
    return np.concatenate(features, axis=0)


# --- t-SNE Plot ---------------------------------------------------------------


def plot_tsne(features, labels, unknown_label, save_path):
    """Run t-SNE on features and plot known vs unknown classes."""
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=args.fix_seed, init="pca", learning_rate="auto")
    embedding = tsne.fit_transform(features)

    known_mask = labels != unknown_label
    unknown_mask = ~known_mask

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        embedding[known_mask, 0],
        embedding[known_mask, 1],
        c="#4C72B0",
        s=6,
        alpha=0.5,
        linewidths=0,
        label=f"Known ({known_mask.sum()})",
    )
    ax.scatter(
        embedding[unknown_mask, 0],
        embedding[unknown_mask, 1],
        c="#DD8452",
        s=6,
        alpha=0.5,
        linewidths=0,
        label=f"Unknown ({unknown_mask.sum()})",
    )
    ax.legend(markerscale=2, fontsize=11)
    ax.set_title(
        f"t-SNE of Target Domain Features\n{args.dataset} / {args.model} / pretrained: {args.load_name}", fontsize=12
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved -> {save_path}")


# --- Entry Point --------------------------------------------------------------


def main():
    device = torch.device(args.device)

    dataset_path = os.path.join("./datasets", args.dataset)
    log_path = os.path.join(args.log_path, args.dataset, args.model)
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(log_path, exist_ok=True)

    test_data, test_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.test_file}.npz"),
        args.feature,
        args.seq_len,
        args.num_tabs,
    )
    test_labels_np = np.asarray(test_labels)

    known_src_mask = test_labels_np != args.unknown_label
    num_classes = int(test_labels_np[known_src_mask].max()) + 1

    print(f"\n{'='*20} Configuration {'='*20}")
    print(f"Dataset: {args.dataset},  Model: {args.model},  Device: {device}")
    print(f"Test: {test_data.shape}")
    print(f"  Known samples  : {known_src_mask.sum()}")
    print(f"  Unknown samples: {(~known_src_mask).sum()}")
    print(f"{'='*55}\n")

    # Build and load pretrained backbone.
    backbone = models.DF(num_classes) if args.model == "DF" else eval(f"models.{args.model}")(num_classes)
    ckp_file = os.path.join(ckp_path, f"{args.load_name}.pth")
    if os.path.exists(ckp_file):
        backbone.load_state_dict(torch.load(ckp_file, map_location="cpu"))
        print(f"Loaded pretrained backbone from {ckp_file}")
    else:
        print(f"[WARNING] Checkpoint not found: {ckp_file}. Using random weights.")
    backbone.to(device)

    features = extract_features(backbone, test_data, device)
    print(f"Extracted features: {features.shape}")

    save_path = os.path.join(log_path, f"tsne_{args.load_name}.png")
    plot_tsne(features, test_labels_np, args.unknown_label, save_path)


if __name__ == "__main__":
    main()
