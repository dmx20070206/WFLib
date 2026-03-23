# Generates the traffic aggregation matrix. More details can be found in the paper:
# Subverting Website Fingerprinting Defenses with Robust Traffic Representation.
# Security 2023.
import argparse
import os
import random
from multiprocessing import freeze_support

import numpy as np
import torch
from WFlib.tools import data_processor

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
    parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
    parser.add_argument("--in_file", type=str, default="train", help="input file")
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = os.path.join("./datasets", args.dataset)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"The dataset path does not exist: {in_path}")

    out_file = os.path.join(in_path, f"tam_{args.in_file}.npz")

    if not os.path.exists(out_file):
        # Load dataset and generate TAM features.
        data = np.load(os.path.join(in_path, f"{args.in_file}.npz"))
        X = data["X"]
        y = data["y"]
        X = data_processor.length_align(X, args.seq_len)
        X = data_processor.extract_TAM(X)
        print(f"{args.in_file} process done: X = {X.shape}, y = {y.shape}")
        np.savez(out_file, X=X, y=y)
    else:
        print(f"{out_file} has been generated.")


if __name__ == "__main__":
    freeze_support()
    main()
