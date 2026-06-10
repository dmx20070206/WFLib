import os
import sys
import torch
import random
import argparse
import numpy as np
import warnings
from multiprocessing import freeze_support
from tqdm import tqdm
from WFlib import models
from WFlib.tools import data_processor, model_utils

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="WFlib")
    parser.add_argument("--dataset", type=str, required=True, default="CW", help="Dataset name")
    parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")
    parser.add_argument("--num_tabs", type=int, default=1, help="Maximum number of tabs opened by users while browsing")

    # Open-set evaluation parameters
    parser.add_argument("--open_set", action="store_true", help="Enable Open-set evaluation (K-class + OOD)")
    parser.add_argument("--unknown_ratio", type=float, default=2.0, help="Test known:unknown target ratio 1:r")

    # Input parameters
    parser.add_argument("--valid_file", type=str, default="valid", help="Valid file")
    parser.add_argument("--extra_valid_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default="test", help="Test file")
    parser.add_argument("--extra_test_file", type=str, default=None)
    parser.add_argument("--feature", type=str, default="DIR", help="Feature type, options=[DIR, DT, DT2, TAM, TAF, CIF]")
    parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")

    # Optimization parameters
    parser.add_argument("--num_workers", type=int, default=10, help="Data loader num workers")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")

    # Output parameters
    parser.add_argument("--eval_method", type=str, default="common", help="options=[common, kNN, holmes]")
    parser.add_argument("--eval_metrics", nargs="+", required=True, type=str, help="Evaluation metrics")
    parser.add_argument("--log_path", type=str, default="./logs/", help="Log path")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Location of model checkpoints")
    parser.add_argument("--load_name", type=str, default="base", help="Name of the model file")
    parser.add_argument("--result_file", type=str, default="result", help="File to save test results")
    return parser.parse_args()


def ensure_npz_suffix(path_str):
    return path_str if path_str.endswith(".npz") else f"{path_str}.npz"


def load_splits_with_progress(file_specs, feature, seq_len, num_tabs):
    loaded = {}
    for split_name, split_path in tqdm(
        file_specs, desc="Loading datasets", unit="file", leave=False, dynamic_ncols=True
    ):
        loaded[split_name] = data_processor.load_data(split_path, feature, seq_len, num_tabs)
    return loaded


def rebalance_open_set_test_data(data, labels, unknown_ratio):
    unknown_label = int(labels.max().item())

    known_idx = torch.where(labels != unknown_label)[0]
    unknown_idx = torch.where(labels == unknown_label)[0]
    known_count = int(known_idx.numel())
    unknown_count = int(unknown_idx.numel())

    keep_known = min(known_count, int(unknown_count / unknown_ratio))
    keep_unknown = min(unknown_count, int(round(keep_known * unknown_ratio)))

    known_perm = torch.randperm(known_count)[:keep_known]
    unknown_perm = torch.randperm(unknown_count)[:keep_unknown]
    selected_idx = torch.cat([known_idx[known_perm], unknown_idx[unknown_perm]])
    selected_idx = selected_idx[torch.randperm(selected_idx.numel())]

    print(f"Test ratio: requested 1:{unknown_ratio:g}, " f"using known={keep_known}, unknown={keep_unknown}")
    return data[selected_idx], labels[selected_idx]


def main():
    args = parse_args()

    # Ensure the specified device is available
    device = torch.device(args.device)

    # Define paths for dataset, logs, and checkpoints
    in_path = os.path.join("./datasets", args.dataset)
    log_path = os.path.join(args.log_path, args.dataset, args.model)
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(log_path, exist_ok=True)
    out_file = os.path.join(log_path, f"{args.result_file}.json")

    valid_path = os.path.join(in_path, ensure_npz_suffix(args.valid_file))
    test_path = os.path.join(in_path, ensure_npz_suffix(args.test_file))
    file_specs = [("valid", valid_path), ("test", test_path)]

    extra_valid_path = None
    if args.extra_valid_file:
        extra_valid_path = os.path.join("./datasets", ensure_npz_suffix(args.extra_valid_file))
        file_specs.append(("extra_valid", extra_valid_path))

    extra_test_path = None
    if args.extra_test_file:
        extra_test_path = os.path.join("./datasets", ensure_npz_suffix(args.extra_test_file))
        file_specs.append(("extra_test", extra_test_path))

    loaded = load_splits_with_progress(file_specs, args.feature, args.seq_len, args.num_tabs)
    valid_X, valid_y = loaded["valid"]
    test_X, test_y = loaded["test"]

    if extra_valid_path is not None:
        extra_valid_X, extra_valid_y = loaded["extra_valid"]
        valid_X = torch.cat([valid_X, extra_valid_X], dim=0)
        valid_y = torch.cat([valid_y, extra_valid_y], dim=0)

    if extra_test_path is not None:
        extra_test_X, extra_test_y = loaded["extra_test"]
        test_X = torch.cat([test_X, extra_test_X], dim=0)
        test_y = torch.cat([test_y, extra_test_y], dim=0)

    # For open-set evaluation, rebalance the test set to achieve the desired known:unknown ratio
    if args.open_set and args.num_tabs == 1:
        test_X, test_y = rebalance_open_set_test_data(
            test_X,
            test_y,
            args.unknown_ratio,
        )

    checkpoint_file = os.path.join(ckp_path, f"{args.load_name}.pth")
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if args.num_tabs == 1:
        if args.open_set:
            unknown_label = int(valid_y.max().item())
            known_valid_y = valid_y[valid_y != unknown_label]
            if known_valid_y.numel() == 0:
                raise ValueError("No known-class samples found in validation set for Open-set mode.")
            num_classes = int(known_valid_y.max().item()) + 1
        else:
            num_classes = int(valid_y.max().item()) + 1
    else:
        num_classes = test_y.shape[1]

    print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
    print(f"Test: X={test_X.shape}, y={test_y.shape}")
    print(f"num_classes: {num_classes}")

    valid_iter = data_processor.load_iter(valid_X, valid_y, args.batch_size, False, args.num_workers)
    test_iter = data_processor.load_iter(test_X, test_y, args.batch_size, False, args.num_workers)

    if args.model in ["BAPM", "TMWF"]:
        model = eval(f"models.{args.model}")(num_classes, args.num_tabs)
    else:
        model = eval(f"models.{args.model}")(num_classes)

    model.load_state_dict(state_dict)
    model.to(device)

    model_utils.model_eval(
        model,
        test_iter,
        valid_iter,
        args.eval_method,
        args.eval_metrics,
        out_file,
        num_classes,
        ckp_path,
        args.num_tabs,
        device,
        args.open_set,
    )


if __name__ == "__main__":
    freeze_support()
    main()
