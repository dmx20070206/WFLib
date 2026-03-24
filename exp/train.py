import os
import sys
import torch
import random
import argparse
import numpy as np
from multiprocessing import freeze_support
from tqdm import tqdm
from WFlib import models
from WFlib.tools import data_processor, model_utils
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def parse_args():
    # Argument parser for command-line options, arguments, and sub-commands
    parser = argparse.ArgumentParser(description="WFlib")
    parser.add_argument("--dataset", type=str, required=True, default="CW", help="Dataset name")
    parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")
    parser.add_argument("--num_tabs", type=int, default=1, help="Maximum number of tabs opened by users while browsing")
    parser.add_argument("--open_set", action="store_true", help="Enable Open-set training with unknown-label handling")
    parser.add_argument(
        "--unknown_label", type=int, default=102, help="Label id used for unknown class samples in Open-set"
    )

    # Input parameters
    parser.add_argument("--train_file", type=str, default="train", help="Train file")
    parser.add_argument("--valid_file", type=str, default="valid", help="Valid file")
    parser.add_argument("--use_extra_train_file", type=str, default=None)
    parser.add_argument("--use_extra_valid_file", type=str, default=None)
    parser.add_argument("--feature", type=str, default="DIR", help="Feature type, options=[DIR, DT, DT2, TAM, TAF]")
    parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")

    # Optimization parameters
    parser.add_argument("--num_workers", type=int, default=10, help="Data loader num workers")
    parser.add_argument("--train_epochs", type=int, default=30, help="Train epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Optimizer learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--loss", type=str, default="CrossEntropyLoss", help="Loss function")
    parser.add_argument("--lradj", type=str, default="None", help="adjust learning rate, option=[None, StepLR]")
    parser.add_argument("--use_energy_loss", action="store_true")
    parser.add_argument("--energy_weight", type=float, default=0.3, help="Weight of energy loss term")
    parser.add_argument("--energy_m_in", type=float, default=-18.0, help="Known-class energy margin m_in")
    parser.add_argument("--energy_m_out", type=float, default=-2.0, help="Unknown-class energy margin m_out")

    # Output parameters
    parser.add_argument("--eval_metrics", nargs="+", required=True, type=str)
    parser.add_argument("--save_metric", type=str, default="F1-score")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Location of model checkpoints")
    parser.add_argument("--load_file", type=str, default=None, help="The pre-trained model file")
    parser.add_argument("--save_name", type=str, default="base", help="Name used to save the model")
    return parser.parse_args()


def ensure_npz_suffix(path_str):
    return path_str if path_str.endswith(".npz") else f"{path_str}.npz"


def resolve_extra_train_path(raw_path, dataset_dir):
    normalized = ensure_npz_suffix(raw_path)
    candidates = [normalized]
    if not os.path.isabs(normalized):
        candidates.append(os.path.join(dataset_dir, normalized))
        candidates.append(os.path.join("./datasets", normalized))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Extra train file not found: {raw_path}. Tried: {candidates}")


def load_splits_with_progress(file_specs, feature, seq_len, num_tabs):
    loaded = {}
    for split_name, split_path in tqdm(file_specs, desc="Loading datasets", unit="file", leave=False, dynamic_ncols=True):
        loaded[split_name] = data_processor.load_data(split_path, feature, seq_len, num_tabs)
    return loaded


def main():
    args = parse_args()

    if args.open_set and args.num_tabs != 1:
        raise ValueError("Open-set mode currently supports only num_tabs=1.")

    # Ensure the specified device is available
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), f"The specified device {args.device} does not exist"
    device = torch.device(args.device)

    # Define paths for dataset and checkpoints
    in_path = os.path.join("./datasets", args.dataset)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"The dataset path does not exist: {in_path}")
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(ckp_path, exist_ok=True)

    out_file = os.path.join(ckp_path, f"{args.save_name}.pth")
    if os.path.exists(out_file):
        print(f"{out_file} has been generated.")
        sys.exit(1)

    # Build file list and load with progress bar
    train_path = os.path.join(in_path, ensure_npz_suffix(args.train_file))
    valid_path = os.path.join(in_path, ensure_npz_suffix(args.valid_file))

    file_specs = [("train", train_path), ("valid", valid_path)]
    extra_train_path = None
    extra_valid_path = None

    if args.use_extra_train_file:
        extra_train_path = resolve_extra_train_path(args.use_extra_train_file, in_path)
        file_specs.append(("extra_train", extra_train_path))
        print(f"[debug] extra train file: {extra_train_path}")

    if args.use_extra_valid_file:
        extra_valid_path = resolve_extra_train_path(args.use_extra_valid_file, in_path)
        file_specs.append(("extra_valid", extra_valid_path))
        print(f"[debug] extra valid file: {extra_valid_path}")

    loaded_data = load_splits_with_progress(file_specs, args.feature, args.seq_len, args.num_tabs)
    train_X, train_y = loaded_data["train"]
    valid_X, valid_y = loaded_data["valid"]

    if extra_train_path is not None:
        extra_X, extra_y = loaded_data["extra_train"]
        train_X = torch.cat([train_X, extra_X], dim=0)
        train_y = torch.cat([train_y, extra_y], dim=0)

    if extra_valid_path is not None:
        extra_X, extra_y = loaded_data["extra_valid"]
        valid_X = torch.cat([valid_X, extra_X], dim=0)
        valid_y = torch.cat([valid_y, extra_y], dim=0)

    if args.num_tabs == 1:
        if args.open_set:
            known_train_mask = train_y != args.unknown_label
            known_train_y = train_y[known_train_mask]
            if known_train_y.numel() == 0:
                raise ValueError("No known-class samples found in training set for Open-set mode.")
            num_classes = int(known_train_y.max().item()) + 1
            assert num_classes == len(torch.unique(known_train_y)), "Known labels are not continuous"
            print(
                f"Open-set enabled: known={known_train_mask.sum().item()}, unknown={(~known_train_mask).sum().item()}"
            )
        else:
            num_classes = len(np.unique(train_y))
            assert num_classes == train_y.max() + 1, "Labels are not continuous"
    else:
        num_classes = train_y.shape[1]

    # Print dataset information
    print(f"Train: X={train_X.shape}, y={train_y.shape}")
    print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
    print(f"num_classes: {num_classes}")

    # Load data into iterators
    train_iter = data_processor.load_iter(train_X, train_y, args.batch_size, True, args.num_workers)
    valid_iter = data_processor.load_iter(valid_X, valid_y, args.batch_size, False, args.num_workers)

    # Initialize model, optimizer, and loss function
    if args.model in ["BAPM", "TMWF"]:
        model = eval(f"models.{args.model}")(num_classes, args.num_tabs)
    else:
        model = eval(f"models.{args.model}")(num_classes)
    optimizer = eval(f"torch.optim.{args.optimizer}")(model.parameters(), lr=args.learning_rate)

    if args.load_file is None:
        print("No pre-trained model")
    else:
        print("Loading the pretrained model in ", args.load_file)
        checkpoint = torch.load(args.load_file)

        for k in list(checkpoint.keys()):
            if k.startswith("backbone."):
                if k.startswith("backbone") and not k.startswith("backbone.fc"):
                    checkpoint[k[len("backbone.") :]] = checkpoint[k]
            del checkpoint[k]

        log = model.load_state_dict(checkpoint, strict=False)
        assert log.missing_keys == ["fc.weight", "fc.bias"]

    model.to(device)

    # Train the model
    model_utils.model_train(
        model,
        optimizer,
        train_iter,
        valid_iter,
        args.loss,
        args.save_metric,
        args.eval_metrics,
        args.train_epochs,
        out_file,
        num_classes,
        args.num_tabs,
        device,
        args.lradj,
        args.open_set,
        args.unknown_label,
        args.use_energy_loss,
        args.energy_weight,
        args.energy_m_in,
        args.energy_m_out,
    )


if __name__ == "__main__":
    freeze_support()
    main()
