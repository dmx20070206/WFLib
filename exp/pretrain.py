import sys
import os
import torch
import random
import argparse
import numpy as np
from WFlib import models
from WFlib.tools import data_processor, netclr_pretrain, netclr_augmentor, swallow_utils


fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="WFlib")
    parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
    parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")
    parser.add_argument("--train_file", type=str, default="train", help="Train file")
    parser.add_argument("--extra_train_file", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8, help="Data loader num workers")
    parser.add_argument("--train_epochs", type=int, default=100, help="Train epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Optimizer learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--log_path", type=str, default="./logs/", help="Log path")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Location of model checkpoints")
    parser.add_argument("--save_name", type=str, default="base", help="Name of the model file")
    return parser.parse_args()


def ensure_npz_suffix(path_str):
    return path_str if path_str.endswith(".npz") else f"{path_str}.npz"


def main():
    args = parse_args()
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), f"The specified device {args.device} does not exist"
    device = torch.device(args.device)

    in_path = os.path.join("./datasets", args.dataset)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"The dataset path does not exist: {in_path}")

    log_path = os.path.join(args.log_path, args.dataset, args.model)
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    out_file = os.path.join(ckp_path, f"{args.save_name}.pth")
    if os.path.exists(out_file):
        print(f"{out_file} has been generated.")
        sys.exit(1)

    if args.model == "Swallow":
        file_paths = [os.path.join(in_path, ensure_npz_suffix(args.train_file))]
        if args.extra_train_file:
            file_paths.append(os.path.join("./datasets", ensure_npz_suffix(args.extra_train_file)))
        swallow_utils.run_pretrain(
            file_paths=file_paths,
            out_file=out_file,
            device=device,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
        )
        return

    train_data = np.load(os.path.join(in_path, ensure_npz_suffix(args.train_file)))
    train_X = np.sign(train_data["X"]).astype(np.int64)
    train_X = data_processor.length_align(train_X, 5000)
    train_y = train_data["y"]
    num_classes = len(np.unique(train_y))
    assert num_classes == train_y.max() + 1, "Labels are not continuous"

    print(f"Train: X={train_X.shape}, y={train_y.shape}")
    print(f"num_classes: {num_classes}")

    outgoing_burst_sizes = []
    random_X = train_X[np.random.choice(range(len(train_X)), size=1000, replace=False)]
    for x in random_X:
        bursts = netclr_augmentor.find_bursts(x)
        outgoing_burst_sizes += [burst[2] for burst in bursts if burst[2] > 0]

    if not outgoing_burst_sizes:
        raise ValueError("No outgoing bursts found in the sampled training traces.")

    max_outgoing_burst_size = int(max(outgoing_burst_sizes))
    count, bins = np.histogram(outgoing_burst_sizes, bins=max(1, max_outgoing_burst_size - 1))
    pdf = count / np.sum(count)
    outgoing_burst_size_cdf = np.zeros_like(bins)
    outgoing_burst_size_cdf[1:] = np.cumsum(pdf)

    augmentor = netclr_augmentor.Augmentor(
        max_outgoing_burst_size,
        outgoing_burst_sizes,
        outgoing_burst_size_cdf,
    )
    train_dataset = netclr_pretrain.PreTrainData(train_X, train_y, augmentor, 2)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    df = eval(f"models.{args.model}")(512)
    model = models.DFsimCLR(df, out_dim=128).to(device)
    optimizer = eval(f"torch.optim.{args.optimizer}")(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    netclr = netclr_pretrain.NetCLR(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        fp16_precision=True,
        temperature=0.5,
        n_views=2,
        num_epoches=args.train_epochs,
        batch_size=args.batch_size,
        out_file=out_file,
        device=device,
    )
    netclr.train(train_loader)


if __name__ == "__main__":
    main()
