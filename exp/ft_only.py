"""Few-shot fine-tuning baseline (FT-only)."""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from WFlib import models
from WFlib.tools import data_processor, evaluator


def unpack_output(output):
    if isinstance(output, (tuple, list)):
        return output[0], output[1] if len(output) > 1 else output[0]
    return output, output


def evaluate_model(model, data_loader, metrics, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for inputs, batch_labels in data_loader:
            logits, _ = unpack_output(model(inputs.to(device)))
            predictions.append(logits.argmax(1).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    if not predictions:
        return {metric: float("nan") for metric in metrics}
    return evaluator.measurement(
        np.concatenate(labels), np.concatenate(predictions), metrics
    )


def select_support_indices(labels, shot, seed=2024):
    if shot < 1:
        raise ValueError("shot must be at least 1")
    labels_np = labels.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    support = []
    for label in np.unique(labels_np):
        candidates = np.flatnonzero(labels_np == label)
        rng.shuffle(candidates)
        support.extend(candidates[:shot].tolist())
    support = np.asarray(sorted(support), dtype=np.int64)
    all_indices = np.arange(labels_np.shape[0])
    mask = np.ones(labels_np.shape[0], dtype=bool)
    mask[support] = False
    return support, all_indices[mask]


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(model_name, num_classes, num_tabs):
    return (
        getattr(models, model_name)(num_classes, num_tabs)
        if model_name in ("BAPM", "TMWF")
        else getattr(models, model_name)(num_classes)
    )


def make_parser():
    parser = argparse.ArgumentParser(description="FT-only baseline")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_tabs", type=int, default=1)
    parser.add_argument("--test_file", default="test")
    parser.add_argument("--feature", default="DIR")
    parser.add_argument("--seq_len", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_metrics", nargs="+", default=["Accuracy"])
    parser.add_argument("--log_path", default="./logs/")
    parser.add_argument("--checkpoints", default="./checkpoints/")
    parser.add_argument("--load_name", default="base")
    parser.add_argument("--result_file", default="result")
    parser.add_argument("--model_save_name", default="ft_only")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--support_seed", type=int, default=20070206)
    parser.add_argument("--ft_epochs", type=int, default=50)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = make_parser().parse_args(argv)
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    device = torch.device(
        "cpu"
        if args.device.startswith("cuda") and not torch.cuda.is_available()
        else args.device
    )
    dataset_path = os.path.join("./datasets", args.dataset)
    log_path = os.path.join(args.log_path, args.dataset, args.model)
    checkpoint_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Use test_file as the target domain (same split logic as proteus)
    target_data, target_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.test_file}.npz"),
        args.feature,
        args.seq_len,
        args.num_tabs,
    )
    if args.num_tabs != 1:
        raise ValueError("FT-only requires integer single-class labels (--num_tabs 1)")
    num_classes = int(torch.unique(target_labels).numel())

    # Split target into support and eval
    support_indices, eval_indices = select_support_indices(
        target_labels, args.shot, args.support_seed
    )
    support_data, support_labels = (
        target_data[support_indices],
        target_labels[support_indices],
    )
    eval_data, eval_labels = target_data[eval_indices], target_labels[eval_indices]

    support_loader = data_processor.load_iter(
        support_data, support_labels, args.batch_size, True, args.num_workers
    )
    eval_loader = data_processor.load_iter(
        eval_data, eval_labels, args.batch_size, False, args.num_workers
    )

    # Load base model
    model = build_model(args.model, num_classes, args.num_tabs)
    model.load_state_dict(
        load_checkpoint(os.path.join(checkpoint_path, f"{args.load_name}.pth"))
    )
    model.to(device)

    # Fine-tune on support set only (CE loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.ft_epochs):
        model.train()
        total_loss, count = 0.0, 0
        for inputs, labels in support_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            logits, _ = unpack_output(model(inputs))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            n = inputs.shape[0]
            total_loss += float(loss.detach()) * n
            count += n
        avg_loss = total_loss / max(count, 1)
        acc = evaluate_model(model, eval_loader, ["Accuracy"], device)
        print(
            f"[FT-only][Epoch {epoch + 1}/{args.ft_epochs}] "
            f"loss={avg_loss:.6f} acc={acc['Accuracy']:.6f}"
        )

    # Final evaluation
    result = evaluate_model(model, eval_loader, ["Accuracy"], device)
    print(f"[FT-only] Final accuracy: {result['Accuracy']:.6f}")
    torch.save(
        model.state_dict(),
        os.path.join(checkpoint_path, f"{args.model_save_name}.pth"),
    )
    with open(
        os.path.join(log_path, f"{args.result_file}.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    return result


if __name__ == "__main__":
    main()
