import argparse
import copy
import os
import random
import warnings

import numpy as np
import torch

from WFlib import models
from WFlib.tools import data_processor, model_utils

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def parse_args():
    parser = argparse.ArgumentParser(description="K-shot supervised fine-tuning for WFlib models")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_tabs", type=int, default=1)
    parser.add_argument("--open_set", action="store_true")

    parser.add_argument("--tune_file", type=str, required=True)
    parser.add_argument("--extra_tune_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--extra_test_file", type=str, default=None)
    parser.add_argument("--feature", type=str, default="DIR")
    parser.add_argument("--seq_len", type=int, default=5000)

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k_shot", type=int, required=True)
    parser.add_argument("--sft_epochs", type=int, default=30)
    parser.add_argument("--sft_lr", type=float, default=1e-4)
    parser.add_argument("--sft_weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="Adam")

    parser.add_argument("--eval_metrics", nargs="+", required=True, type=str)
    parser.add_argument("--eval_method", type=str, default="common")
    parser.add_argument("--save_metric", type=str, default="F1-score")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--load_name", type=str, default="base")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--fix_seed", type=int, default=20070206)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_npz_suffix(path_str):
    return path_str if path_str.endswith(".npz") else f"{path_str}.npz"


def load_and_merge(file_list, feature, seq_len, num_tabs):
    all_data = []
    all_labels = []
    for file_path in file_list:
        npz_path = ensure_npz_suffix(file_path)
        data, labels = data_processor.load_data(npz_path, feature, seq_len, num_tabs)
        all_data.append(data)
        all_labels.append(labels)
    return torch.cat(all_data), torch.cat(all_labels)


def get_logits(raw_outputs):
    if isinstance(raw_outputs, (tuple, list)):
        return raw_outputs[0]
    return raw_outputs


def build_model(model_name, num_classes, num_tabs):
    if model_name in ["BAPM", "TMWF"]:
        return getattr(models, model_name)(num_classes, num_tabs)
    return getattr(models, model_name)(num_classes)


def make_loader(data, labels, batch_size, shuffle, num_workers):
    dataset = torch.utils.data.TensorDataset(data, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
    )


def evaluate_with_loader(
    model,
    data_loader,
    metrics,
    device,
    num_tabs,
    open_set=False,
    eval_method="common",
    memory_loader=None,
    num_classes=None,
):
    if eval_method == "kNN":
        ground_truths, predictions = model_utils.knn_monitor(model, device, memory_loader, data_loader, num_classes, 10)
        return model_utils.measure_open_set(ground_truths, predictions, metrics, num_tabs, open_set=open_set)

    predictions = []
    ground_truths = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            labels = batch[1]
            logits = get_logits(model(inputs))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            ground_truths.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions)
    ground_truths = np.concatenate(ground_truths)
    return model_utils.measure_open_set(ground_truths, predictions, metrics, num_tabs, open_set=open_set)


def sample_k_shot_indices(labels, k_shot):
    selected_parts = []
    remaining_parts = []

    for class_id in sorted(torch.unique(labels).tolist()):
        class_indices = torch.where(labels == class_id)[0]
        shuffled = class_indices[torch.randperm(class_indices.numel())]
        selected_parts.append(shuffled[:k_shot])
        remaining_parts.append(shuffled[k_shot:])

    selected_idx = torch.cat(selected_parts)
    selected_idx = selected_idx[torch.randperm(selected_idx.numel())]
    remaining_idx = torch.cat(remaining_parts)
    remaining_idx = remaining_idx[torch.randperm(remaining_idx.numel())]
    return selected_idx, remaining_idx


def supervised_finetune(model, train_loader, valid_loader, args, device, num_classes):
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), lr=args.sft_lr, weight_decay=args.sft_weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_metric = float("-inf")
    best_state = None
    best_valid_metrics = {}

    for epoch in range(args.sft_epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            optimizer.zero_grad()
            logits = get_logits(model(inputs))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            sample_count += batch_size

        train_loss = loss_sum / sample_count
        valid_metrics = evaluate_with_loader(
            model,
            valid_loader,
            args.eval_metrics,
            device,
            args.num_tabs,
            open_set=False,
            eval_method=args.eval_method,
            memory_loader=train_loader,
            num_classes=num_classes,
        )
        metric_value = float(valid_metrics[args.save_metric])
        metric_str = ", ".join(f"{key}: {value:.4f}" for key, value in valid_metrics.items())
        print(f"SFT Epoch {epoch + 1:03d}/{args.sft_epochs} | Loss: {train_loss:.4f} | {metric_str}")

        if metric_value > best_metric:
            best_metric = metric_value
            best_valid_metrics = dict(valid_metrics)
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_valid_metrics


def main():
    args = parse_args()
    set_seed(args.fix_seed)
    device = torch.device(args.device)

    dataset_path = os.path.join("./datasets", args.dataset)
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(ckp_path, exist_ok=True)

    tune_files = [os.path.join(dataset_path, ensure_npz_suffix(args.tune_file))]
    if args.extra_tune_file:
        tune_files.append(os.path.join("./datasets", ensure_npz_suffix(args.extra_tune_file)))

    eval_name = args.test_file if args.test_file else args.tune_file
    extra_eval_name = args.extra_test_file if args.extra_test_file else args.extra_tune_file
    eval_files = [os.path.join(dataset_path, ensure_npz_suffix(eval_name))]
    if extra_eval_name:
        eval_files.append(os.path.join("./datasets", ensure_npz_suffix(extra_eval_name)))

    tune_data, tune_labels = load_and_merge(tune_files, args.feature, args.seq_len, args.num_tabs)
    eval_data, eval_labels = load_and_merge(eval_files, args.feature, args.seq_len, args.num_tabs)

    if args.open_set:
        # Use the largest tuning label as the unknown class.
        tune_unknown_label = int(tune_labels.max().item())
        known_mask = tune_labels != tune_unknown_label
        labeled_data = tune_data[known_mask]
        labeled_labels = tune_labels[known_mask]
    else:
        labeled_data = tune_data
        labeled_labels = tune_labels

    num_classes = int(labeled_labels.max().item()) + 1
    selected_idx, remaining_idx = sample_k_shot_indices(labeled_labels, args.k_shot)
    sft_train_data = labeled_data[selected_idx]
    sft_train_labels = labeled_labels[selected_idx]
    if remaining_idx.numel() > 0:
        valid_data = labeled_data[remaining_idx]
        valid_labels = labeled_labels[remaining_idx]
    else:
        valid_data = sft_train_data
        valid_labels = sft_train_labels

    train_loader = make_loader(sft_train_data, sft_train_labels, args.batch_size, True, args.num_workers)
    valid_loader = make_loader(valid_data, valid_labels, args.batch_size, False, args.num_workers)
    eval_loader = make_loader(eval_data, eval_labels, args.batch_size, False, args.num_workers)

    checkpoint_in = os.path.join(ckp_path, f"{args.load_name}.pth")
    model = build_model(args.model, num_classes, args.num_tabs)
    model.load_state_dict(torch.load(checkpoint_in, map_location="cpu"))
    model.to(device)

    pre_metrics = evaluate_with_loader(
        model,
        eval_loader,
        args.eval_metrics,
        device,
        args.num_tabs,
        open_set=args.open_set,
        eval_method=args.eval_method,
        memory_loader=train_loader,
        num_classes=num_classes,
    )
    print("SFT pre-evaluation:")
    print(pre_metrics)

    best_valid_metrics = supervised_finetune(model, train_loader, valid_loader, args, device, num_classes)

    post_metrics = evaluate_with_loader(
        model,
        eval_loader,
        args.eval_metrics,
        device,
        args.num_tabs,
        open_set=args.open_set,
        eval_method=args.eval_method,
        memory_loader=train_loader,
        num_classes=num_classes,
    )
    print("SFT post-evaluation:")
    print(post_metrics)

    save_name = args.save_name if args.save_name else args.load_name
    checkpoint_out = os.path.join(ckp_path, f"{save_name}.pth")
    torch.save(model.state_dict(), checkpoint_out)

    print(f"Saved SFT checkpoint to {checkpoint_out}")


if __name__ == "__main__":
    main()
