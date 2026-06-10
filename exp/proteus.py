import os
import json
import torch
import random
import argparse
import numpy as np
from WFlib import models
from sklearn.mixture import GaussianMixture
from WFlib.tools import data_processor, evaluator, model_utils
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")


def compute_gaussian_kernel(source, target):
    sample_count = int(source.size(0)) + int(target.size(0))
    combined = torch.cat([source, target], dim=0)
    l2_distance = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(2)
    bandwidth = torch.sum(l2_distance) / (sample_count**2 - sample_count)
    bandwidth = torch.clamp(bandwidth, min=1e-5)
    return torch.exp(-l2_distance / (bandwidth + 1e-5))


def calculate_mmd_loss(source_features, target_features):
    batch_size = min(source_features.size(0), target_features.size(0))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]
    kernels = compute_gaussian_kernel(source_features, target_features)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)


def compute_softmax_entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def evaluate_model(model, data_loader, metrics, device, open_set=False, num_tabs=1):
    with torch.no_grad():
        model.eval()
        predictions = []
        ground_truths = []
        for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs, _ = model(inputs)
            preds = torch.argsort(outputs, dim=1, descending=True)[:, 0]
            predictions.append(preds.cpu().numpy())
            ground_truths.append(labels.cpu().numpy())
        predictions = np.concatenate(predictions)
        ground_truths = np.concatenate(ground_truths)
    return model_utils.measure_open_set(ground_truths, predictions, metrics, num_tabs, open_set=open_set)


def compute_gmm_probabilities(model, data_loader, device):
    entropies = []
    predictions = []
    with torch.no_grad():
        model.eval()
        for batch in data_loader:
            inputs, _ = batch[0].to(device), batch[1].to(device)
            outputs, _ = model(inputs)
            preds = torch.argsort(outputs, dim=1, descending=True)[:, 0]
            entropies.append(compute_softmax_entropy(outputs).cpu().numpy())
            predictions.append(preds.cpu().numpy())
    entropies = np.concatenate(entropies).flatten()
    predictions = np.concatenate(predictions).flatten()
    entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min())
    entropies = entropies.reshape(-1, 1)
    predictions = torch.tensor(predictions, dtype=torch.int64)
    gmm = GaussianMixture(n_components=2, tol=1e-6)
    gmm.fit(entropies)
    probabilities = gmm.predict_proba(entropies)
    low_uncertainty_index = np.argmin(gmm.means_.flatten())
    return probabilities[:, low_uncertainty_index], predictions


def create_pseudo_labels(clean_probs, inputs, labels, threshold, batch_size, num_workers):
    clean_indices = clean_probs >= threshold
    pseudo_inputs = inputs[clean_indices]
    pseudo_labels = labels[clean_indices]
    return data_processor.load_iter(pseudo_inputs, pseudo_labels, batch_size, True, num_workers)


def adapt_model(
    model,
    test_data,
    adapt_data_loader,
    origin_data_loader,
    test_data_loader,
    metrics,
    device,
    threshold,
    open_set,
    num_tabs,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    origin_data_iter = iter(origin_data_loader)
    loss_function = torch.nn.CrossEntropyLoss()
    best_results = {}
    for epoch in range(100):
        if epoch % 5 == 0:
            clean_probs, pseudo_labels = compute_gmm_probabilities(model, test_data_loader, device)
            pseudo_loader = create_pseudo_labels(
                clean_probs, test_data, pseudo_labels, threshold, args.batch_size, args.num_workers
            )
            pseudo_data_iter = iter(pseudo_loader)
        model.train()
        origin_loss_sum, mmd_loss_sum, entropy_loss_sum, pseudo_loss_sum, total_samples = 0, 0, 0, 0, 0
        for adapt_batch in adapt_data_loader:
            try:
                origin_batch = next(origin_data_iter)
            except StopIteration:
                origin_data_iter = iter(origin_data_loader)
                origin_batch = next(origin_data_iter)
            try:
                pseudo_batch = next(pseudo_data_iter)
            except StopIteration:
                pseudo_data_iter = iter(pseudo_loader)
                pseudo_batch = next(pseudo_data_iter)
            adapt_inputs, adapt_labels = adapt_batch[0].to(device), adapt_batch[1].to(device)
            origin_inputs, origin_labels = origin_batch[0].to(device), origin_batch[1].to(device)
            pseudo_inputs, pseudo_labels = pseudo_batch[0].to(device), pseudo_batch[1].to(device)
            optimizer.zero_grad()
            origin_outputs, origin_features = model(origin_inputs)
            adapt_outputs, adapt_features = model(adapt_inputs)
            pseudo_outputs, pseudo_features = model(pseudo_inputs)
            softmax_out = F.softmax(adapt_outputs, dim=-1)
            mean_softmax = softmax_out.mean(dim=0)
            classification_loss = loss_function(origin_outputs, origin_labels)
            pseudo_loss = loss_function(pseudo_outputs, pseudo_labels)
            entropy_loss = compute_softmax_entropy(adapt_outputs).mean(0) + torch.sum(
                mean_softmax * torch.log(mean_softmax + 1e-5)
            )
            mmd_loss = calculate_mmd_loss(origin_features, adapt_features)
            total_loss = classification_loss + pseudo_loss + entropy_loss + mmd_loss
            total_loss.backward()
            optimizer.step()
            origin_loss_sum += classification_loss.data.cpu().numpy() * origin_outputs.shape[0]
            mmd_loss_sum += mmd_loss.data.cpu().numpy() * origin_outputs.shape[0]
            entropy_loss_sum += entropy_loss.data.cpu().numpy() * origin_outputs.shape[0]
            pseudo_loss_sum += pseudo_loss.data.cpu().numpy() * origin_outputs.shape[0]
            total_samples += adapt_outputs.shape[0]
        epoch_result = evaluate_model(model, test_data_loader, metrics, device, open_set, num_tabs)
        print(f"{epoch}:", epoch_result)
        for metric_name, metric_val in epoch_result.items():
            if metric_name not in best_results or metric_val > best_results[metric_name]:
                best_results[metric_name] = metric_val
    return best_results


def load_and_merge(file_list):
    all_data, all_labels = [], []
    for f in file_list:
        print(f"merge {f}")
        name = f if f.endswith(".npz") else f"{f}.npz"
        data, labels = data_processor.load_data(name, args.feature, args.seq_len, args.num_tabs)
        all_data.append(data)
        all_labels.append(labels)
    return torch.cat(all_data), torch.cat(all_labels)


def rebalance_tune_data(data, labels):
    unknown_label = int(labels.max().item())
    known_idx = torch.where(labels != unknown_label)[0]
    unknown_idx = torch.where(labels == unknown_label)[0]
    known_count = int(known_idx.numel())
    unknown_count = int(unknown_idx.numel())

    keep_known = min(known_count, int(unknown_count / args.tune_unknown_ratio))
    keep_unknown = min(unknown_count, int(round(keep_known * args.tune_unknown_ratio)))

    if keep_known == 0 or keep_unknown == 0:
        keep_known = min(known_count, 1)
        keep_unknown = min(unknown_count, max(1, int(round(keep_known * args.tune_unknown_ratio))))

    known_perm = torch.randperm(known_count)[:keep_known]
    unknown_perm = torch.randperm(unknown_count)[:keep_unknown]
    selected_idx = torch.cat([known_idx[known_perm], unknown_idx[unknown_perm]])
    selected_idx = selected_idx[torch.randperm(selected_idx.numel())]

    actual_ratio = keep_unknown / keep_known if keep_known > 0 else float("inf")
    print(
        f"Tune ratio: requested 1:{args.tune_unknown_ratio:g}, "
        f"using known={keep_known}, unknown={keep_unknown} (actual 1:{actual_ratio:.2f})"
    )
    return data[selected_idx], labels[selected_idx]


# Argument parsing and setup omitted for brevity
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Command-line arguments
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, required=True, default="CW", help="Dataset name")
parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")
parser.add_argument("--num_tabs", type=int, default=1, help="Maximum number of tabs opened by users while browsing")
parser.add_argument("--open_set", action="store_true", help="Enable Open-Set adaptation/evaluation (K-class + OOD)")

# Input parameters
parser.add_argument("--train_file", type=str, default="train", help="Train file")
parser.add_argument("--extra_train_file", type=str, default=None, help="Extra train file (relative to ./datasets/)")
parser.add_argument(
    "--tune_file",
    type=str,
    default=None,
    help="Tune file for adaptation (unlabeled target domain); defaults to test_file if not set",
)
parser.add_argument("--extra_tune_file", type=str, default=None, help="Extra tune file (relative to ./datasets/)")
parser.add_argument("--test_file", type=str, default="test", help="Test file")
parser.add_argument("--extra_test_file", type=str, default=None, help="Extra test file (relative to ./datasets/)")
parser.add_argument("--feature", type=str, default="DIR", help="Feature type, options=[DIR, DT, DT2, TAM, TAF]")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")

# Optimization parameters
parser.add_argument("--num_workers", type=int, default=10, help="Data loader num workers")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")

# Output parameters
parser.add_argument(
    "--eval_method", type=str, default="common", help="Method used in the evaluation, options=[common, kNN, holmes]"
)
parser.add_argument(
    "--eval_metrics",
    nargs="+",
    required=True,
    type=str,
    help="Evaluation metrics, options=[Accuracy, Precision, Recall, F1-score, P@min, r-Precision]",
)
parser.add_argument("--log_path", type=str, default="./logs/", help="Log path")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Location of model checkpoints")
parser.add_argument("--load_name", type=str, default="base", help="Name of the model file")
parser.add_argument("--result_file", type=str, default="result", help="File to save test results")
parser.add_argument("--gmm_threshold", type=float, default=0.6, help="GMM threshold")
parser.add_argument("--model_save_name", type=str, default="proteus", help="Name used to save the model")
parser.add_argument("--tune_unknown_ratio", type=float, default=2.0, help="Tune known:unknown target ratio 1:r")

# Parse arguments
args = parser.parse_args()

# Ensure the specified device is available
if args.device.startswith("cuda"):
    assert torch.cuda.is_available(), f"The specified device {args.device} does not exist"
device = torch.device(args.device)

# Define paths for dataset, logs, and checkpoints
dataset_path = os.path.join("./datasets", args.dataset)
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The dataset path does not exist: {dataset_path}")
log_path = os.path.join(args.log_path, args.dataset, args.model)
ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
os.makedirs(log_path, exist_ok=True)
output_file = os.path.join(log_path, f"{args.result_file}.json")

# Load training and validation data
train_files = [os.path.join(dataset_path, f"{args.train_file}.npz")] + (
    [os.path.join("./datasets", args.extra_train_file)] if args.extra_train_file else []
)
_tune_name = args.tune_file if args.tune_file else args.test_file
_extra_tune_name = args.extra_tune_file if args.extra_tune_file else args.extra_test_file
tune_files = [os.path.join(dataset_path, f"{_tune_name}.npz")] + (
    [os.path.join("./datasets", _extra_tune_name)] if _extra_tune_name else []
)
train_data, train_labels = load_and_merge(train_files)
tune_data, tune_labels = load_and_merge(tune_files)
if args.open_set and args.num_tabs == 1:
    tune_data, tune_labels = rebalance_tune_data(tune_data, tune_labels)

if args.test_file != _tune_name or args.extra_test_file != _extra_tune_name:
    print("Test arguments are kept for compatibility but ignored; using tune data for adaptation and evaluation.")

test_data, test_labels = tune_data, tune_labels
if args.num_tabs == 1:
    if args.open_set:
        unknown_label = int(tune_labels.max().item())
        known_tune_labels = tune_labels[tune_labels != unknown_label]
        if known_tune_labels.numel() == 0:
            raise ValueError("No known-class samples found in tune/test set for Open-Set mode.")
        num_classes = int(known_tune_labels.max().item()) + 1
    else:
        num_classes = len(np.unique(tune_labels))
        assert num_classes == tune_labels.max() + 1, "Labels are not continuous"
else:
    num_classes = tune_labels.shape[1]

# Print dataset information
print(f"Train data shape: X={train_data.shape}, y={train_labels.shape}")
print(f"Tune/Test data shape: X={tune_data.shape}, y={tune_labels.shape}")
print(f"Number of classes: {num_classes}")

# Load data into iterators
origin_data_loader = data_processor.load_iter(train_data, train_labels, args.batch_size, True, args.num_workers)
adapt_data_loader = data_processor.load_iter(
    tune_data, torch.zeros(len(tune_data), dtype=torch.int64), args.batch_size, True, args.num_workers
)
test_data_loader = data_processor.load_iter(test_data, test_labels, args.batch_size, False, args.num_workers)

# Initialize model, optimizer, and loss function
ckpt_state_dict = torch.load(os.path.join(ckp_path, f"{args.load_name}.pth"), map_location="cpu")

if args.model in ["BAPM", "TMWF"]:
    model = eval(f"models.{args.model}")(num_classes, args.num_tabs)
else:
    model = eval(f"models.{args.model}")(num_classes)

model.load_state_dict(ckpt_state_dict)
model.to(device)

# Evaluation before adaptation
initial_result = evaluate_model(model, test_data_loader, args.eval_metrics, device, args.open_set, args.num_tabs)
print("Initial evaluation result:")
print(initial_result)

# Model adaptation
best_results = adapt_model(
    model,
    test_data,
    adapt_data_loader,
    origin_data_loader,
    test_data_loader,
    args.eval_metrics,
    device,
    args.gmm_threshold,
    args.open_set,
    args.num_tabs,
)

# Evaluation after adaptation
final_result = evaluate_model(model, test_data_loader, args.eval_metrics, device, args.open_set, args.num_tabs)
for k, v in best_results.items():
    final_result[f"best_{k}"] = v
print("Evaluation after adaptation:")
print(final_result)

# Save model
model_save_path = os.path.join(ckp_path, f"{args.model_save_name}.pth")
torch.save(model.state_dict(), model_save_path)

# Save results to file
with open(output_file, "w") as result_file:
    json.dump(final_result, result_file, indent=4)
