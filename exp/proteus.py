import os
import json
import torch
import random
import argparse
import numpy as np
from WFlib import models
from sklearn.mixture import GaussianMixture
from WFlib.tools import data_processor, evaluator
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

def compute_gaussian_kernel(source, target):
    sample_count = int(source.size(0)) + int(target.size(0))
    combined = torch.cat([source, target], dim=0)
    l2_distance = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(2)
    bandwidth = torch.sum(l2_distance) / (sample_count ** 2 - sample_count)
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

def evaluate_model(model, data_loader, metrics, device, desc="Evaluating"):
    with torch.no_grad():
        model.eval()
        predictions = []
        ground_truths = []
        # Use tqdm for progress bar
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            raw_output = model(inputs)
            if isinstance(raw_output, (tuple, list)):
                outputs = raw_output[0]
            else:
                outputs = raw_output
            preds = torch.argsort(outputs, dim=1, descending=True)[:, 0]
            predictions.append(preds.cpu().numpy())
            ground_truths.append(labels.cpu().numpy())
        predictions = np.concatenate(predictions)
        ground_truths = np.concatenate(ground_truths)
    return evaluator.measurement(ground_truths, predictions, metrics)

def compute_gmm_probabilities(model, data_loader, device):
    entropies = []
    predictions = []
    with torch.no_grad():
        model.eval()
        # Use tqdm for progress bar
        for batch in tqdm(data_loader, desc="Computing GMM Probs", leave=False):
            inputs, _ = batch[0].to(device), batch[1].to(device)
            raw_output = model(inputs)
            if isinstance(raw_output, (tuple, list)):
                outputs = raw_output[0]
            else:
                outputs = raw_output
            preds = torch.argsort(outputs, dim=1, descending=True)[:, 0]
            entropies.append(compute_softmax_entropy(outputs).cpu().numpy())
            predictions.append(preds.cpu().numpy())
    entropies = np.concatenate(entropies).flatten()
    predictions = np.concatenate(predictions).flatten()
    
    # Handle NaN or Inf values in entropies
    if np.isnan(entropies).any() or np.isinf(entropies).any():
        print("Warning: NaN or Inf detected in raw entropies. Replacing with 0.")
        entropies = np.nan_to_num(entropies, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize entropies to [0, 1]
    denom = entropies.max() - entropies.min()
    if denom > 1e-8:
        entropies = (entropies - entropies.min()) / denom
    else:
        entropies = np.zeros_like(entropies)
    entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min())
    # Final check for NaN values
    if np.isnan(entropies).any():
        entropies = np.nan_to_num(entropies, nan=0.0)
    
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

def adapt_model(model, test_data, adapt_data_loader, origin_data_loader, test_data_loader, metrics, device, threshold):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    origin_data_iter = iter(origin_data_loader)
    loss_function = torch.nn.CrossEntropyLoss()
    best_f1_score, best_f1_score_epoch, best_accuracy, best_accuracy_epoch = 0,0,0,0
    pseudo_loader = None 
    for epoch in range(100):
        if epoch % 5 == 0:
            # Recompute pseudo-labels every 5 epochs
            clean_probs, pseudo_labels = compute_gmm_probabilities(model, test_data_loader, device)
            # Create new pseudo-label data loader
            pseudo_loader = create_pseudo_labels(clean_probs, test_data, pseudo_labels, threshold, args.batch_size, args.num_workers)
            pseudo_data_iter = iter(pseudo_loader)
        model.train()
        # origin_loss: classification loss on original labeled data
        # mmd_loss: MMD loss between origin and adapt features
        # entropy_loss: entropy loss on adapt data
        # pseudo_loss: classification loss on pseudo-labeled data
        origin_loss_sum, mmd_loss_sum, entropy_loss_sum, pseudo_loss_sum, total_samples = 0, 0, 0, 0, 0
        
        # Use tqdm for training loop
        pbar = tqdm(adapt_data_loader, desc=f"Epoch {epoch+1}/100", leave=False)
        
        for adapt_batch in pbar:
            try:
                origin_batch = next(origin_data_iter)
            except StopIteration:
                origin_data_iter = iter(origin_data_loader)
                origin_batch = next(origin_data_iter)
            pseudo_batch = None
            if pseudo_loader is not None and len(pseudo_loader) > 0:
                try:
                    pseudo_batch = next(pseudo_data_iter)
                except StopIteration:
                    pseudo_data_iter = iter(pseudo_loader)
                    pseudo_batch = next(pseudo_data_iter)
                    
            
            # X,  y (labeled)
            origin_inputs, origin_labels = origin_batch[0].to(device), origin_batch[1].to(device)
            # X', _ (no labels)
            adapt_inputs, adapt_labels = adapt_batch[0].to(device), adapt_batch[1].to(device)
            # X', y'(pseudo)
            pseudo_inputs, pseudo_labels = (pseudo_batch[0].to(device), pseudo_batch[1].to(device)) if pseudo_batch is not None else (None, None)
                
            optimizer.zero_grad()
            # Forward pass
            # X -> outputs, features
            unpack = lambda x: x if isinstance(x, (tuple, list)) else (x, x)
            origin_outputs, origin_features = unpack(model(origin_inputs))
            adapt_outputs, adapt_features = unpack(model(adapt_inputs))
            pseudo_outputs, pseudo_features = unpack(model(pseudo_inputs)) if pseudo_inputs is not None else (None, None)
            
            softmax_out = F.softmax(adapt_outputs, dim=-1)
            mean_softmax = softmax_out.mean(dim=0)
            
            # origin_loss
            classification_loss = loss_function(origin_outputs, origin_labels)
            # pseudo_loss
            pseudo_loss = loss_function(pseudo_outputs, pseudo_labels) if pseudo_outputs is not None else torch.tensor(0.0, device=device)
            # entropy_loss
            entropy_loss = compute_softmax_entropy(adapt_outputs).mean(0) + torch.sum(mean_softmax * torch.log(mean_softmax + 1e-5))
            # mmd_loss
            mmd_loss = calculate_mmd_loss(origin_features, adapt_features)
            
            # Total loss = origin_loss + pseudo_loss + entropy_loss + mmd_loss
            total_loss = classification_loss + pseudo_loss + entropy_loss + mmd_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            origin_loss_sum += classification_loss.data.cpu().numpy() * origin_outputs.shape[0]
            mmd_loss_sum += mmd_loss.data.cpu().numpy() * origin_outputs.shape[0]
            entropy_loss_sum += entropy_loss.data.cpu().numpy() * origin_outputs.shape[0]
            pseudo_loss_sum += pseudo_loss.data.cpu().numpy() * origin_outputs.shape[0]
            total_samples += adapt_outputs.shape[0]
        
            # Update progress bar description with current loss
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}"})
        
        # evaluate after each epoch
        epoch_result = evaluate_model(model, test_data_loader, metrics, device, desc="Validating")
        
        # Format output for better readability
        res_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in epoch_result.items()])
        print(f"Epoch {epoch+1:03d} | {res_str}")
        
        if epoch_result["F1-score"] > best_f1_score:
            best_f1_score = epoch_result["F1-score"]
            best_f1_score_epoch = epoch
        if "Accuracy" in epoch_result and epoch_result["Accuracy"] > best_accuracy:
            best_accuracy = epoch_result["Accuracy"]
            best_accuracy_epoch = epoch

    print(f"{'='*20} Adaptation Finished {'='*20}\n")
    return best_f1_score, best_f1_score_epoch, best_accuracy, best_accuracy_epoch

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
parser.add_argument("--scenario", type=str, default="Closed-world", help="Attack scenario, options=[Closed-world, Open-world]")

# Input parameters
parser.add_argument("--train_file", type=str, default="train", help="Train file")
parser.add_argument("--test_file", type=str, default="test", help="Test file")
parser.add_argument("--feature", type=str, default="DIR", help="Feature type, options=[DIR, DT, DT2, TAM, TAF]")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")

# Optimization parameters
parser.add_argument("--num_workers", type=int, default=10, help="Data loader num workers")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")

# Output parameters
parser.add_argument("--eval_method", type=str, default="common", help="Method used in the evaluation, options=[common, kNN, holmes]")
parser.add_argument('--eval_metrics', nargs='+', required=True, type=str, help="Evaluation metrics, options=[Accuracy, Precision, Recall, F1-score, P@min, r-Precision]")
parser.add_argument("--log_path", type=str, default="./logs/", help="Log path")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Location of model checkpoints")
parser.add_argument("--load_name", type=str, default="base", help="Name of the model file")
parser.add_argument("--result_file", type=str, default="result", help="File to save test results")
parser.add_argument("--gmm_threshold", type=float, default=0.6, help="GMM threshold")
parser.add_argument("--model_save_name", type=str, default="proteus", help="Name used to save the model")
parser.add_argument("--limit_n", type=int, default=30, help="Limit number of samples for fine-tuning")

# Parse arguments
args = parser.parse_args()

# Ensure the specified device is available, fallback to available device if not
if args.device.startswith("cuda") and not torch.cuda.is_available():
    device = torch.device("cpu")
else:
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
print(f"\n{'='*20} Configuration {'='*20}")
print(f"Dataset: {args.dataset}")
print(f"Model: {args.model}")
print(f"Device: {device}")
print(f"Test File: {os.path.join(dataset_path, f'{args.test_file}.npz')}")

train_data, train_labels = data_processor.load_data(os.path.join(dataset_path, f"{args.train_file}.npz"), args.feature, args.seq_len, args.num_tabs)
test_data, test_labels = data_processor.load_data(os.path.join(dataset_path, f"{args.test_file}.npz"), args.feature, args.seq_len, args.num_tabs)
adapt_data, adapt_labels = test_data, test_labels

if args.num_tabs == 1:
    num_classes = len(np.unique(test_labels))
    assert num_classes == test_labels.max() + 1, "Labels are not continuous"
else:
    num_classes = test_labels.shape[1]

if args.limit_n > 0:
    print(f"Truncating adaptation data to {args.limit_n} samples per class...")
    indices = []
    labels_np = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels
    unique_labels = np.unique(labels_np)
    
    for label in unique_labels:
        label_indices = np.where(labels_np == label)[0]
        if len(label_indices) > args.limit_n:
            label_indices = label_indices[:args.limit_n]
        indices.extend(label_indices)
    
    indices = sorted(indices)
    adapt_data = test_data[indices]
    adapt_labels = test_labels[indices]
    print(f"Truncated adaptation data. Total samples: {len(adapt_labels)}")

num_classes = len(np.unique(test_labels))

# Print dataset information
print(f"Train data shape: X={train_data.shape}, y={train_labels.shape}")
print(f"Adapt data shape: X={adapt_data.shape}, y={adapt_labels.shape}")
print(f"Test data shape: X={test_data.shape}, y={test_labels.shape}")
print(f"Number of classes: {num_classes}")
print(f"{'='*55}\n")

# Load data into iterators
origin_data_loader = data_processor.load_iter(train_data, train_labels, args.batch_size, True, args.num_workers)
adapt_data_loader = data_processor.load_iter(adapt_data, torch.zeros_like(adapt_labels), args.batch_size, True, args.num_workers)
test_data_loader = data_processor.load_iter(test_data, test_labels, args.batch_size, False, args.num_workers)

# Initialize model, optimizer, and loss function
if args.model in ["BAPM", "TMWF"]:
    model = eval(f"models.{args.model}")(num_classes, args.num_tabs)
else:
    model = eval(f"models.{args.model}")(num_classes)

model.load_state_dict(torch.load(os.path.join(ckp_path, f"{args.load_name}.pth"), map_location="cpu"))
model.to(device)

# Evaluation before adaptation
print(f"{'='*20} Initial Evaluation {'='*20}")
initial_result = evaluate_model(model, test_data_loader, args.eval_metrics, device, desc="Initial Eval")
res_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in initial_result.items()])
print(f"Result: {res_str}")
print(f"{'='*60}\n")

# Model adaptation
best_f1_score, best_f1_score_epoch, best_accuracy, best_accuracy_epoch = adapt_model(model, test_data, adapt_data_loader, origin_data_loader, test_data_loader, args.eval_metrics, device, args.gmm_threshold)

# Evaluation after adaptation
print(f"{'='*20} Final Evaluation {'='*20}")
final_result = evaluate_model(model, test_data_loader, args.eval_metrics, device, desc="Final Eval")
final_result["best_f1_score"] = best_f1_score
final_result["best_f1_epoch"] = best_f1_score_epoch
final_result["best_accuracy"] = best_accuracy
final_result["best_accuracy_epoch"] = best_accuracy_epoch

res_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in final_result.items()])
print(f"Result: {res_str}")
print(f"{'='*58}\n")

# Save model
model_save_path = os.path.join(ckp_path, f"{args.model_save_name}.pth")
torch.save(model.state_dict(), model_save_path)

# Save results to file
with open(output_file, "w") as result_file:
    json.dump(final_result, result_file, indent=4)