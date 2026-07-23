import argparse
import copy
import json
import os
import random
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

from WFlib import models
from WFlib.tools import data_processor, model_utils
from WFlib.tools.debug import print_energy_detection_stats, print_gmm_pseudo_stats

from other_experiments import ablation_losses as loss_utils

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


@dataclass(frozen=True)
class StageSpec:
	key: str
	name: str
	use_energy_filter: bool
	use_adapt: bool
	use_mmd: bool
	use_pseudo: bool
	use_entropy: bool
	use_energy_loss: bool


STAGE_SPECS = {
	"base": StageSpec(
		key="base",
		name="Base",
		use_energy_filter=True,
		use_adapt=False,
		use_mmd=False,
		use_pseudo=False,
		use_entropy=False,
        use_energy_loss = False,
	),
	"energy": StageSpec(
		key="energy",
		name="Base + Energy OSR",
		use_energy_filter=True,
		use_adapt=True,
		use_mmd=False,
		use_pseudo=False,
		use_entropy=False,
        use_energy_loss = True,
	),
	"mmd": StageSpec(
		key="mmd",
		name="Base + Energy OSR + MMD",
		use_energy_filter=True,
		use_adapt=True,
		use_mmd=True,
		use_pseudo=False,
		use_entropy=False,
        use_energy_loss = True,
	),
	"pl": StageSpec(
		key="pl",
		name="Base + Energy OSR + MMD + Entropy",
		use_energy_filter=True,
		use_adapt=True,
		use_mmd=True,
		use_pseudo=False,
		use_entropy=True,
        use_energy_loss = True,
	),
	"full": StageSpec(
		key="full",
		name="Full",
		use_energy_filter=True,
		use_adapt=True,
		use_mmd=True,
		use_pseudo=True,
		use_entropy=True,
        use_energy_loss = True,
	),
}


def parse_args():
	parser = argparse.ArgumentParser(description="Proteus ablation runner")
	parser.add_argument("--dataset", type=str, required=True)
	parser.add_argument("--model", type=str, required=True)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--num_tabs", type=int, default=1)

	parser.add_argument("--train_file", type=str, required=True)
	parser.add_argument("--extra_train_file", type=str, default=None)
	parser.add_argument("--tune_file", type=str, required=True)
	parser.add_argument("--extra_tune_file", type=str, default=None)
	parser.add_argument("--test_file", type=str, required=True)
	parser.add_argument("--extra_test_file", type=str, default=None)
	parser.add_argument("--feature", type=str, default="DIR")
	parser.add_argument("--seq_len", type=int, default=5000)

	parser.add_argument("--num_workers", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--adapt_epochs", type=int, default=100)
	parser.add_argument("--adapt_lr", type=float, default=1e-4)
	parser.add_argument("--split_refresh", type=int, default=5)
	parser.add_argument("--pseudo_refresh", type=int, default=5)
	parser.add_argument("--pseudo_threshold", type=float, default=0.6)

	parser.add_argument("--energy_temperature", type=float, default=1.0)
	parser.add_argument("--tau_pct", type=float, default=99.0)
	parser.add_argument("--tau_ema", type=float, default=0.99)
	parser.add_argument("--energy_m_in", type=float, default=-12.0)
	parser.add_argument("--energy_m_out", type=float, default=-2.0)
	parser.add_argument("--energy_loss_weight", type=float, default=0.2)

	parser.add_argument("--eval_metrics", nargs="+", required=True, type=str)
	parser.add_argument("--log_path", type=str, default="./logs/")
	parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
	parser.add_argument("--load_name", type=str, default="base")
	parser.add_argument("--result_file", type=str, default="ablation")
	parser.add_argument("--model_save_prefix", type=str, default="ablation")
	parser.add_argument("--tune_unknown_ratio", type=float, default=2.0)
	parser.add_argument("--tune_known_keep_ratio", type=float, default=1.0)
	parser.add_argument("--fix_seed", type=int, default=20070206)
	parser.add_argument(
		"--stages",
		nargs="+",
		default=["base", "energy", "mmd", "pl", "full"],
		choices=list(STAGE_SPECS.keys()),
	)
	parser.add_argument("--save_stage_checkpoints", action="store_true")
	return parser.parse_args()


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def load_and_merge(file_list, feature, seq_len, num_tabs):
	all_data, all_labels = [], []
	for file_path in file_list:
		npz_path = file_path if file_path.endswith(".npz") else f"{file_path}.npz"
		data, labels = data_processor.load_data(npz_path, feature, seq_len, num_tabs)
		all_data.append(data)
		all_labels.append(labels)
	return torch.cat(all_data), torch.cat(all_labels)


def infer_unknown_label(labels):
	return int(labels.max().item())


def rebalance_tune_data(data, labels, args, unknown_label):
	if labels.ndim != 1:
		raise ValueError("tune_unknown_ratio only supports 1D labels")

	if args.tune_unknown_ratio <= 0:
		raise ValueError("tune_unknown_ratio must be positive")

	if not (0 < args.tune_known_keep_ratio <= 1):
		raise ValueError("tune_known_keep_ratio must be in the range (0, 1]")

	known_idx = torch.where(labels != unknown_label)[0]
	unknown_idx = torch.where(labels == unknown_label)[0]
	known_count = int(known_idx.numel())
	unknown_count = int(unknown_idx.numel())

	if known_count == 0 or unknown_count == 0:
		print(
			f"Tune ratio skipped: known={known_count}, unknown={unknown_count}, "
			f"requested 1:{args.tune_unknown_ratio:g}"
		)
		return data, labels

	keep_known = min(known_count, int(unknown_count / args.tune_unknown_ratio))
	keep_unknown = min(unknown_count, int(round(keep_known * args.tune_unknown_ratio)))

	if keep_known == 0 or keep_unknown == 0:
		keep_known = min(known_count, 1)
		keep_unknown = min(unknown_count, max(1, int(round(keep_known * args.tune_unknown_ratio))))

	known_perm = torch.randperm(known_count)[:keep_known]
	unknown_perm = torch.randperm(unknown_count)[:keep_unknown]
	known_selected = known_idx[known_perm]
	unknown_selected = unknown_idx[unknown_perm]

	if args.tune_known_keep_ratio < 1.0:
		keep_known_final = max(1, int(known_selected.numel() * args.tune_known_keep_ratio))
		keep_unknown_final = max(1, int(unknown_selected.numel() * args.tune_known_keep_ratio))
		known_selected = known_selected[torch.randperm(known_selected.numel())[:keep_known_final]]
		unknown_selected = unknown_selected[torch.randperm(unknown_selected.numel())[:keep_unknown_final]]

	selected_idx = torch.cat([known_selected, unknown_selected])
	selected_idx = selected_idx[torch.randperm(selected_idx.numel())]

	actual_ratio = int(unknown_selected.numel()) / max(int(known_selected.numel()), 1)
	print(
		f"Tune sampling: requested 1:{args.tune_unknown_ratio:g}, "
		f"known={int(known_selected.numel())}, unknown={int(unknown_selected.numel())}, "
		f"actual 1:{actual_ratio:.2f}"
	)
	return data[selected_idx], labels[selected_idx]


def collect_energies(model, data, batch_size, device, temperature):
	model.eval()
	chunks = []
	with torch.no_grad():
		for idx in range(0, len(data), batch_size):
			batch = torch.as_tensor(data[idx : idx + batch_size], dtype=torch.float32, device=device)
			logits, _ = model(batch)
			chunks.append(loss_utils.energy_score(logits, temperature).cpu().numpy())
	return np.concatenate(chunks)


def compute_energy_threshold(model, source_data, args, device):
	energies = collect_energies(model, source_data, args.batch_size, device, args.energy_temperature)
	return float(np.percentile(energies, args.tau_pct))


def predict_unknown_mask(model, eval_data, tau, args, device, eval_labels=None):
	known_mask = collect_energies(model, eval_data, args.batch_size, device, args.energy_temperature) < tau
	if eval_labels is not None:
		eval_labels_np = np.asarray(eval_labels)
		print_energy_detection_stats(known_mask, eval_labels_np, int(eval_labels_np.max()), tau)
	return ~known_mask


def gmm_clean_probs(model, data_loader, device):
	entropies = []
	predictions = []
	model.eval()
	with torch.no_grad():
		for batch in data_loader:
			logits, _ = model(batch[0].to(device))
			entropies.append(loss_utils.softmax_entropy(logits).cpu().numpy())
			predictions.append(logits.argmax(dim=1).cpu().numpy())

	entropies = np.concatenate(entropies).reshape(-1)
	predictions = np.concatenate(predictions).reshape(-1)
	if entropies.size == 1:
		return np.ones(1, dtype=np.float32), torch.tensor(predictions, dtype=torch.int64)

	entropies = (entropies - entropies.min()) / max(entropies.max() - entropies.min(), 1e-6)
	gmm = GaussianMixture(n_components=2, tol=1e-6, random_state=0)
	gmm.fit(entropies.reshape(-1, 1))
	probabilities = gmm.predict_proba(entropies.reshape(-1, 1))
	clean_index = np.argmin(gmm.means_.reshape(-1))
	return probabilities[:, clean_index], torch.tensor(predictions, dtype=torch.int64)


def make_pseudo_loader(probs, data, predictions, threshold, batch_size, num_workers):
	selected_mask = np.asarray(probs) >= float(threshold)
	if not np.any(selected_mask):
		selected_mask[np.argmax(probs)] = True

	inputs = torch.as_tensor(data[selected_mask], dtype=torch.float32)
	labels = predictions[selected_mask]
	return data_processor.load_iter(inputs, labels, batch_size, True, num_workers), selected_mask


def infinite_iter(loader):
	while True:
		yield from loader


def build_model(model_name, num_classes, num_tabs):
	if model_name in ["BAPM", "TMWF"]:
		return getattr(models, model_name)(num_classes, num_tabs)
	return getattr(models, model_name)(num_classes)


def load_stage_model(args, num_classes, device):
	checkpoint_dir = os.path.join(args.checkpoints, args.dataset, args.model)
	checkpoint_file = os.path.join(checkpoint_dir, f"{args.load_name}.pth")
	model = build_model(args.model, num_classes, args.num_tabs)
	model.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))
	model.to(device)
	return model, checkpoint_file


def evaluate_raw(model, eval_data, eval_labels, args, device):
	predictions = []
	model.eval()
	with torch.no_grad():
		for idx in range(0, len(eval_data), args.batch_size):
			batch = torch.as_tensor(eval_data[idx : idx + args.batch_size], dtype=torch.float32, device=device)
			logits, _ = model(batch)
			predictions.append(logits.argmax(dim=1).cpu().numpy())
	predictions = np.concatenate(predictions)
	metrics = model_utils.measure_open_set(eval_labels, predictions, args.eval_metrics, args.num_tabs, open_set=False)
	return metrics


def evaluate_open_set(model, eval_data, eval_labels, unknown_mask, args, device):
	predictions = []
	model.eval()
	with torch.no_grad():
		for idx in range(0, len(eval_data), args.batch_size):
			batch = torch.as_tensor(eval_data[idx : idx + args.batch_size], dtype=torch.float32, device=device)
			logits, _ = model(batch)
			predictions.append(logits.argmax(dim=1).cpu().numpy())
	predictions = np.concatenate(predictions)
	full_predictions = predictions.copy()
	full_predictions[np.asarray(unknown_mask, dtype=bool)] = infer_unknown_label(eval_labels)
	metrics = model_utils.measure_open_set(eval_labels, full_predictions, args.eval_metrics, args.num_tabs, open_set=True)
	return metrics


def run_energy_eval(model, stage_name, eval_data, eval_labels, source_known_data, args, device):
	tau = compute_energy_threshold(model, source_known_data, args, device)
	print(f"\n[{stage_name}] Energy threshold: tau={tau:.4f}")
	unknown_mask = predict_unknown_mask(model, eval_data, tau, args, device, eval_labels=eval_labels)
	metrics = evaluate_open_set(model, eval_data, np.asarray(eval_labels), unknown_mask, args, device)
	print(f"[{stage_name}] {', '.join(f'{k}: {v:.4f}' for k, v in metrics.items())}")
	return metrics, tau, unknown_mask


def refresh_known_split(model, tune_data, tune_labels_np, tau, args, device):
	known_mask = collect_energies(model, tune_data, args.batch_size, device, args.energy_temperature) < tau
	if not np.any(known_mask):
		energies = collect_energies(model, tune_data, args.batch_size, device, args.energy_temperature)
		known_mask[np.argmin(energies)] = True
	unknown_mask = ~known_mask
	tune_unknown_label = int(np.max(tune_labels_np))
	print_energy_detection_stats(known_mask, tune_labels_np, tune_unknown_label, tau)
	known_data = tune_data[known_mask]
	known_loader = data_processor.load_iter(
		torch.as_tensor(known_data, dtype=torch.float32),
		torch.zeros(len(known_data), dtype=torch.int64),
		args.batch_size,
		True,
		args.num_workers,
	)
	return known_mask, unknown_mask, known_data, known_loader


def adapt_stage(backbone, train_data, train_labels, tune_data, tune_labels, tau, stage, args, device):
	optimizer = torch.optim.Adam(backbone.parameters(), lr=args.adapt_lr)
	criterion = nn.CrossEntropyLoss()
	tune_labels_np = np.asarray(tune_labels)
	train_unknown_label = infer_unknown_label(train_labels)
	source_known_mask = train_labels != train_unknown_label
	primary_metric = args.eval_metrics[0]

	src_loader = data_processor.load_iter(train_data, train_labels, args.batch_size, True, args.num_workers)
	src_iter = infinite_iter(src_loader)
	tgt_loader = data_processor.load_iter(
		torch.as_tensor(tune_data, dtype=torch.float32),
		torch.zeros(len(tune_data), dtype=torch.int64),
		args.batch_size,
		True,
		args.num_workers,
	)

	if stage.use_energy_filter:
		known_mask, unknown_mask, known_data, known_loader = refresh_known_split(
			backbone, tune_data, tune_labels_np, tau, args, device
		)
	else:
		known_mask = np.ones(len(tune_data), dtype=bool)
		unknown_mask = ~known_mask
		known_data = tune_data
		known_loader = data_processor.load_iter(
			torch.as_tensor(known_data, dtype=torch.float32),
			torch.zeros(len(known_data), dtype=torch.int64),
			args.batch_size,
			True,
			args.num_workers,
		)

	known_iter = infinite_iter(known_loader)
	pseudo_iter = None
	tau_center = tau
	best_score = float("-inf")
	best_state = copy.deepcopy(backbone.state_dict())
	best_tune_metrics = {}

	for epoch in range(args.adapt_epochs):
		if stage.use_energy_filter and epoch % args.split_refresh == 0:
			tau_new = compute_energy_threshold(backbone, train_data[source_known_mask], args, device)
			tau = args.tau_ema * tau + (1.0 - args.tau_ema) * tau_new
			tau_center = tau
			known_mask, unknown_mask, known_data, known_loader = refresh_known_split(
				backbone, tune_data, tune_labels_np, tau, args, device
			)
			known_iter = infinite_iter(known_loader)

		if stage.use_pseudo and epoch % args.pseudo_refresh == 0:
			tmp_loader = torch.utils.data.DataLoader(
				torch.utils.data.TensorDataset(
					torch.as_tensor(known_data, dtype=torch.float32),
					torch.zeros(len(known_data), dtype=torch.int64),
				),
				batch_size=args.batch_size,
				shuffle=False,
				num_workers=args.num_workers,
			)
			confidences, predictions = gmm_clean_probs(backbone, tmp_loader, device)
			pseudo_loader, selected_mask = make_pseudo_loader(
				confidences,
				known_data,
				predictions,
				args.pseudo_threshold,
				args.batch_size,
				args.num_workers,
			)
			print_gmm_pseudo_stats(
				confidences,
				predictions.numpy(),
				tune_labels_np[known_mask],
				args.pseudo_threshold,
				prefix=f" [{stage.key}]",
			)
			pseudo_iter = infinite_iter(pseudo_loader)

		backbone.train()
		epoch_losses = {"cls": 0.0, "mmd": 0.0, "pse": 0.0, "ent": 0.0, "eng": 0.0}
		sample_count = 0

		for tgt_batch in tqdm(
			tgt_loader,
			desc=f"{stage.key} {epoch + 1:03d}/{args.adapt_epochs}",
			leave=False,
			dynamic_ncols=True,
		):
			src_batch = next(src_iter)
			src_x, src_y = src_batch[0].to(device), src_batch[1].to(device)
			tgt_x = tgt_batch[0].to(device)

			optimizer.zero_grad()
			src_out, src_feat = backbone(src_x)
			tgt_out, _ = backbone(tgt_x)

			losses = {}
			losses["cls"] = loss_utils.source_classification_loss(src_out, src_y, train_unknown_label, criterion)

			kn_out = kn_feat = None
			if stage.use_mmd or stage.use_entropy:
				kn_x = next(known_iter)[0].to(device)
				kn_out, kn_feat = backbone(kn_x)

			if stage.use_mmd:
				known_src_mask = src_y != train_unknown_label
				losses["mmd"] = loss_utils.mmd_loss(src_feat[known_src_mask], kn_feat)
			if stage.use_entropy:
				losses["ent"] = loss_utils.target_entropy_loss(kn_out)
			if stage.use_pseudo and pseudo_iter is not None:
				ps_x, ps_y = [tensor.to(device) for tensor in next(pseudo_iter)]
				losses["pse"] = loss_utils.pseudo_label_loss(backbone, ps_x, ps_y, criterion)
			if stage.use_energy_loss:
				losses["eng"] = loss_utils.source_target_energy_loss(
					src_out,
					src_y,
					tgt_out,
					train_unknown_label,
					tau_center,
					args.energy_temperature,
					args.energy_m_in,
					args.energy_m_out,
				)

			total = losses["cls"]
			if "mmd" in losses:
				total = total + losses["mmd"]
			if "ent" in losses:
				ent_weight = 0.1 if epoch < max(args.adapt_epochs // 10, 1) else 1.0
				total = total + ent_weight * losses["ent"]
			if "pse" in losses:
				pseudo_weight = 0.1 if epoch < max(args.adapt_epochs // 10, 1) else 1.0
				total = total + pseudo_weight * losses["pse"]
			if "eng" in losses:
				total = total + args.energy_loss_weight * losses["eng"]

			total.backward()
			optimizer.step()

			batch_size = src_out.size(0)
			for key in epoch_losses:
				if key in losses:
					epoch_losses[key] += float(losses[key].detach().cpu().item()) * batch_size
			sample_count += batch_size

		tune_metrics = evaluate_open_set(backbone, tune_data, tune_labels_np, unknown_mask, args, device)
		metric_value = float(tune_metrics.get(primary_metric, 0.0))
		if metric_value > best_score:
			best_score = metric_value
			best_state = copy.deepcopy(backbone.state_dict())
			best_tune_metrics = dict(tune_metrics)

		loss_str = ", ".join(
			f"{key}: {epoch_losses[key] / max(sample_count, 1):.4f}" for key in ["cls", "mmd", "pse", "ent", "eng"]
		)
		metric_str = ", ".join(f"{key}: {val:.4f}" for key, val in tune_metrics.items())
		print(f"  Epoch {epoch + 1:03d} | {metric_str} | {loss_str}")

	backbone.load_state_dict(best_state)
	return tau, best_tune_metrics


def save_stage_checkpoint(model, args, stage_key):
	checkpoint_dir = os.path.join(args.checkpoints, args.dataset, args.model)
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint_path = os.path.join(checkpoint_dir, f"{args.model_save_prefix}_{stage_key}.pth")
	torch.save(model.state_dict(), checkpoint_path)
	return checkpoint_path


def serializable_metrics(metrics):
	return {key: float(value) for key, value in metrics.items()}


def run_stage(stage, args, device, train_data, train_labels, tune_data, tune_labels, test_data, test_labels, source_known_data):
	model, checkpoint_file = load_stage_model(args, int(train_labels[train_labels != infer_unknown_label(train_labels)].max()) + 1, device)
	stage_result = {
		"stage": stage.name,
		"checkpoint": checkpoint_file,
		"used_energy_filter": stage.use_energy_filter,
		"used_adaptation": stage.use_adapt,
		"used_mmd": stage.use_mmd,
		"used_pseudo": stage.use_pseudo,
		"used_entropy": stage.use_entropy,
		"used_energy_loss": stage.use_energy_loss,
	}

	start_time = time.time()
	if stage.use_adapt:
		initial_tau = compute_energy_threshold(model, source_known_data, args, device) if stage.use_energy_filter else None
		final_tau, best_tune_metrics = adapt_stage(
			model,
			train_data,
			train_labels,
			tune_data,
			tune_labels,
			initial_tau,
			stage,
			args,
			device,
		)
		test_metrics, _, _ = run_energy_eval(model, stage.name, test_data, test_labels, source_known_data, args, device)
		stage_result["best_tune_metrics"] = serializable_metrics(best_tune_metrics)
		stage_result["final_tau"] = None if final_tau is None else float(final_tau)
	elif stage.use_energy_filter:
		test_metrics, tau, _ = run_energy_eval(model, stage.name, test_data, test_labels, source_known_data, args, device)
		stage_result["final_tau"] = float(tau)
	else:
		test_metrics = evaluate_raw(model, test_data, np.asarray(test_labels), args, device)
		print(f"\n[{stage.name}] {', '.join(f'{k}: {v:.4f}' for k, v in test_metrics.items())}")

	stage_result["test_metrics"] = serializable_metrics(test_metrics)
	stage_result["elapsed_seconds"] = round(time.time() - start_time, 3)

	if args.save_stage_checkpoints and stage.use_adapt:
		stage_result["saved_checkpoint"] = save_stage_checkpoint(model, args, stage.key)

	return stage_result


def main():
	args = parse_args()
	set_seed(args.fix_seed)

	if args.num_tabs != 1:
		raise ValueError("Ablation runner currently supports only num_tabs=1.")

	device = torch.device(args.device)
	dataset_path = os.path.join("./datasets", args.dataset)

	train_files = [os.path.join(dataset_path, args.train_file)]
	tune_files = [os.path.join(dataset_path, args.tune_file)]
	test_files = [os.path.join(dataset_path, args.test_file)]

	if args.extra_train_file:
		train_files.append(os.path.join("./datasets", args.extra_train_file))
	if args.extra_tune_file:
		tune_files.append(os.path.join("./datasets", args.extra_tune_file))
	if args.extra_test_file:
		test_files.append(os.path.join("./datasets", args.extra_test_file))

	train_data, train_labels = load_and_merge(train_files, args.feature, args.seq_len, args.num_tabs)
	tune_data, tune_labels = load_and_merge(tune_files, args.feature, args.seq_len, args.num_tabs)
	test_data, test_labels = load_and_merge(test_files, args.feature, args.seq_len, args.num_tabs)

	tune_unknown_label = infer_unknown_label(tune_labels)
	tune_data, tune_labels = rebalance_tune_data(tune_data, tune_labels, args, tune_unknown_label)

	train_unknown_label = infer_unknown_label(train_labels)
	source_known_data = train_data[train_labels != train_unknown_label]

	print(f"Dataset: {args.dataset}, Model: {args.model}, Device: {device}")
	print(f"Train: {train_data.shape}, Tune: {tune_data.shape}, Test: {test_data.shape}")
	print(f"Stages: {', '.join(args.stages)}")

	stage_results = []
	for stage_key in args.stages:
		stage = STAGE_SPECS[stage_key]
		print(f"\n{'=' * 20} {stage.name} {'=' * 20}")
		stage_results.append(
			run_stage(
				stage,
				args,
				device,
				train_data,
				train_labels,
				tune_data,
				tune_labels,
				test_data,
				test_labels,
				source_known_data,
			)
		)

	output_dir = os.path.join(args.log_path, "OtherExperiments", "Ablation")
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, f"{args.result_file}.json")
	with open(output_path, "w", encoding="utf-8") as handle:
		json.dump(stage_results, handle, indent=4)

	print(f"\nSaved ablation results to {output_path}")
	for stage_result in stage_results:
		metric_str = ", ".join(
			f"{key}: {value:.4f}" for key, value in stage_result["test_metrics"].items()
		)
		print(f"{stage_result['stage']}: {metric_str}")


if __name__ == "__main__":
	main()
