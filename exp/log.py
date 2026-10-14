"""Proteus logging and detached diagnostics.

Console output contains epoch loss/accuracy only. Detailed training metrics,
configuration and optional target geometry go to a single readable log file.
Stage 2 trains a class-conditional affine translator with local OT. Legacy
prototype diagnostic helpers remain available for older experiments.
"""

from __future__ import annotations

import logging
import math
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F


def _normalise(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.reshape(features.shape[0], -1), dim=1, eps=1e-8)


def _class_mean(values: torch.Tensor) -> float | None:
    return float(values.mean().item()) if values.numel() else None


def _within_class_cosine(
    features: torch.Tensor, labels: torch.Tensor
) -> float | None:
    """Mean pairwise cosine similarity within classes.

    Singleton classes do not provide a pair and are skipped.  The result is
    averaged over pairs, so a class with many support examples contributes
    more pairs; with few-shot support this is still a small, transparent
    diagnostic rather than a training statistic.
    """
    features = _normalise(features).detach()
    labels = labels.reshape(-1).to(features.device)
    pair_values = []
    for label in torch.unique(labels, sorted=True):
        current = features[labels == label]
        if current.shape[0] >= 2:
            similarity = current @ current.T
            upper = similarity[torch.triu_indices(current.shape[0], current.shape[0], offset=1, device=current.device).unbind()]
            pair_values.append(upper)
    if not pair_values:
        return None
    return float(torch.cat(pair_values).mean().item())


@torch.no_grad()
def compute_prototype_diagnostics(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    evaluation_features: torch.Tensor,
    evaluation_labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> Dict[str, Any]:
    """Compute prototype geometry and nearest-prototype accuracy.

    ``support_to_prototype_distance`` and
    ``evaluation_to_prototype_distance`` contain one mean cosine distance per
    class.  Missing classes are represented by ``None``.  The nearest
    prototype prediction is evaluated only on the supplied evaluation set.
    """
    support = _normalise(support_features)
    evaluation = _normalise(evaluation_features)
    prototypes = _normalise(prototypes)
    support_labels = support_labels.reshape(-1).to(support.device).long()
    evaluation_labels = evaluation_labels.reshape(-1).to(evaluation.device).long()

    prototype_similarity = evaluation @ prototypes.T
    nearest_prediction = prototype_similarity.argmax(dim=1)
    nearest_accuracy = (
        float((nearest_prediction == evaluation_labels).float().mean().item())
        if evaluation.shape[0]
        else None
    )

    inter = prototypes @ prototypes.T
    off_diagonal = ~torch.eye(
        prototypes.shape[0], dtype=torch.bool, device=prototypes.device
    )
    inter_similarity = (
        float(inter[off_diagonal].mean().item()) if off_diagonal.any() else None
    )

    support_distances = 1.0 - support @ prototypes.T
    evaluation_distances = 1.0 - evaluation @ prototypes.T
    classes = range(prototypes.shape[0])
    support_by_class: Dict[str, float | None] = {}
    evaluation_by_class: Dict[str, float | None] = {}
    for class_id in classes:
        support_mask = support_labels == class_id
        evaluation_mask = evaluation_labels == class_id
        support_by_class[str(class_id)] = _class_mean(
            support_distances[support_mask, class_id]
        )
        evaluation_by_class[str(class_id)] = _class_mean(
            evaluation_distances[evaluation_mask, class_id]
        )

    return {
        "within_class_cosine_similarity": _within_class_cosine(
            support, support_labels
        ),
        "inter_class_prototype_cosine_similarity": inter_similarity,
        "prototype_nearest_neighbor_accuracy": nearest_accuracy,
        "support_to_prototype_distance": support_by_class,
        "evaluation_to_prototype_distance": evaluation_by_class,
    }


_LOGGER = logging.getLogger("proteus.training")
_LOGGER.addHandler(logging.NullHandler())
_LOGGER.propagate = False


def _format(value):
    if value is None:
        return "N/A"
    if isinstance(value, (float, int)):
        return f"{value:.6f}" if math.isfinite(value) else "N/A"
    return str(value)


def configure_logging(path, config, device):
    """Append a separated run to one UTF-8 file; never attach a console handler."""
    for handler in list(_LOGGER.handlers):
        handler.close()
        _LOGGER.removeHandler(handler)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.info("\n%s\nProteus run | %s\n%s", "=" * 80,
                 datetime.now().isoformat(timespec="seconds"), "=" * 80)
    _LOGGER.info("Configuration:")
    for key, value in sorted(config.items()):
        _LOGGER.info("  %-28s : %s", key, value)
    _LOGGER.info("  %-28s : %s", "effective_device", device)
    _LOGGER.info(
        "\nMetric guide:\n"
        "  Stage 1: frozen target support bank; no optimization.\n"
        "  Stage 2 loss: ot + map_reg_weight * regularization.\n"
        "  ot: class-averaged local cosine/time OT with entropy regularization.\n"
        "  regularization: mean squared affine scale delta and bias.\n"
        "  Stage 2 epoch metrics: equally weighted batch means; no accuracy.\n"
        "  Stage 3 loss: target + augmented_loss_weight * source_weighted.\n"
        "  target: support CE; source_weighted: reliability-weighted source CE.\n"
        "  Stage 3 epoch metrics: weighted by the number of source samples\n"
        "  (target samples for the target-only ablation).\n"
        "  acc: held-out target evaluation, never used for optimization.\n"
        "  N/A means unavailable.\n"
    )


def log_epoch(stage, epoch, epochs, values, accuracy=None, acc_scope=""):
    tag = "[Final]" if stage == "Final" else f"[Stage {stage}][Epoch {epoch}/{epochs}]"
    summary = tag
    if "loss" in values:
        summary += f" loss={_format(values['loss'])}"
    if accuracy is not None:
        summary += f" acc={_format(accuracy)}"
    print(summary, flush=True)
    _LOGGER.info(summary)
    if acc_scope:
        _LOGGER.info("  Accuracy scope: %s", acc_scope)
    for name, value in values.items():
        if name != "loss":
            _LOGGER.info("  %-24s : %s", name, _format(value))
    _LOGGER.info("")


class EpochMetrics:
    """Accumulate scalar batch metrics, detached from the training graph.

    The default weight gives an equal-batch average (Stage 2). Passing a
    batch size gives a sample-weighted average (Stage 3). Each metric keeps
    its own denominator so optional metrics can be omitted in some batches.
    """

    def __init__(self):
        self.sums = {}
        self.weights = {}
        self.count = 0.0
        self.batches = 0

    def update(self, weight=1, **values):
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Metric weight must be finite and non-negative")
        if weight == 0:
            return
        scalars = {
            name: float(value.detach().item()) if isinstance(value, torch.Tensor)
            else float(value)
            for name, value in values.items()
        }
        for name, value in scalars.items():
            self.sums[name] = self.sums.get(name, 0.0) + weight * value
            self.weights[name] = self.weights.get(name, 0.0) + weight
        self.count += weight
        self.batches += 1

    def log(self, stage, epoch, epochs, accuracy=None, acc_scope=""):
        values = {name: total / self.weights[name] for name, total in self.sums.items()}
        values.setdefault("loss", None)
        values.update(batches=self.batches, total_weight=self.count)
        log_epoch(stage, epoch, epochs, values, accuracy, acc_scope)
        return values


def _target_bank_summary(bank):
    """Summarize support metadata without serializing the large feature bank."""
    labels = bank.support_labels.detach().cpu().long().reshape(-1)
    counts = torch.bincount(labels, minlength=len(bank.descriptors))
    return {
        "num_classes": len(bank.descriptors),
        "support_samples": len(labels),
        "support_map_shape": list(bank.support_maps.shape),
        "per_class": {
            str(class_id): {
                "support_count": int(counts[class_id]),
                "descriptor_count": len(descriptors),
                "variance": float(bank.variances[class_id].detach().cpu()),
                "weight": float(bank.weights[class_id].detach().cpu()),
            }
            for class_id, descriptors in enumerate(bank.descriptors)
        },
    }


def log_target_bank(bank):
    """Log the frozen support bank and per-class reliability statistics."""
    summary = _target_bank_summary(bank)
    _LOGGER.info("%s\n[Stage 1] Target support bank\n%s", "-" * 80, "-" * 80)
    for name, value in summary.items():
        if name != "per_class":
            _LOGGER.info("  %-24s : %s", name, value)
    _LOGGER.info("  Variance and reliability weights use target support only.")
    _LOGGER.info("\n  %-8s %12s %12s %16s %16s",
                 "Class", "Support", "Descriptors", "Variance", "Weight")
    for label, row in summary["per_class"].items():
        _LOGGER.info("  %-8s %12d %12d %16s %16s", label,
                     row["support_count"], row["descriptor_count"],
                     _format(row["variance"]), _format(row["weight"]))
    _LOGGER.info("")
    return summary


def log_stage2_checkpoint(path, loaded=False):
    """Record whether Stage 2 trained a translator or reused a checkpoint."""
    _LOGGER.info("[Stage 2] Translator checkpoint %s: %s",
                 "loaded (training skipped)" if loaded else "saved", path)
    _LOGGER.info("")


def _json_value(value):
    """Convert experiment values to portable JSON, using null for NaN/Inf."""
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, (np.ndarray, np.generic)):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_run_report(args, result, baseline, bank, support_indices, stage2_path):
    """Save reproducible run metadata beside the readable training log."""
    path = Path(args.log_path) / args.dataset / args.model / f"{args.result_file}.json"
    report = _json_value({
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": vars(args),
        "baseline": baseline,
        "result": result,
        "target_bank": _target_bank_summary(bank),
        "support_indices": support_indices,
        "stage2_checkpoint": stage2_path,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    _LOGGER.info("Run report saved: %s", path)
    return path


class Stage2Metrics:
    """Accumulate detached diagnostics without changing training gradients."""

    def __init__(self):
        self.sums = {}
        self.count = 0
        self.batches = 0
        self.gradient_metrics = {}

    def measure_gradient_conflict(self, align, contrast, translator, weight):
        """Probe only the first valid batch per epoch, without touching .grad."""
        if self.gradient_metrics:
            return
        parameters = tuple(p for p in translator.parameters() if p.requires_grad)

        def gradient(loss):
            if not loss.requires_grad:
                return torch.cat([torch.zeros_like(p).flatten() for p in parameters])
            parts = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten()
                              for p, g in zip(parameters, parts)])

        ga = gradient(align)
        gc = gradient(contrast)
        na, nc = ga.norm(), gc.norm()
        denominator = na * nc
        combined_scale = na + abs(weight) * nc
        self.gradient_metrics = {
            "align_grad_norm": float(na),
            "contrast_grad_norm_unweighted": float(nc),
            "contrast_grad_norm_weighted": float(abs(weight) * nc),
            "gradient_cosine": float(torch.dot(ga, gc) / denominator)
                if float(denominator) > 0 else None,
            "gradient_cancellation_ratio": float((ga + weight * gc).norm() / combined_scale)
                if float(combined_scale) > 0 else None,
        }

    @torch.no_grad()
    def update(self, translated, prototypes, labels, translator, alpha, **losses):
        distances = torch.cdist(translated.detach(), prototypes.detach())
        rows = torch.arange(len(labels), device=distances.device)
        positive = distances[rows, labels]
        negative = distances.clone()
        negative[rows, labels] = float("inf")
        negatives = negative[torch.isfinite(negative)]
        has_negative = prototypes.shape[0] > 1
        hard = negative.min(dim=1).values
        grad_sq = sum(float(p.grad.detach().square().sum())
                      for p in translator.parameters() if p.grad is not None)
        values = {key: float(value.detach()) for key, value in losses.items()}
        values.update(
            pos_dist=float(positive.mean()),
            neg_dist=float(negatives.mean()) if has_negative else float("nan"),
            hard_neg_dist=float(hard.mean()) if has_negative else float("nan"),
            margin=float((hard - positive).mean()) if has_negative else float("nan"),
            active_ratio=float((negatives < alpha).float().mean()) if has_negative else float("nan"),
            nearest_acc=float((distances.argmin(1) == labels).float().mean()),
            grad_norm=grad_sq ** 0.5,
        )
        n = len(labels)
        self.count += n
        self.batches += 1
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + value * n

    def log_epoch(self, epoch, epochs):
        values = {key: value / self.count for key, value in self.sums.items()}
        accuracy = values.pop("nearest_acc", float("nan"))
        values.setdefault("loss", float("nan"))
        values.update(samples=self.count, batches=self.batches)
        log_epoch(2, epoch, epochs, values, accuracy,
                  acc_scope="translated source training / nearest prototype")
        if self.gradient_metrics:
            _LOGGER.info("  Gradient conflict probe: first valid batch only (not epoch average)")
            _LOGGER.info("  cosine < 0: opposing gradients; cancellation ratio near 0: strong cancellation")
            for name, value in self.gradient_metrics.items():
                _LOGGER.info("  %-38s : %s", name, _format(value))
            _LOGGER.info("")


@torch.no_grad()
def collect_model_embeddings(model, data_loader, device):
    """Collect the model's second output and labels without changing state."""
    model.eval()
    features, labels = [], []
    for inputs, batch_labels in data_loader:
        output = model(inputs.to(device))
        embedding = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else (output[0] if isinstance(output, (tuple, list)) else output)
        features.append(embedding.detach().cpu())
        labels.append(batch_labels.detach().cpu())
    if not features:
        return torch.empty((0, 0)), torch.empty((0,), dtype=torch.long)
    return torch.cat(features), torch.cat(labels).long()


def log_prototype_diagnostics(metrics: Mapping[str, Any], prefix: str = "") -> None:
    """Write aggregate geometry and an aligned per-class table to the file."""
    _LOGGER.info("%s\n[%s] Target prototype diagnostics (cosine)\n%s",
                 "-" * 80, prefix, "-" * 80)
    for name, value in metrics.items():
        if not isinstance(value, Mapping):
            _LOGGER.info("  %-44s : %s", name, _format(value))
    _LOGGER.info("\n  %-10s %20s %20s", "Class", "Support distance", "Evaluation distance")
    support = metrics["support_to_prototype_distance"]
    evaluation = metrics["evaluation_to_prototype_distance"]
    for label in support:
        _LOGGER.info("  %-10s %20s %20s", label, _format(support[label]),
                     _format(evaluation.get(label)))
    _LOGGER.info("")


def log_target_diagnostics(model, support_loader, eval_loader, prototypes, device, prefix):
    # Diagnostic loader iterations must not change subsequent shuffling/dropout.
    was_training = model.training
    try:
        with torch.random.fork_rng(devices=[]):
            support, support_labels = collect_model_embeddings(model, support_loader, device)
            evaluation, evaluation_labels = collect_model_embeddings(model, eval_loader, device)
    finally:
        model.train(was_training)
    metrics = compute_prototype_diagnostics(
        support, support_labels, evaluation, evaluation_labels, prototypes.detach().cpu(),
    )
    log_prototype_diagnostics(metrics, prefix)


@torch.no_grad()
def compute_prototype_geometry(prototypes, source_labels, alpha=1.0, contrast_weight=1.0):
    """Evaluate the actual fixed prototypes at perfect alignment z_i = p_y.

    Source-frequency weighting matches the source training loss expectation.
    This reference is not a lower bound on the combined optimization problem.
    """
    prototypes = prototypes.detach()
    classes = len(prototypes)
    labels = source_labels.detach().to(prototypes.device).long().flatten()
    labels = labels[(labels >= 0) & (labels < classes)]
    counts = torch.bincount(labels, minlength=classes).to(prototypes.dtype)
    distances = torch.cdist(prototypes, prototypes)
    mask = ~torch.eye(classes, dtype=torch.bool, device=prototypes.device)
    pairs = distances[mask]
    result = {
        "classes": classes, "alpha": alpha, "lambda_contrast": contrast_weight,
        "prototype_norm_min": float(prototypes.norm(dim=1).min()),
        "prototype_norm_max": float(prototypes.norm(dim=1).max()),
        "valid_source_samples": len(labels),
    }
    if pairs.numel():
        for name, q in (("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1)):
            result[f"prototype_distance_{name}"] = float(torch.quantile(pairs, q))
        result["prototype_distance_mean"] = float(pairs.mean())
        result["prototype_pair_fraction_below_alpha"] = float((pairs < alpha).float().mean())
        per_class = (F.relu(alpha - distances) * mask).sum(1) / (classes - 1)
        closest = distances.masked_fill(~mask, float("inf")).min(1).values
        result["nearest_prototype_distance_mean"] = float(closest.mean())
        result["ideal_alignment_contrast_class_balanced"] = float(per_class.mean())
        weighted = float((per_class * counts).sum() / counts.sum()) if len(labels) else None
    else:
        per_class = torch.zeros(classes, device=prototypes.device)
        closest = torch.full((classes,), float("nan"), device=prototypes.device)
        weighted = 0.0 if len(labels) else None
        result["prototype_pair_fraction_below_alpha"] = None
        result["ideal_alignment_contrast_class_balanced"] = 0.0
    result["ideal_alignment_align"] = 0.0
    result["ideal_alignment_contrast_source_weighted"] = weighted
    result["ideal_alignment_total_source_weighted"] = (
        contrast_weight * weighted if weighted is not None else None
    )
    result["per_class"] = {
        str(i): {"source_count": int(counts[i]), "nearest_distance": float(closest[i]),
                 "ideal_contrast": float(per_class[i])}
        for i in range(classes)
    }
    return result


def log_prototype_geometry(prototypes, source_labels, alpha=1.0, contrast_weight=1.0):
    metrics = compute_prototype_geometry(prototypes, source_labels, alpha, contrast_weight)
    _LOGGER.info("%s\n[Before Stage 2] Fixed prototype geometry (Euclidean)\n%s", "-" * 80, "-" * 80)
    _LOGGER.info("  Reference: translated output equals its true-class prototype (align = 0).")
    _LOGGER.info("  Positive reference contrast means perfect alignment still violates repulsion.")
    _LOGGER.info("  This is NOT a lower bound on total loss; moving away can reduce total loss.")
    for name, value in metrics.items():
        if name != "per_class":
            _LOGGER.info("  %-46s : %s", name, _format(value))
    _LOGGER.info("\n  %-8s %14s %20s %20s", "Class", "Source count", "Nearest prototype", "Ideal contrast")
    for label, row in metrics["per_class"].items():
        _LOGGER.info("  %-8s %14d %20s %20s", label, row["source_count"],
                     _format(row["nearest_distance"]), _format(row["ideal_contrast"]))
    _LOGGER.info("")
    return metrics


def log_trainable_parameters(target_encoder, translator):
    _LOGGER.info("[Stage 3] Parameter trainability for Stage 3")
    for name, module in (("target_encoder", target_encoder), ("translator", translator)):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        _LOGGER.info("  %-20s : trainable=%d / total=%d", name, trainable, total)
    _LOGGER.info("")
