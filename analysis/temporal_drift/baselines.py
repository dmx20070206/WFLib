#!/usr/bin/env python3
"""Leakage-controlled classical baselines for the Tor temporal-drift analysis.

Only Day 14 trains source models. Target splits are made WITHOUT stratification;
unlabelled adaptation features never include test rows. Target labels may be used
only by evaluation and by a benchmark support sampler with a 1/5-label per-class
budget. All hyperparameters below are fixed before viewing target results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import sklearn
from scipy.spatial.distance import cdist
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from extract_features import CORE, RATIOS, transform

DAYS = (14, 30, 90, 150, 270)
SEEDS = (11, 29, 47)
TREE_COUNT = 250
PRIOR_SAMPLE_SIZE = 5.0
COVARIANCE_SHRINKAGE = 0.1


def load_features(path: Path):
    frame = pd.read_csv(path)
    x = frame[CORE].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy()
    names = list(CORE)
    row_ids = frame["row_id"].to_numpy(dtype=np.int64)
    if len(x) != len(y) or len(row_ids) != len(y):
        raise ValueError(f"Inconsistent row counts: {path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite features: {path}")
    return x, y, names, row_ids


def transform_features(x: np.ndarray, names: list[str]):
    """Source-defined schema transformation, with no learned target statistics.

    Explicit ratios retain native scale; time measurements are converted from
    seconds to milliseconds (variance to milliseconds squared), then log1p.
    Count measurements use log1p. This imports the shared extraction schema.
    A negative measurement is rejected rather than silently reinterpreted.
    """
    logged = np.array([name not in RATIOS for name in names])
    if (x[:, logged] < 0).any():
        bad = np.array(names)[logged][np.any(x[:, logged] < 0, axis=0)]
        raise ValueError(f"Negative feature(s) need an explicit schema: {bad}")
    return transform(pd.DataFrame(x, columns=names), names=names), logged


def regularized_covariance(x: np.ndarray):
    cov = np.cov(x, rowvar=False)
    scale = max(float(np.trace(cov)) / cov.shape[0], 1e-8)
    return ((1 - COVARIANCE_SHRINKAGE) * cov
            + COVARIANCE_SHRINKAGE * scale * np.eye(cov.shape[0]))


def matrix_power_spd(matrix: np.ndarray, power: float):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 1e-10) ** power) @ vectors.T


def fit_prototypes(x: np.ndarray, y: np.ndarray, classes: np.ndarray):
    prototypes = np.stack([x[y == label].mean(axis=0) for label in classes])
    positions = np.searchsorted(classes, y)
    residuals = x - prototypes[positions]
    metric = matrix_power_spd(regularized_covariance(residuals), -0.5)
    return prototypes, metric


def prototype_predict(x, prototypes, metric, classes):
    distances = cdist(x @ metric, prototypes @ metric, "sqeuclidean")
    return classes[np.argmin(distances, axis=1)]


def evaluate(day, seed, method, y, prediction, adaptation_n, support_n=0):
    return {
        "day": int(day), "seed": int(seed), "method": method,
        "test_n": int(len(y)), "adaptation_n": int(adaptation_n),
        "support_n": int(support_n),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
    }


def hash_indices(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def support_sample(adaptation_idx, target_y, classes, seed, day):
    """Benchmark sampler: only returned support labels are given to adaptation.

    The dataset labels are consulted to simulate a known website's k traces;
    this is not a deployable strategy for finding classes in anonymous traffic.
    k=1 is a subset of k=5, supporting paired label-budget comparisons.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, day, 307]))
    samples = []
    for label in classes:
        candidates = adaptation_idx[target_y[adaptation_idx] == label]
        if len(candidates) < 5:
            raise ValueError(f"Day {day}, seed {seed}, class {label}: fewer than 5 support candidates")
        samples.append(rng.choice(candidates, size=5, replace=False))
    return np.stack(samples)


def aggregate(records):
    summaries = []
    for day in DAYS:
        methods = sorted({r["method"] for r in records if r["day"] == day})
        for method in methods:
            group = [r for r in records if r["day"] == day and r["method"] == method]
            row = {"day": day, "method": method, "seeds": len(group)}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                values = np.array([r[metric] for r in group])
                row[metric + "_mean"] = float(values.mean())
                row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row["test_n_min"] = min(r["test_n"] for r in group)
            row["test_n_max"] = max(r["test_n"] for r in group)
            row["support_n"] = group[0]["support_n"]
            summaries.append(row)
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(results_dir: Path):
    started = time.monotonic()
    for day in DAYS:
        audit = json.loads((results_dir / f"audit_day{day}.json").read_text())
        if audit["valid_feature_rows"] != audit["shape"][0]:
            raise ValueError(f"Day {day} extraction incomplete or trace-rejecting; expected all original rows")
    source_raw, source_y, names, source_rows = load_features(results_dir / "features_day14.csv")
    source_all, logged = transform_features(source_raw, names)
    classes = np.unique(source_y)
    records, comparisons, split_manifest = [], [], []
    predictions_to_save = {}
    for seed in SEEDS:
        train_idx, source_test_idx = train_test_split(
            np.arange(len(source_y)), test_size=0.3, random_state=seed, stratify=source_y
        )
        scaler = StandardScaler().fit(source_all[train_idx])
        source_train = scaler.transform(source_all[train_idx])
        source_test = scaler.transform(source_all[source_test_idx])
        tree = ExtraTreesClassifier(
            n_estimators=TREE_COUNT, max_features="sqrt", min_samples_leaf=1,
            random_state=seed, n_jobs=4,
        ).fit(source_train, source_y[train_idx])
        prototypes, metric = fit_prototypes(source_train, source_y[train_idx], classes)
        source_mean = source_train.mean(axis=0)
        source_cov_half = matrix_power_spd(regularized_covariance(source_train), 0.5)
        for method, prediction in (
            ("source_extratrees", tree.predict(source_test)),
            ("source_prototype", prototype_predict(source_test, prototypes, metric, classes)),
        ):
            records.append(evaluate(14, seed, method, source_y[source_test_idx], prediction, 0))
        split_manifest.append({
            "day": 14, "seed": seed, "train_n": len(train_idx), "test_n": len(source_test_idx),
            "train_row_sha256": hash_indices(source_rows[train_idx]),
            "test_row_sha256": hash_indices(source_rows[source_test_idx]),
        })
        predictions_to_save[f"day14_seed{seed}_train_row_id"] = source_rows[train_idx]
        predictions_to_save[f"day14_seed{seed}_test_row_id"] = source_rows[source_test_idx]
        for day in DAYS[1:]:
            raw, target_y, target_names, target_rows = load_features(results_dir / f"features_day{day}.csv")
            if names != target_names:
                raise ValueError(f"Feature schema mismatch on day {day}")
            if not np.array_equal(np.unique(target_y), classes):
                raise ValueError(f"Class set mismatch on day {day}")
            target_all, target_logged = transform_features(raw, names)
            assert np.array_equal(logged, target_logged)
            rng = np.random.default_rng(np.random.SeedSequence([seed, day, 101]))
            permutation = rng.permutation(len(target_all))  # Label-independent UDA split.
            midpoint = len(permutation) // 2
            adapt_idx, test_idx = permutation[:midpoint], permutation[midpoint:]
            assert len(np.intersect1d(adapt_idx, test_idx)) == 0
            target_adapt = scaler.transform(target_all[adapt_idx])
            target_test = scaler.transform(target_all[test_idx])
            mean_adapt = target_adapt.mean(axis=0)
            aligned_test = target_test - mean_adapt + source_mean
            coral_map = matrix_power_spd(regularized_covariance(target_adapt), -0.5) @ source_cov_half
            coral_test = (target_test - mean_adapt) @ coral_map + source_mean
            predictions = {
                "source_extratrees": tree.predict(target_test),
                "mean_uda_extratrees": tree.predict(aligned_test),
                "coral_uda_extratrees": tree.predict(coral_test),
                "source_prototype": prototype_predict(target_test, prototypes, metric, classes),
                "mean_uda_prototype": prototype_predict(aligned_test, prototypes, metric, classes),
                "coral_uda_prototype": prototype_predict(coral_test, prototypes, metric, classes),
            }
            sampled = support_sample(adapt_idx, target_y, classes, seed, day)
            for shots in (1, 5):
                support_idx = sampled[:, :shots].reshape(-1)
                assert len(np.intersect1d(support_idx, test_idx)) == 0
                support_x = scaler.transform(target_all[support_idx]) - mean_adapt + source_mean
                support_y = target_y[support_idx]
                support_prototypes = np.stack([support_x[support_y == label].mean(axis=0) for label in classes])
                weight = shots / (shots + PRIOR_SAMPLE_SIZE)
                new_prototypes = (1 - weight) * prototypes + weight * support_prototypes
                method = f"mean_uda_{shots}shot_prototype"
                predictions[method] = prototype_predict(aligned_test, new_prototypes, metric, classes)
                predictions_to_save[f"day{day}_seed{seed}_{shots}shot_support_row_id"] = target_rows[support_idx]
            for method, prediction in predictions.items():
                shots = 1 if "1shot" in method else (5 if "5shot" in method else 0)
                records.append(evaluate(day, seed, method, target_y[test_idx], prediction, len(adapt_idx), shots * len(classes)))
                predictions_to_save[f"day{day}_seed{seed}_{method}"] = prediction
            for base, candidate in (
                ("source_extratrees", "mean_uda_extratrees"),
                ("source_extratrees", "coral_uda_extratrees"),
                ("source_prototype", "mean_uda_prototype"),
                ("mean_uda_prototype", "mean_uda_1shot_prototype"),
                ("mean_uda_prototype", "mean_uda_5shot_prototype"),
            ):
                correct_base = predictions[base] == target_y[test_idx]
                correct_candidate = predictions[candidate] == target_y[test_idx]
                gained = int(np.sum(~correct_base & correct_candidate))
                lost = int(np.sum(correct_base & ~correct_candidate))
                pvalue = float(binomtest(gained, gained + lost, 0.5).pvalue) if gained + lost else 1.0
                comparisons.append({
                    "day": day, "seed": seed, "reference": base, "candidate": candidate,
                    "gained_correct": gained, "lost_correct": lost,
                    "accuracy_delta": float(correct_candidate.mean() - correct_base.mean()),
                    "mcnemar_exact_p_unadjusted": pvalue,
                })
            split_manifest.append({
                "day": day, "seed": seed, "adaptation_n": len(adapt_idx), "test_n": len(test_idx),
                "adaptation_row_sha256": hash_indices(target_rows[adapt_idx]),
                "test_row_sha256": hash_indices(target_rows[test_idx]),
                "support_5shot_row_sha256": hash_indices(target_rows[sampled.reshape(-1)]),
                "test_classes_present": int(len(np.unique(target_y[test_idx]))),
            })
            predictions_to_save[f"day{day}_seed{seed}_test_row_id"] = target_rows[test_idx]
            predictions_to_save[f"day{day}_seed{seed}_adapt_row_id"] = target_rows[adapt_idx]
            predictions_to_save[f"day{day}_seed{seed}_test_y"] = target_y[test_idx]
            print(f"seed={seed} day={day}: extra={records[-8]['accuracy']:.4f} "
                  f"mean_proto={predictions['mean_uda_prototype'].shape[0]} evaluated", flush=True)
    summaries = aggregate(records)
    pvalues = np.array([row["mcnemar_exact_p_unadjusted"] for row in comparisons])
    order = np.argsort(pvalues)
    adjusted = np.minimum(1.0, np.maximum.accumulate(pvalues[order] * (len(pvalues) - np.arange(len(pvalues)))))
    for position, value in zip(order, adjusted):
        comparisons[position]["mcnemar_exact_p_holm"] = float(value)
    write_csv(results_dir / "baselines_metrics.csv", records)
    write_csv(results_dir / "baselines_summary.csv", summaries)
    write_csv(results_dir / "baselines_paired_comparisons.csv", comparisons)
    np.savez_compressed(results_dir / "baselines_predictions_and_splits.npz", **predictions_to_save)
    manifest = {
        "purpose": "Diagnostic classical baselines; not the proposed deep drift-adaptation model.",
        "source": "Day 14 only; 70% stratified source training / 30% held-out source evaluation per seed.",
        "target_split": "Each target day independently split 50% adaptation / 50% test by label-independent permutation.",
        "target_test_policy": "Test features are used only for prediction; test labels only for final evaluation. No transductive normalization.",
        "feature_transform": "16 shared CORE engineered features from extract_features.py; time seconds -> milliseconds and variance -> milliseconds squared, log1p nonnegative non-ratio features; native ratios retained; StandardScaler fitted only to source train.",
        "log1p_features": np.array(names)[logged].tolist(),
        "native_features": np.array(names)[~logged].tolist(),
        "uda": "Estimate target mean/covariance using all adaptation features without labels; map target observations into source feature space.",
        "coral": "Target-to-source symmetric covariance whitening/recoloring; covariance shrinkage 0.1 toward trace(C)/p identity, means aligned.",
        "prototype": "Equal-prior class centroids; Mahalanobis distance from source pooled within-class covariance, with shrinkage 0.1.",
        "few_shot": "Simulate known-class collection from adaptation pool; exactly k labels/class; k=1 support nested within k=5; prototype=(5*source+k*support_mean)/(5+k) after global mean alignment.",
        "support_sampler_limitation": "Uses dataset class membership to simulate per-site data collection; it is not an anonymous-traffic label-acquisition algorithm. Only sampled labels affect model parameters.",
        "hyperparameter_policy": "All hyperparameters fixed in code before evaluation; no target test selection or tuning.",
        "independence_caveat": "No session/circuit identifiers are available for grouped splits; random row splits may overestimate deployment accuracy if traces are correlated. Source and target original row identities are saved, with no cross-day duplicate assumption.",
        "statistical_caveat": "Seed standard deviation describes split variation, not a confidence interval. Repeated seeds overlap; McNemar p-values assume independent test rows. Holm correction is applied over all paired comparisons, but cannot correct unobserved session dependence. Use only as exploratory evidence.",
        "seeds": list(SEEDS), "source_classes": len(classes), "feature_count": len(names),
        "trees": TREE_COUNT, "prior_sample_size": PRIOR_SAMPLE_SIZE,
        "covariance_shrinkage": COVARIANCE_SHRINKAGE,
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "elapsed_seconds": time.monotonic() - started, "splits": split_manifest,
    }
    (results_dir / "baselines_protocol.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    run(args.results_dir)
