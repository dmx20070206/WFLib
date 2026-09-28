#!/usr/bin/env python3
"""Evaluate global and class-conditional drift between two labelled NPZ files.

The main decomposition is performed in a robustly standardized feature space:

    class shift_k = mean(after_k) - mean(before_k)
    global/shared shift = mean_k(class shift_k)
    local residual_k = class shift_k - global/shared shift

Classes are equally weighted, so a change in class frequency is not mistaken for
a change in traffic shape.  Class-prior drift is reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


TRAFFIC_FEATURE_NAMES = [
    "log_packet_count",
    "log_duration",
    "positive_ratio",
    "direction_switch_rate",
    "log_positive_count",
    "log_negative_count",
    "early_100_positive_ratio",
    "early_500_positive_ratio",
    "log_burst_count",
    "log_mean_burst_length",
    "log_max_burst_length",
    "log_iat_mean",
    "log_iat_std",
    "log_iat_p50",
    "log_iat_p90",
    "log_iat_p99",
    "zero_iat_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "评估两个带标签 NPZ 数据集之间的全局漂移、逐类别局部漂移，"
            "以及类别间漂移方向/程度是否一致。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("before", type=Path, help="漂移前的 .npz 文件")
    parser.add_argument("after", type=Path, help="漂移后的 .npz 文件")
    parser.add_argument("--x-key", default="X", help="样本数组字段名")
    parser.add_argument("--y-key", default="y", help="标签数组字段名")
    parser.add_argument(
        "--representation",
        choices=("traffic", "flat"),
        default="traffic",
        help=(
            "traffic: 从带符号时间戳序列提取流量统计特征；"
            "flat: 将 X 展平并直接作为特征"
        ),
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="每个文件每类最多抽取的样本数；0 表示全部",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="类别内 bootstrap 次数；0 表示不计算置信区间",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="若指定，保存 summary.json、per_class.csv 和 feature_shifts.csv",
    )
    return parser.parse_args()


def python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def label_text(value: Any) -> str:
    value = python_scalar(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def validate_labels(y: np.ndarray, path: Path) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if y.dtype.kind == "f" and not np.all(np.isfinite(y)):
        raise ValueError(f"{path}: 标签包含 NaN/Inf")
    return y


def choose_indices(y: np.ndarray, classes: Sequence[Any], limit: int, rng: np.random.Generator) -> np.ndarray:
    chosen: List[np.ndarray] = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if limit > 0 and idx.size > limit:
            idx = np.sort(rng.choice(idx, size=limit, replace=False))
        chosen.append(idx)
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)


def traffic_features(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(
            f"traffic 表示要求 X 为二维 [样本, 序列]，实际形状为 {X.shape}；"
            "对普通特征张量请使用 --representation flat"
        )
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X 必须是数值数组")

    result = np.zeros((len(X), len(TRAFFIC_FEATURE_NAMES)), dtype=np.float64)
    for row_id, row in enumerate(X):
        valid = np.asarray(row[row != 0], dtype=np.float64)
        if valid.size == 0:
            continue
        if not np.all(np.isfinite(valid)):
            raise ValueError(f"第 {row_id} 个已抽样序列包含 NaN/Inf")

        direction = np.sign(valid)
        times = np.abs(valid)
        monotone_times = np.maximum.accumulate(times)
        iat = np.diff(monotone_times)
        switches = np.count_nonzero(direction[1:] != direction[:-1])
        burst_starts = np.r_[0, np.flatnonzero(direction[1:] != direction[:-1]) + 1]
        burst_ends = np.r_[burst_starts[1:], valid.size]
        burst_lengths = burst_ends - burst_starts

        def positive_ratio(prefix: int) -> float:
            part = direction[:prefix]
            return float(np.mean(part > 0)) if part.size else 0.0

        if iat.size:
            iat_mean = float(iat.mean())
            iat_std = float(iat.std())
            q50, q90, q99 = np.quantile(iat, [0.50, 0.90, 0.99])
            zero_iat_ratio = float(np.mean(iat == 0))
        else:
            iat_mean = iat_std = q50 = q90 = q99 = zero_iat_ratio = 0.0

        result[row_id] = [
            np.log1p(valid.size),
            np.log1p(max(float(monotone_times[-1] - monotone_times[0]), 0.0)),
            np.mean(direction > 0),
            switches / max(valid.size - 1, 1),
            np.log1p(np.count_nonzero(direction > 0)),
            np.log1p(np.count_nonzero(direction < 0)),
            positive_ratio(100),
            positive_ratio(500),
            np.log1p(burst_lengths.size),
            np.log1p(burst_lengths.mean()),
            np.log1p(burst_lengths.max()),
            np.log1p(iat_mean),
            np.log1p(iat_std),
            np.log1p(q50),
            np.log1p(q90),
            np.log1p(q99),
            zero_iat_ratio,
        ]
    return result


def make_features(X: np.ndarray, representation: str) -> Tuple[np.ndarray, List[str]]:
    if representation == "traffic":
        return traffic_features(X), list(TRAFFIC_FEATURE_NAMES)
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X 必须是数值数组")
    flat = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    if not np.all(np.isfinite(flat)):
        raise ValueError("抽样后的 X 包含 NaN/Inf")
    return flat, [f"x_{i}" for i in range(flat.shape[1])]


def load_sampled(
    path: Path,
    x_key: str,
    y_key: str,
    classes: Sequence[Any] | None,
    limit: int,
    representation: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in (x_key, y_key) if key not in archive.files]
        if missing:
            raise KeyError(f"{path}: 缺少字段 {missing}；现有字段为 {archive.files}")
        y_all = validate_labels(archive[y_key], path)
        if classes is None:
            classes = np.unique(y_all)
        indices = choose_indices(y_all, classes, limit, rng)
        X_all = archive[x_key]
        if len(X_all) != len(y_all):
            raise ValueError(f"{path}: X 有 {len(X_all)} 条、y 有 {len(y_all)} 条，长度不一致")
        X = X_all[indices]
        del X_all
    features, names = make_features(X, representation)
    return features, y_all[indices], y_all, names


def robust_standardize(before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pooled = np.vstack((before, after))
    center = np.median(pooled, axis=0)
    q25, q75 = np.quantile(pooled, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = pooled.std(axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (before - center) / scale, (after - center) / scale, center, scale


def class_arrays(features: np.ndarray, labels: np.ndarray, classes: Sequence[Any]) -> List[np.ndarray]:
    return [features[labels == cls] for cls in classes]


def drift_metrics(shifts: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    magnitudes = np.linalg.norm(shifts, axis=1)
    shared = shifts.mean(axis=0)
    shared_norm = float(np.linalg.norm(shared))
    residuals = shifts - shared
    residual_rms = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    total_rms = float(np.sqrt(np.mean(magnitudes * magnitudes)))
    local_fraction = residual_rms**2 / max(total_rms**2, eps)

    active = magnitudes > eps
    unit = shifts[active] / magnitudes[active, None]
    if len(unit) >= 2:
        resultant = float(np.linalg.norm(unit.mean(axis=0)))
        cosine_matrix = unit @ unit.T
        upper = cosine_matrix[np.triu_indices(len(unit), 1)]
        pairwise_mean = float(upper.mean())
        pairwise_negative = float(np.mean(upper < 0))
    elif len(unit) == 1:
        resultant, pairwise_mean, pairwise_negative = 1.0, 1.0, 0.0
    else:
        resultant = pairwise_mean = pairwise_negative = 0.0

    mean_mag = float(magnitudes.mean())
    median_mag = float(np.median(magnitudes))
    q25, q75 = np.quantile(magnitudes, [0.25, 0.75])
    return {
        "shared_shift_norm": shared_norm,
        "local_residual_rms": residual_rms,
        "total_class_shift_rms": total_rms,
        "shared_fraction": float(1.0 - local_fraction),
        "local_fraction": float(local_fraction),
        "direction_dispersion": float(1.0 - resultant),
        "pairwise_cosine_mean": pairwise_mean,
        "pairwise_opposite_fraction": pairwise_negative,
        "magnitude_mean": mean_mag,
        "magnitude_median": median_mag,
        "magnitude_cv": float(magnitudes.std() / max(mean_mag, eps)),
        "magnitude_robust_cv": float((q75 - q25) / max(median_mag, eps)),
        "magnitude_min": float(magnitudes.min()),
        "magnitude_max": float(magnitudes.max()),
    }


def bootstrap_intervals(
    before_by_class: Sequence[np.ndarray],
    after_by_class: Sequence[np.ndarray],
    repetitions: int,
    rng: np.random.Generator,
) -> Dict[str, List[float]]:
    if repetitions <= 0:
        return {}
    tracked = [
        "shared_shift_norm",
        "local_residual_rms",
        "shared_fraction",
        "local_fraction",
        "direction_dispersion",
        "pairwise_cosine_mean",
        "magnitude_cv",
        "magnitude_robust_cv",
    ]
    samples = {key: [] for key in tracked}
    for _ in range(repetitions):
        shifts = []
        for old, new in zip(before_by_class, after_by_class):
            old_mean = old[rng.integers(0, len(old), len(old))].mean(axis=0)
            new_mean = new[rng.integers(0, len(new), len(new))].mean(axis=0)
            shifts.append(new_mean - old_mean)
        metrics = drift_metrics(np.asarray(shifts))
        for key in tracked:
            samples[key].append(metrics[key])
    return {
        key: [float(x) for x in np.quantile(values, [0.025, 0.975])]
        for key, values in samples.items()
    }


def domain_auc(before_by_class: Sequence[np.ndarray], after_by_class: Sequence[np.ndarray], seed: int) -> Tuple[float, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise RuntimeError("计算 domain AUC 需要 scikit-learn") from exc

    old_parts, new_parts = [], []
    rng = np.random.default_rng(seed)
    for old, new in zip(before_by_class, after_by_class):
        n = min(len(old), len(new))
        old_parts.append(old[rng.choice(len(old), n, replace=False)])
        new_parts.append(new[rng.choice(len(new), n, replace=False)])
    old_equal = np.vstack(old_parts)
    new_equal = np.vstack(new_parts)
    X = np.vstack((old_equal, new_equal))
    y = np.r_[np.zeros(len(old_equal), dtype=int), np.ones(len(new_equal), dtype=int)]
    splits = min(5, int(np.bincount(y).min()))
    if splits < 2:
        return float("nan"), float("nan")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    aucs = []
    for train, test in cv.split(X, y):
        model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        model.fit(X[train], y[train])
        aucs.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0)


def prior_jsd(before_y: np.ndarray, after_y: np.ndarray, all_classes: Sequence[Any]) -> float:
    p = np.asarray([np.mean(before_y == cls) for cls in all_classes], dtype=float)
    q = np.asarray([np.mean(after_y == cls) for cls in all_classes], dtype=float)
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return (kl(p, m) + kl(q, m)) / 2


def top_feature_rows(
    feature_names: Sequence[str], shared: np.ndarray, residuals: np.ndarray
) -> List[Dict[str, Any]]:
    rows = []
    local_std = residuals.std(axis=0)
    for name, common, heterogeneous in zip(feature_names, shared, local_std):
        rows.append(
            {
                "feature": name,
                "shared_shift": float(common),
                "class_shift_std": float(heterogeneous),
                "absolute_shared_shift": float(abs(common)),
            }
        )
    return sorted(rows, key=lambda row: row["absolute_shared_shift"] + row["class_shift_std"], reverse=True)


def ci_text(value: float, intervals: Dict[str, List[float]], key: str) -> str:
    if key not in intervals:
        return f"{value:.4f}"
    low, high = intervals[key]
    return f"{value:.4f} (95% CI {low:.4f}–{high:.4f})"


def write_outputs(
    output_dir: Path,
    summary: Dict[str, Any],
    per_class: Sequence[Dict[str, Any]],
    feature_rows: Sequence[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    for filename, rows in (("per_class.csv", per_class), ("feature_shifts.csv", feature_rows)):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_per_class < 0 or args.bootstrap < 0:
        raise ValueError("--max-per-class 和 --bootstrap 不能为负数")

    rng = np.random.default_rng(args.seed)
    # Load labels first.  This determines the shared class set without retaining X.
    with np.load(args.before, allow_pickle=False) as old_archive, np.load(args.after, allow_pickle=False) as new_archive:
        if args.y_key not in old_archive.files or args.y_key not in new_archive.files:
            raise KeyError(f"两个文件都必须包含标签字段 {args.y_key!r}")
        old_y_all_probe = validate_labels(old_archive[args.y_key], args.before)
        new_y_all_probe = validate_labels(new_archive[args.y_key], args.after)
        old_classes = np.unique(old_y_all_probe)
        new_classes = np.unique(new_y_all_probe)
        classes = np.intersect1d(old_classes, new_classes)
    if len(classes) < 2:
        raise ValueError(f"两个数据集共同类别仅 {len(classes)} 个，至少需要 2 个")

    before, before_y, before_y_all, feature_names = load_sampled(
        args.before, args.x_key, args.y_key, classes, args.max_per_class,
        args.representation, rng,
    )
    after, after_y, after_y_all, after_feature_names = load_sampled(
        args.after, args.x_key, args.y_key, classes, args.max_per_class,
        args.representation, rng,
    )
    if before.shape[1] != after.shape[1]:
        raise ValueError(f"两边特征维数不同：{before.shape[1]} vs {after.shape[1]}")
    if feature_names != after_feature_names:
        raise ValueError("两边生成的特征定义不一致")

    before_z, after_z, center, scale = robust_standardize(before, after)
    before_by_class = class_arrays(before_z, before_y, classes)
    after_by_class = class_arrays(after_z, after_y, classes)
    old_means = np.vstack([values.mean(axis=0) for values in before_by_class])
    new_means = np.vstack([values.mean(axis=0) for values in after_by_class])
    shifts = new_means - old_means
    metrics = drift_metrics(shifts)
    shared = shifts.mean(axis=0)
    residuals = shifts - shared
    intervals = bootstrap_intervals(before_by_class, after_by_class, args.bootstrap, rng)
    auc_mean, auc_std = domain_auc(before_by_class, after_by_class, args.seed)

    magnitudes = np.linalg.norm(shifts, axis=1)
    shared_norm = np.linalg.norm(shared)
    per_class: List[Dict[str, Any]] = []
    for i, cls in enumerate(classes):
        cosine = float(np.dot(shifts[i], shared) / (max(magnitudes[i], 1e-12) * max(shared_norm, 1e-12)))
        per_class.append(
            {
                "class": label_text(cls),
                "before_n": int(np.sum(before_y == cls)),
                "after_n": int(np.sum(after_y == cls)),
                "shift_magnitude": float(magnitudes[i]),
                "cosine_to_shared": cosine,
                "local_residual_magnitude": float(np.linalg.norm(residuals[i])),
            }
        )
    per_class.sort(key=lambda row: row["shift_magnitude"], reverse=True)
    feature_rows = top_feature_rows(feature_names, shared, residuals)

    old_only = [label_text(x) for x in np.setdiff1d(old_classes, new_classes)]
    new_only = [label_text(x) for x in np.setdiff1d(new_classes, old_classes)]
    all_classes = np.union1d(old_classes, new_classes)
    prior_drift = prior_jsd(before_y_all, after_y_all, all_classes)

    # These are transparent descriptive thresholds, not universal hypothesis tests.
    global_detectable = bool(auc_mean >= 0.60)
    direction_inconsistent = bool(metrics["direction_dispersion"] >= 0.25)
    magnitude_inconsistent = bool(metrics["magnitude_robust_cv"] >= 0.30)
    local_substantial = bool(metrics["local_fraction"] >= 0.25)
    both_global_local = bool(global_detectable and local_substantial)

    summary: Dict[str, Any] = {
        "inputs": {
            "before": str(args.before.resolve()),
            "after": str(args.after.resolve()),
            "representation": args.representation,
            "x_key": args.x_key,
            "y_key": args.y_key,
            "max_per_class": args.max_per_class,
            "seed": args.seed,
        },
        "data": {
            "before_total": int(len(before_y_all)),
            "after_total": int(len(after_y_all)),
            "before_used": int(len(before_y)),
            "after_used": int(len(after_y)),
            "common_class_count": int(len(classes)),
            "before_only_classes": old_only,
            "after_only_classes": new_only,
            "feature_count": int(before.shape[1]),
        },
        "global_drift": {
            "domain_auc_mean": auc_mean,
            "domain_auc_fold_std": auc_std,
            "class_prior_jsd_bits": prior_drift,
            "detectable_auc_ge_0_60": global_detectable,
        },
        "decomposition": metrics,
        "bootstrap_95_ci": intervals,
        "assessment": {
            "global_and_local_drift": both_global_local,
            "local_component_substantial": local_substantial,
            "class_directions_inconsistent": direction_inconsistent,
            "class_magnitudes_inconsistent": magnitude_inconsistent,
            "thresholds": {
                "domain_auc": 0.60,
                "local_fraction": 0.25,
                "direction_dispersion": 0.25,
                "magnitude_robust_cv": 0.30,
            },
            "note": "结论采用描述性阈值，需结合 bootstrap CI、逐类结果和业务尺度解释。",
        },
        "top_feature_shifts": feature_rows[:10],
    }

    print("\n=== 数据 ===")
    print(f"共同类别: {len(classes)}；使用样本: before={len(before_y)}, after={len(after_y)}")
    if old_only or new_only:
        print(f"警告：仅 before 有 {old_only}；仅 after 有 {new_only}（不进入条件漂移分解）")
    print(f"类别先验 JSD: {prior_drift:.4f} bits（0 表示类别比例相同）")

    print("\n=== 全局漂移 ===")
    print(f"类别等权 domain-classifier AUC: {auc_mean:.4f} ± {auc_std:.4f}（0.5≈不可区分）")
    print(f"共同位移范数: {ci_text(metrics['shared_shift_norm'], intervals, 'shared_shift_norm')}")

    print("\n=== 全局 + 局部分解 ===")
    print(f"共同成分占比: {ci_text(metrics['shared_fraction'], intervals, 'shared_fraction')}")
    print(f"局部残差占比: {ci_text(metrics['local_fraction'], intervals, 'local_fraction')}")
    print(f"方向离散度:   {ci_text(metrics['direction_dispersion'], intervals, 'direction_dispersion')}（0=完全同向）")
    print(f"两类方向相反比例: {metrics['pairwise_opposite_fraction']:.4f}")
    print(f"幅度 CV:      {ci_text(metrics['magnitude_cv'], intervals, 'magnitude_cv')}")
    print(f"幅度 robust-CV(IQR/median): {ci_text(metrics['magnitude_robust_cv'], intervals, 'magnitude_robust_cv')}")

    print("\n=== 描述性判断 ===")
    print(f"可检测全局漂移: {'是' if global_detectable else '否'}")
    print(f"同时体现全局 + 局部漂移: {'是' if both_global_local else '否'}")
    print(f"类别漂移方向不一致: {'是' if direction_inconsistent else '否'}")
    print(f"类别漂移程度不一致: {'是' if magnitude_inconsistent else '否'}")
    print("漂移幅度最大的类别: " + ", ".join(
        f"{row['class']}({row['shift_magnitude']:.3f})" for row in per_class[:10]
    ))

    if args.output_dir is not None:
        write_outputs(args.output_dir, summary, per_class, feature_rows)
        print(f"\n结果已保存到: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
