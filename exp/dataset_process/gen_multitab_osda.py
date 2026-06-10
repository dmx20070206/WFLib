from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

# 为了便于后续修改，这里显式保留时间戳列索引与目标截断长度。
TIMESTAMP_COL = 0
MAX_LEN = 5000
OVERLAP_RATIOS = [0.2, 0.4, 0.6, 0.8]
FIX_SEED = 2024

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWN_DIR = REPO_ROOT / "datasets" / "TemporalDrift"
DEFAULT_UNKNOWN_DIR = REPO_ROOT / "datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "MultiTab"
DEFAULT_KNOWN_FILES = [
    "train.npz",
    "valid.npz",
    "day14.npz",
    "day30.npz",
    "day90.npz",
    "day150.npz",
    "day270.npz",
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate Multi-Tab OSDA datasets")
    parser.add_argument(
        "--known-dir",
        type=Path,
        default=DEFAULT_KNOWN_DIR,
        help="已知类单标签页数据目录，默认是 datasets/TemporalDrift",
    )
    parser.add_argument(
        "--unknown-dir",
        type=Path,
        default=DEFAULT_UNKNOWN_DIR,
        help="未知类背景流量目录，默认是 datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录，默认是 datasets/MultiTab",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_KNOWN_FILES,
        help="需要处理的已知类文件名列表，默认处理 TemporalDrift 全部目标文件",
    )
    parser.add_argument(
        "--overlap-ratios",
        nargs="*",
        type=float,
        default=OVERLAP_RATIOS,
        help="需要生成的重叠率列表，例如 --overlap-ratios 0.2 0.4",
    )
    parser.add_argument(
        "--limit-known",
        type=int,
        default=None,
        help="仅生成前若干个 Known 样本，主要用于快速验证；默认使用全部已知样本",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIX_SEED,
        help="随机种子，默认 2024",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在，则覆盖重建",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，确保结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)


def load_npz_dataset(file_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    """加载 npz，并兼容 X/y 与 data/labels 两种键名。

    返回值中的第三项是原始键名，便于输出时尽量保持与源数据一致。
    """
    with np.load(file_path) as data:
        if "data" in data and "labels" in data:
            features = data["data"]
            labels = data["labels"]
            key_names = ("data", "labels")
        elif "X" in data and "y" in data:
            features = data["X"]
            labels = data["y"]
            key_names = ("X", "y")
        else:
            raise KeyError(
                f"{file_path} 缺少可识别的键名，期望 (data, labels) 或 (X, y)，实际为 {data.files}"
            )

    return features, labels, key_names


def infer_num_known_samples(total_known: int, limit_known: int | None) -> int:
    """确定当前文件需要生成多少个 Known 样本。

    默认情况下，M 直接取当前已知类源文件中的样本总数；
    若指定 --limit-known，则仅生成前 limit_known 个，便于快速验证脚本流程。
    """
    if limit_known is None:
        return total_known
    if limit_known <= 0:
        raise ValueError("--limit-known 必须是正整数")
    return min(total_known, limit_known)


def trim_padding(trace: np.ndarray) -> np.ndarray:
    """移除尾部 padding，仅保留真实流量。

    三维场景下的单条样本形状应为 [L, F]；
    当前仓库中的二维数据在单条样本层面是 [L]，这里也一并兼容。
    """
    if trace.ndim == 1:
        valid_mask = trace != 0
    elif trace.ndim == 2:
        valid_mask = np.any(trace != 0, axis=1)
    else:
        raise ValueError(f"不支持的 trace 维度: {trace.ndim}")

    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return trace[:0].copy()
    last_valid_index = valid_indices[-1] + 1
    return trace[:last_valid_index].copy()


def get_timestamps(trace: np.ndarray) -> np.ndarray:
    """提取单条 trace 的时间戳序列。

    对于当前仓库常见的一维表示，绝对值表示时间戳，符号保留方向信息。
    对于多特征表示，时间戳位于 TIMESTAMP_COL 列。
    """
    if trace.ndim == 1:
        return np.abs(trace)
    return trace[:, TIMESTAMP_COL]


def apply_time_offset(trace: np.ndarray, delta_t: float) -> np.ndarray:
    """给 trace 的时间戳整体施加偏移量。"""
    shifted = trace.copy()
    if shifted.ndim == 1:
        directions = np.sign(shifted)
        shifted = directions * (np.abs(shifted) + delta_t)
    else:
        shifted[:, TIMESTAMP_COL] += delta_t
    return shifted


def sort_trace_by_timestamp(trace: np.ndarray) -> np.ndarray:
    """按照时间戳对拼接后的流量进行全局排序，实现拉链式合并。"""
    sort_index = np.argsort(get_timestamps(trace), kind="stable")
    return trace[sort_index]


def pad_or_truncate(trace: np.ndarray, max_len: int) -> np.ndarray:
    """将合并后的样本截断或补零到固定长度。"""
    current_len = trace.shape[0]
    if current_len >= max_len:
        return trace[:max_len].copy()

    if trace.ndim == 1:
        padded = np.zeros((max_len,), dtype=trace.dtype)
        padded[:current_len] = trace
        return padded

    padded = np.zeros((max_len, trace.shape[1]), dtype=trace.dtype)
    padded[:current_len] = trace
    return padded


def to_time_direction_trace(trace: np.ndarray) -> np.ndarray:
    """将一维 signed-timestamp trace 转成 [timestamp, direction] 二维表示。

    当前仓库的一维数据用符号编码方向、用绝对值编码时间。
    在发生负偏移时，若继续直接在一维数值上平移，会把“负时间”误解释为“负方向”。
    因此先展开为二维表示，再做偏移与排序，最后再还原回一维格式。
    """
    converted = np.zeros((trace.shape[0], 2), dtype=trace.dtype)
    converted[:, 0] = np.abs(trace)
    converted[:, 1] = np.sign(trace)
    return converted


def from_time_direction_trace(trace: np.ndarray) -> np.ndarray:
    """将 [timestamp, direction] 还原为一维 signed-timestamp 表示。"""
    normalized = trace.copy()
    if normalized.shape[0] > 0:
        normalized[:, 0] -= normalized[0, 0]
    return normalized[:, 1] * normalized[:, 0]


def compute_duration(trace: np.ndarray) -> float:
    """计算 trace 的总耗时。"""
    timestamps = get_timestamps(trace)
    if timestamps.size <= 1:
        return 0.0
    return float(timestamps[-1] - timestamps[0])


def merge_traces(trace_a: np.ndarray, trace_b: np.ndarray, overlap_ratio: float) -> np.ndarray:
    """按照指定重叠率合并两条去 padding 后的真实流量。

    逻辑与用户要求保持一致：
    1. 计算 A、B 的持续时间 T_A、T_B
    2. 计算交集时间 T_overlap = max(T_A, T_B) * overlap_ratio
    3. 给 B 施加偏移量 delta_t = T_A - T_overlap
    4. 先拼接，再根据时间戳做全局排序
    5. 最后截断或补零到 MAX_LEN
    """
    if not 0 <= overlap_ratio <= 1:
        raise ValueError(f"overlap_ratio 必须位于 [0, 1]，当前为 {overlap_ratio}")

    if trace_a.ndim != trace_b.ndim:
        raise ValueError("trace_A 与 trace_B 的维度不一致，无法合并")

    if trace_a.ndim == 1:
        merged = merge_traces(
            to_time_direction_trace(trace_a),
            to_time_direction_trace(trace_b),
            overlap_ratio,
        )
        restored = from_time_direction_trace(merged)
        return pad_or_truncate(restored.astype(trace_a.dtype, copy=False), MAX_LEN)

    if trace_a.ndim == 2 and trace_a.shape[1] != trace_b.shape[1]:
        raise ValueError("trace_A 与 trace_B 的特征维数不一致，无法合并")

    if trace_a.shape[0] == 0:
        return pad_or_truncate(trace_b, MAX_LEN)
    if trace_b.shape[0] == 0:
        return pad_or_truncate(trace_a, MAX_LEN)

    duration_a = compute_duration(trace_a)
    duration_b = compute_duration(trace_b)
    overlap_time = max(duration_a, duration_b) * overlap_ratio
    delta_t = duration_a - overlap_time

    shifted_b = apply_time_offset(trace_b, delta_t)
    merged = np.concatenate([trace_a, shifted_b], axis=0)
    merged = sort_trace_by_timestamp(merged)
    return pad_or_truncate(merged, MAX_LEN)


def sample_index(rng: np.random.Generator, pool_size: int) -> int:
    """从样本池中随机采样一个索引。"""
    if pool_size <= 0:
        raise ValueError("样本池为空，无法采样")
    return int(rng.integers(0, pool_size))


def resolve_unknown_file(known_file_name: str) -> str:
    """根据已知类文件名，映射对应的未知类背景文件。"""
    if known_file_name == "train.npz":
        return "bg_train.npz"
    if known_file_name == "valid.npz":
        return "bg_valid.npz"
    return "bg_tune.npz"


def ensure_parent_dir(path: Path) -> None:
    """确保输出目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def shuffle_in_unison(data: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """对数据与标签执行一致的随机打乱。"""
    permutation = rng.permutation(data.shape[0])
    return data[permutation], labels[permutation]


def get_output_shape(sample_data: np.ndarray, target_count: int) -> tuple[int, ...]:
    """根据输入数据维度推断输出数组形状。"""
    if sample_data.ndim == 2:
        return (target_count, MAX_LEN)
    return (target_count, MAX_LEN, sample_data.shape[2])


def generate_known_samples(
    known_data: np.ndarray,
    known_labels: np.ndarray,
    unknown_data: np.ndarray,
    overlap_ratio: float,
    target_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 M 个 Known 样本。

    对于 trace_A，默认从当前已知文件中抽取；
    其中 50% 概率执行 Known + Known，50% 概率执行 Known + Unknown；
    输出标签始终保持 trace_A 的原始标签。
    """
    results_data = np.zeros(get_output_shape(known_data, target_count), dtype=known_data.dtype)
    results_labels = np.zeros((target_count,), dtype=known_labels.dtype)

    shuffled_known_indices = rng.permutation(known_data.shape[0])
    selected_known_indices = shuffled_known_indices[:target_count]

    for output_idx, known_idx in enumerate(
        tqdm(selected_known_indices, desc="生成 Known 样本", dynamic_ncols=True, leave=False)
    ):
        trace_a = trim_padding(known_data[known_idx])
        label_a = known_labels[known_idx]

        if rng.random() < 0.5:
            trace_b = trim_padding(known_data[sample_index(rng, known_data.shape[0])])
        else:
            trace_b = trim_padding(unknown_data[sample_index(rng, unknown_data.shape[0])])

        results_data[output_idx] = merge_traces(trace_a, trace_b, overlap_ratio)
        results_labels[output_idx] = label_a

    return results_data, results_labels


def generate_unknown_samples(
    unknown_data: np.ndarray,
    unknown_label: int | float,
    label_dtype: np.dtype,
    overlap_ratio: float,
    target_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 2M 个 Unknown 样本，标签统一设为当前文件中的最大类编号。"""
    results_data = np.zeros(get_output_shape(unknown_data, target_count), dtype=unknown_data.dtype)
    results_labels = np.full((target_count,), unknown_label, dtype=label_dtype)

    for output_idx in tqdm(range(target_count), desc="生成 Unknown 样本", dynamic_ncols=True, leave=False):
        trace_a = trim_padding(unknown_data[sample_index(rng, unknown_data.shape[0])])
        trace_b = trim_padding(unknown_data[sample_index(rng, unknown_data.shape[0])])
        results_data[output_idx] = merge_traces(trace_a, trace_b, overlap_ratio)

    return results_data, results_labels


def save_dataset(file_path: Path, data: np.ndarray, labels: np.ndarray, key_names: tuple[str, str]) -> None:
    """按输入数据的键名风格保存输出数据。"""
    feature_key, label_key = key_names
    np.savez(file_path, **{feature_key: data, label_key: labels})


def validate_feature_shapes(known_data: np.ndarray, unknown_data: np.ndarray) -> None:
    """校验已知类与未知类数据能否用于合并。"""
    if known_data.ndim != unknown_data.ndim:
        raise ValueError(
            f"已知类与未知类维度不一致：known={known_data.ndim}, unknown={unknown_data.ndim}"
        )

    if known_data.ndim == 3 and known_data.shape[2] != unknown_data.shape[2]:
        raise ValueError(
            f"已知类与未知类特征维数不一致：known={known_data.shape[2]}, unknown={unknown_data.shape[2]}"
        )

    if known_data.ndim not in (2, 3):
        raise ValueError(f"仅支持二维或三维输入，当前 known_data.ndim={known_data.ndim}")


def build_output_dataset(
    known_file: Path,
    unknown_file: Path,
    output_file: Path,
    overlap_ratio: float,
    limit_known: int | None,
    seed: int,
) -> None:
    """构建单个输出 npz 文件。"""
    known_data, known_labels, key_names = load_npz_dataset(known_file)
    unknown_data, _, _ = load_npz_dataset(unknown_file)

    validate_feature_shapes(known_data, unknown_data)

    num_known = infer_num_known_samples(known_data.shape[0], limit_known)
    num_unknown = num_known * 2
    label_dtype = np.result_type(known_labels.dtype, np.int64)
    unknown_label = int(np.max(known_labels)) + 1
    rng = np.random.default_rng(seed)

    known_samples, known_sample_labels = generate_known_samples(
        known_data=known_data,
        known_labels=known_labels,
        unknown_data=unknown_data,
        overlap_ratio=overlap_ratio,
        target_count=num_known,
        rng=rng,
    )
    unknown_samples, unknown_labels = generate_unknown_samples(
        unknown_data=unknown_data,
        unknown_label=unknown_label,
        label_dtype=label_dtype,
        overlap_ratio=overlap_ratio,
        target_count=num_unknown,
        rng=rng,
    )

    merged_data = np.concatenate([known_samples, unknown_samples], axis=0)
    merged_labels = np.concatenate(
        [known_sample_labels.astype(label_dtype, copy=False), unknown_labels.astype(label_dtype, copy=False)], axis=0
    )

    merged_data, merged_labels = shuffle_in_unison(merged_data, merged_labels, rng)

    ensure_parent_dir(output_file.parent)
    save_dataset(output_file, merged_data, merged_labels, key_names)

    print(
        f"[完成] {output_file} | overlap={overlap_ratio:.1f} | "
        f"known={num_known} | unknown={num_unknown} | data={merged_data.shape} | labels={merged_labels.shape}"
    )


def iter_existing_known_files(known_dir: Path, file_names: Iterable[str]) -> list[Path]:
    """返回实际存在的已知类文件列表，并对缺失文件给出提示。"""
    existing_files: list[Path] = []
    for file_name in file_names:
        candidate = known_dir / file_name
        if candidate.exists():
            existing_files.append(candidate)
        else:
            print(f"[跳过] 未找到已知类文件：{candidate}")
    return existing_files


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    set_seed(args.seed)

    known_files = iter_existing_known_files(args.known_dir, args.files)
    if not known_files:
        raise FileNotFoundError(f"在 {args.known_dir} 下未找到任何可处理的已知类文件")

    for overlap_ratio in args.overlap_ratios:
        overlap_folder = args.output_dir / f"overlap_{int(overlap_ratio * 100):02d}"
        ensure_parent_dir(overlap_folder)

        progress_desc = f"处理 overlap={overlap_ratio:.1f}"
        for known_file in tqdm(known_files, desc=progress_desc, dynamic_ncols=True):
            unknown_file_name = resolve_unknown_file(known_file.name)
            unknown_file = args.unknown_dir / unknown_file_name
            if not unknown_file.exists():
                raise FileNotFoundError(f"背景流量文件不存在：{unknown_file}")

            output_file = overlap_folder / known_file.name
            if output_file.exists() and not args.overwrite:
                print(f"[跳过] {output_file} 已存在，如需重建请添加 --overwrite")
                continue

            current_seed = args.seed + int(overlap_ratio * 1000) + sum(ord(ch) for ch in known_file.name)
            build_output_dataset(
                known_file=known_file,
                unknown_file=unknown_file,
                output_file=output_file,
                overlap_ratio=overlap_ratio,
                limit_known=args.limit_known,
                seed=current_seed,
            )


if __name__ == "__main__":
    main()