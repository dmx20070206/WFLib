from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

TIMESTAMP_COL = 0
MAX_LEN = 18000
FIX_SEED = 2024
OVERLAP_RATIO = 0.5
LIMIT_KNOWN = None
OVERWRITE_OUTPUT = False

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
    """确定当前文件需要生成多少个 Known+Unknown 样本。"""
    if limit_known is None:
        return total_known
    if limit_known <= 0:
        raise ValueError("LIMIT_KNOWN 必须是正整数")
    return min(total_known, limit_known)


def trim_padding(trace: np.ndarray) -> np.ndarray:
    """移除尾部 padding，仅保留真实流量。"""
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


def extract_timestamps(trace: np.ndarray) -> np.ndarray:
    """提取可排序的时间戳序列。"""
    return trace[:, TIMESTAMP_COL]


def stable_sort_trace(trace: np.ndarray) -> np.ndarray:
    """按时间戳稳定排序，保证同一时刻的包顺序尽量保持原样。"""
    if trace.shape[0] <= 1:
        return trace.copy()
    sort_index = np.argsort(extract_timestamps(trace), kind="stable")
    return trace[sort_index]


def normalize_trace_start(trace: np.ndarray) -> np.ndarray:
    """将单条 trace 的起始时间对齐到 0。"""
    normalized = trace.copy()
    if normalized.shape[0] == 0:
        return normalized
    normalized[:, TIMESTAMP_COL] -= normalized[0, TIMESTAMP_COL]
    return normalized


def to_time_direction_trace(trace: np.ndarray) -> np.ndarray:
    """将一维 signed-timestamp trace 展开为 [timestamp, direction] 表示。"""
    converted = np.zeros((trace.shape[0], 2), dtype=np.result_type(trace.dtype, np.float64))
    converted[:, 0] = np.abs(trace)
    converted[:, 1] = np.where(trace < 0, -1.0, 1.0)
    return converted


def from_time_direction_trace(trace: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """将 [timestamp, direction] 还原为一维 signed-timestamp 表示。"""
    normalized = normalize_trace_start(trace)
    directions = np.where(normalized[:, 1] < 0, -1.0, 1.0)
    restored = directions * normalized[:, 0]
    return restored.astype(target_dtype, copy=False)


def prepare_trace_for_merge(trace: np.ndarray) -> np.ndarray:
    """裁掉 padding、转到统一表示，并把起点时间归零。"""
    trimmed = trim_padding(trace)
    if trimmed.ndim == 1:
        merge_ready = to_time_direction_trace(trimmed)
    elif trimmed.ndim == 2:
        merge_ready = trimmed.astype(np.result_type(trimmed.dtype, np.float64), copy=True)
    else:
        raise ValueError(f"不支持的 trace 维度: {trimmed.ndim}")

    merge_ready = stable_sort_trace(merge_ready)
    return normalize_trace_start(merge_ready)


def shift_trace_in_time(trace: np.ndarray, delta_t: float) -> np.ndarray:
    """整体平移 trace 的时间轴。"""
    shifted = trace.copy()
    shifted[:, TIMESTAMP_COL] += delta_t
    return shifted


def compute_duration(trace: np.ndarray) -> float:
    """计算单条 trace 的持续时间。"""
    if trace.shape[0] <= 1:
        return 0.0
    timestamps = extract_timestamps(trace)
    return float(timestamps[-1] - timestamps[0])


def compute_insert_time(foreground_trace: np.ndarray, overlap_ratio: float) -> float:
    """根据主流量 A 的耗时，计算背景流量 B 的插播时间点。"""
    if not 0.0 <= overlap_ratio <= 1.0:
        raise ValueError(f"overlap_ratio 必须位于 [0, 1]，当前为 {overlap_ratio}")
    duration_a = compute_duration(foreground_trace)
    return duration_a * (1.0 - overlap_ratio)


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


def finalize_merged_trace(merged_trace: np.ndarray, template: np.ndarray) -> np.ndarray:
    """将统一表示的合并结果还原成输入样本的数据格式。"""
    merged_trace = stable_sort_trace(merged_trace)
    merged_trace = normalize_trace_start(merged_trace)

    if template.ndim == 1:
        restored = from_time_direction_trace(merged_trace, template.dtype)
        return pad_or_truncate(restored, MAX_LEN)

    restored = merged_trace.astype(template.dtype, copy=False)
    return pad_or_truncate(restored, MAX_LEN)


def merge_foreground_and_background(
    foreground_trace: np.ndarray,
    background_trace: np.ndarray,
    overlap_ratio: float,
) -> np.ndarray:
    """以 foreground 为主流量，在其尾段插播 background。"""
    if foreground_trace.ndim != background_trace.ndim:
        raise ValueError("foreground_trace 与 background_trace 的维度不一致，无法合并")

    if foreground_trace.ndim == 2 and foreground_trace.shape[1] != background_trace.shape[1]:
        raise ValueError("foreground_trace 与 background_trace 的特征维数不一致，无法合并")

    foreground = prepare_trace_for_merge(foreground_trace)
    background = prepare_trace_for_merge(background_trace)

    if foreground.shape[0] == 0:
        return finalize_merged_trace(background, background_trace)
    if background.shape[0] == 0:
        return finalize_merged_trace(foreground, foreground_trace)

    insert_time = compute_insert_time(foreground, overlap_ratio)
    shifted_background = shift_trace_in_time(background, insert_time)
    merged = np.concatenate([foreground, shifted_background], axis=0)
    return finalize_merged_trace(merged, foreground_trace)


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


def generate_known_unknown_samples(
    known_data: np.ndarray,
    known_labels: np.ndarray,
    unknown_data: np.ndarray,
    overlap_ratio: float,
    target_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 Known+Unknown 样本，标签跟随主流量的已知类标签。"""
    results_data = np.zeros(get_output_shape(known_data, target_count), dtype=known_data.dtype)
    results_labels = np.zeros((target_count,), dtype=known_labels.dtype)

    selected_known_indices = rng.permutation(known_data.shape[0])[:target_count]
    for output_idx, known_idx in enumerate(
        tqdm(selected_known_indices, desc="生成 Known+Unknown", dynamic_ncols=True, leave=False)
    ):
        unknown_idx = sample_index(rng, unknown_data.shape[0])
        results_data[output_idx] = merge_foreground_and_background(
            foreground_trace=known_data[known_idx],
            background_trace=unknown_data[unknown_idx],
            overlap_ratio=overlap_ratio,
        )
        results_labels[output_idx] = known_labels[known_idx]

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

    num_known_unknown = infer_num_known_samples(known_data.shape[0], limit_known)
    rng = np.random.default_rng(seed)

    merged_data, merged_labels = generate_known_unknown_samples(
        known_data=known_data,
        known_labels=known_labels,
        unknown_data=unknown_data,
        overlap_ratio=overlap_ratio,
        target_count=num_known_unknown,
        rng=rng,
    )
    merged_data, merged_labels = shuffle_in_unison(merged_data, merged_labels, rng)

    ensure_parent_dir(output_file.parent)
    save_dataset(output_file, merged_data, merged_labels, key_names)

    print(
        f"[完成] {output_file} | overlap={overlap_ratio:.2f} | "
        f"known+unknown={num_known_unknown} | "
        f"data={merged_data.shape} | labels={merged_labels.shape}"
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
    set_seed(FIX_SEED)

    known_files = iter_existing_known_files(DEFAULT_KNOWN_DIR, DEFAULT_KNOWN_FILES)
    if not known_files:
        raise FileNotFoundError(f"在 {DEFAULT_KNOWN_DIR} 下未找到任何可处理的已知类文件")

    ensure_parent_dir(DEFAULT_OUTPUT_DIR)
    for known_file in known_files:
        unknown_file_name = resolve_unknown_file(known_file.name)
        unknown_file = DEFAULT_UNKNOWN_DIR / unknown_file_name
        if not unknown_file.exists():
            raise FileNotFoundError(f"背景流量文件不存在：{unknown_file}")

        output_file = DEFAULT_OUTPUT_DIR / known_file.name
        if output_file.exists() and not OVERWRITE_OUTPUT:
            print(f"[跳过] {output_file} 已存在，如需重建请修改 OVERWRITE_OUTPUT")
            continue

        current_seed = FIX_SEED + int(OVERLAP_RATIO * 1000) + sum(ord(ch) for ch in known_file.name)
        build_output_dataset(
            known_file=known_file,
            unknown_file=unknown_file,
            output_file=output_file,
            overlap_ratio=OVERLAP_RATIO,
            limit_known=LIMIT_KNOWN,
            seed=current_seed,
        )


if __name__ == "__main__":
    main()
