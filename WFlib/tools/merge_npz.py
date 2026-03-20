import numpy as np
import os
import argparse


def merge_npz(input_paths, output_path):
    """
    合并多个 .npz 文件（均含 X、y 键）为一个 .npz 文件。

    Args:
        input_paths: 输入 .npz 文件路径列表
        output_path: 输出 .npz 文件路径
    """
    X_list, y_list = [], []

    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        data = np.load(path, allow_pickle=True)
        if "X" not in data or "y" not in data:
            raise KeyError(f"文件缺少 X 或 y 键: {path}")
        X_list.append(data["X"])
        y_list.append(data["y"])
        print(f"  已加载 {path}: X={data['X'].shape}, y={data['y'].shape}")

    X_merged = np.concatenate(X_list, axis=0)
    y_merged = np.concatenate(y_list, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(output_path, X=X_merged, y=y_merged)
    print(f"\n合并完成 -> {output_path}")
    print(f"  X: {X_merged.shape}, dtype={X_merged.dtype}")
    print(f"  y: {y_merged.shape}, dtype={y_merged.dtype}, min={y_merged.min()}, max={y_merged.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多个 .npz 文件（需含 X、y 键）")
    parser.add_argument("inputs", nargs="+", type=str, help="输入 .npz 文件路径（可多个）")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 .npz 文件路径")
    args = parser.parse_args()

    merge_npz(args.inputs, args.output)
