import numpy as np
import os
import argparse 
from tqdm import tqdm

def convert_directional_ts_to_trace(npz_path, output_dir, default_len=0):
    """
    将 '带方向的时间戳' (Directional Timestamp) 格式的 npz 转换为标准 Trace。
    格式: <timestamp> <direction> <length>
    
    Args:
        npz_path: .npz 文件路径
        output_dir: 输出文件夹
        default_len: 由于源数据丢失了包大小，第三列填充的默认值 (建议填 0 或 1)
    """
    
    print(f"正在处理: {npz_path}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        # 自动查找数据键名
        keys = list(data.keys())
        x_key = next((k for k in ['data', 'x', 'features'] if k in keys), keys[0])
        y_key = next((k for k in ['labels', 'y', 'label'] if k in keys), None)
        
        X = data[x_key]
        Y = data[y_key] if y_key else None
        
        print(f" -> 数据形状: {X.shape}")
        
    except Exception as e:
        print(f"读取错误: {e}")
        return

    # 创建子文件夹
    file_stem = os.path.splitext(os.path.basename(npz_path))[0]
    save_dir = os.path.join(output_dir, file_stem)
    os.makedirs(save_dir, exist_ok=True)
    print(f" -> 输出至: {save_dir}")

    # 循环处理每个样本
    count = 0
    for i in tqdm(range(len(X)), desc="Writing TXT"):
        # 获取单个样本的数据 (一行)
        row = X[i]
        
        # 展平数据 (防止维度是 [5000, 1])
        row = row.flatten()
        
        # 过滤掉填充的0 (通常尾部会有很多0)
        # 注意：在 Directional Timestamp 格式中，0.0 通常代表第一个包，
        # 所以我们只过滤掉 "尾部的0" 或者 "绝对值极小的填充"
        # 这里简单起见，如果你的数据是用0做padding的，可以这样过滤：
        valid_mask = (row != 0) 
        trace_data = row[valid_mask]
        
        # 如果第一个包恰好是0.0，上面的过滤可能会误删，这里做一个保险：
        # 如果数据开头就是0，且后面有数据，说明0是有效的时间戳。
        # (根据你提供的数据，通常padding在末尾且非常多)
        if len(row) > 0 and row[0] == 0:
             # 简单的处理：只取非零部分，再加上第一个0 (如果它被误删了)
             # 但更稳妥的方式是不按值过滤，而是按索引（如果你知道有效长度）。
             # 针对你提供的这种数据，通常直接 trim 掉后面的 0 即可。
             trace_data = np.trim_zeros(row, 'b') # 'b' 表示只修剪 back (尾部) 的 0
        else:
             trace_data = np.trim_zeros(row, 'b')

        if len(trace_data) == 0:
            continue

        # --- 核心转换逻辑 ---
        lines = []
        for val in trace_data:
            # 1. 解析时间戳 (Timestamp) = 绝对值
            ts = abs(val)
            
            # 2. 解析方向 (Direction) = 符号
            # 正数为出 (1), 负数为入 (-1)
            # 如果 val 是 0，通常默认为出 (1) 或者视作第一个包
            direction = 1 if val >= 0 else -1
            
            # 3. 解析长度 (Length)
            # 你的数据里没有长度，所以用 default_len 填充
            length = default_len
            
            # 格式化字符串: TS <tab> DIR <tab> LEN
            line = f"{ts:.6f}\t{direction}\t{length}"
            lines.append(line)
        
        # --- 写入文件 ---
        # 命名格式: Label_Index.txt
        if Y is not None:
            label = int(Y[i])
            fname = f"{label}_{i}.txt"
        else:
            fname = f"{i}.txt"
            
        with open(os.path.join(save_dir, fname), 'w') as f:
            f.write('\n'.join(lines))
            
        count += 1

    print(f"成功转换 {count} 个文件。\n")

# --- 运行配置 ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Convert directional timestamp npz to trace files.")
    parser.add_argument('--input', type=str, required=True, help='Input trace folder path')
    parser.add_argument('--output', type=str, required=True, help='Output npz file path')
    args = parser.parse_args()

    convert_directional_ts_to_trace(args.input, args.output, default_len=0)