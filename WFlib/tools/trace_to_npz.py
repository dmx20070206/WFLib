import numpy as np
import os
import glob
from tqdm import tqdm
import argparse

def convert_trace_to_directional_ts(trace_dir, output_path, max_len=10000):
    """
    将标准 Trace 文件夹转换为 '带方向的时间戳' (Directional Timestamp) 格式的 npz。
    逆向逻辑: Value = Timestamp * Direction
    
    Args:
        trace_dir: 包含 .txt trace 文件的文件夹路径
        output_path: 输出 .npz 文件的完整路径
        max_len: npz 矩阵的固定列宽 (最大包数量) 超过截断
    """
    
    print(f"正在处理文件夹: {trace_dir}")
    
    # 获取所有 txt 文件
    files = glob.glob(os.path.join(trace_dir, "*.txt"))
    if not files:
        print("未找到 .txt 文件")
        return

    data_list = []
    label_list = []
    
    # 排序以保证顺序可复现 (可选，根据文件名中的 index 排序)
    # 假设文件名格式为 Label_Index.txt，尝试按 Index 排序，如果失败则按文件名排序
    try:
        files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    except:
        files.sort()

    for file_path in tqdm(files, desc="Reading Traces", dynamic_ncols=True):
        filename = os.path.basename(file_path)
        
        # 1. 解析 Label (假设文件名格式: Label_Index.txt)
        try:
            label_str = filename.split('_')[0]
            label = float(label_str)
        except ValueError:
            # 如果文件名不符合格式，默认标签为 -1
            label = -1
        
        # 2. 读取 Trace 数据
        row_data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            try:
                ts = float(parts[0])
                direction = int(parts[1]) # 1 or -1
                
                # --- 核心逆向转换逻辑 ---
                # 还原 Directional Timestamp: 
                # 如果 direction 是 1 (正), val = ts
                # 如果 direction 是 -1 (负), val = -ts
                val = ts * direction
                row_data.append(val)
            except ValueError:
                continue
        
        # 3. 填充或截断 (Padding / Truncating)
        row_np = np.array(row_data, dtype=np.float64)
        
        if len(row_np) > max_len:
            # 截断
            padded_row = row_np[:max_len]
        else:
            # 不足补0 (Padding)
            padded_row = np.zeros(max_len, dtype=np.float64)
            padded_row[:len(row_np)] = row_np
            
        data_list.append(padded_row)
        label_list.append(label)

    # 转换为 Numpy 数组
    X = np.array(data_list)
    Y = np.array(label_list)
    
    print(f" -> 数据形状: {X.shape}")
    print(f" -> 标签形状: {Y.shape}")
    
    # 保存为 npz
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, X=X, y=Y)
    print(f"保存成功: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trace files to directional timestamp npz.")
    parser.add_argument('--input', type=str, required=True, help='Input trace folder path')
    parser.add_argument('--output', type=str, required=True, help='Output npz file path')
    parser.add_argument('--max_len', type=int, default=10000, help='Maximum sequence length (default: 10000)')
    args = parser.parse_args()

    convert_trace_to_directional_ts(args.input, args.output, max_len=args.max_len)
