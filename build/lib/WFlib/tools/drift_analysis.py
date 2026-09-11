import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from utils import set_academic_style

set_academic_style()
SAMPLE_RATIO = 1

def load_and_clean(path):
    """读取并清洗数据"""
    with np.load(path) as data:
        key = "data" if "data" in data else list(data.keys())[0]
        raw = data[key]
    
    # --- 新增：随机采样逻辑 ---
    total_size = len(raw)
    sample_size = int(total_size * SAMPLE_RATIO)
    if sample_size < total_size:
        indices = np.random.choice(total_size, sample_size, replace=False)
        raw = raw[indices]
        print(f"  [Info] Sampled {sample_size}/{total_size} samples from {path}")
    # -----------------------

    cleaned = []
    for row in raw:
        valid = row[row != 0]
        if len(valid) > 10:
            cleaned.append(valid)
    return cleaned


def get_bursts_stats(trace):
    dirs = np.sign(trace)
    change_points = np.where(np.diff(dirs) != 0)[0] + 1
    bursts = np.split(trace, change_points)
    num_bursts = len(bursts)
    # 提取所有 Burst 的长度用于计算平均大小
    burst_sizes = [len(b) for b in bursts if len(b) > 0]
    mean_burst_size = np.mean(burst_sizes) if burst_sizes else 0
    return num_bursts, mean_burst_size

def extract_advanced_features(traces):
    """
    修改后的特征提取：与 TrafficDriftSimulator 的 7 个参数一一对应
    """
    feats = {
        "global_shift": [],      # 对应 global_shift
        "jitter_std": [],        # 对应 jitter_sigma
        "packet_count": [],      # 对应 packet_loss (观察包总量减少)
        "burst_size_avg": [],    # 对应 burst_scale
        "frag_burst_cnt": [],    # 对应 frag_prob (观察 Burst 数量增加)
        "head_density": [],      # 对应 front_pad_crop
        "tail_duration": [],     # 对应 tail_crop_ratio
    }
    
    for t in traces:
        timestamps = np.abs(t)
        directions = np.sign(t)
        
        # 1. Global Shift: 考察首包延迟 (RTT基准)
        feats["global_shift"].append(timestamps[50])

        # 2. Jitter: 考察 IAT (间隙) 的波动性
        if len(timestamps) > 1:
            iat = np.diff(timestamps)
            feats["jitter_std"].append(np.std(iat))
        else:
            feats["jitter_std"].append(0)

        # 3. Packet Loss: 考察序列总长度 (有效包数)
        feats["packet_count"].append(len(t))

        # 4. Burst Scale: 考察平均每个 Burst 的包数量
        nb, mbs = get_bursts_stats(t)
        feats["burst_size_avg"].append(mbs)

        # 5. Frag Prob: 考察 Burst 总数 (分片越多，Burst 越多)
        feats["frag_burst_cnt"].append(nb)

        # 6. Front Pad/Crop: 考察前 1.0s 内的数据包密度
        head_mask = timestamps < (timestamps[0] + 1.0)
        feats["head_density"].append(np.sum(head_mask))

        # 7. Tail Crop: 考察从首包到末包的总持续时间
        feats["tail_duration"].append(timestamps[-1] - timestamps[0])

    return feats

def plot_metrics(feat0, feat30, name, ax):
    sns.kdeplot(feat0, label="Source", fill=True, alpha=0.3, ax=ax, color="blue", common_norm=False)
    sns.kdeplot(feat30, label="Target", fill=True, alpha=0.3, ax=ax, color="red", common_norm=False)
    wd = wasserstein_distance(feat0, feat30)
    m0, m30 = np.mean(feat0), np.mean(feat30)
    diff = (m30 - m0) / m0 * 100 if m0 != 0 else 0
    ax.set_title(f"{name}\nWD={wd:.3f}, Diff={diff:.1f}%")
    ax.legend()

def main(path0, path30):
    print(f"Loading (Sampling Rate: {SAMPLE_RATIO*100}%)...")
    t0 = load_and_clean(path0)
    t30 = load_and_clean(path30)

    f0 = extract_advanced_features(t0)
    f30 = extract_advanced_features(t30)

    # 定义指标与参数的映射关系用于绘图
    metrics = [
        ("global_shift", "Global Shift (RTT)"),
        ("jitter_std", "Jitter Sigma"),
        ("packet_count", "Packet Loss Effect"),
        ("burst_size_avg", "Burst Scale"),
        ("frag_burst_cnt", "Fragmentation (Frag Prob)"),
        ("head_density", "Front Pad/Crop"),
        ("tail_duration", "Tail Crop Ratio Effect"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    summary_results = []

    for i, (key, title) in enumerate(metrics):
        wd = wasserstein_distance(f0[key], f30[key])
        m0, m30 = np.mean(f0[key]), np.mean(f30[key])
        diff = (m30 - m0) / m0 * 100 if m0 != 0 else 0
        
        summary_results.append({"name": title, "wd": wd, "diff": diff})
        
        # 绘图逻辑
        sns.kdeplot(f0[key], label="Original", fill=True, ax=axes[i], color="blue")
        sns.kdeplot(f30[key], label="Target", fill=True, ax=axes[i], color="red")
        axes[i].set_title(f"{title}\nWD={wd:.3f}, Δ={diff:.1f}%")
        axes[i].legend()

    axes[-1].axis("off")
    save_path = "plots/result_sampled.png"
    plt.savefig(save_path, dpi=300)
    
    # ==========================================
    # 打印最终总结
    # ==========================================
    print("\n" + "="*60)
    print("             TRAFFIC DRIFT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Source Data: {path0}")
    print(f"Target Data: {path30}")
    print(f"Sample Rate: {SAMPLE_RATIO*100}%")
    print(f"Samples (Source/Target): {len(t0)} / {len(t30)}")
    print(f"Visualization saved to: {save_path}")
    print("-"*60)
    print(f"{'Metric Name':<30} | {'WD Score':<10} | {'Mean Diff (%)':<10}")
    print("-"*60)
    for res in summary_results:
        print(f"{res['name']:<30} | {res['wd']:<10.4f} | {res['diff']:>+9.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main(
        "datazoo/TemporalDrift/Undefended/npzs/train.npz",
        "datazoo/TemporalDrift/Undefended/npzs/day270.npz",
    )
