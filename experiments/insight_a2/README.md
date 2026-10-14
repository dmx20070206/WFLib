# Insight A2：目标域特征中是否仍有类别结构？

冻结源域模型，以目标域特征做无监督聚类，聚类完成后才使用真实标签计算诊断指标。入口与 A1 一致，无需手动指定类别数、TAM 或长度。

## 运行

在项目 `WF` 环境、仓库根目录执行：

```bash
python experiments/insight_a2/run.py \
  --checkpoint checkpoints/TemporalDrift/DMX/max_f1.pth \
  --datasets datasets/TemporalDrift
```

自动选择 day0（优先 valid，缺少时回退 train）和全部可用 day 文件，按天数排序；复用 A1 的权重解析、浮点整数标签转换和 TAM 自动选择/生成。模型支持当前仓库的 DMX/RF，类别数从权重推断。默认使用可见 cuda:0，无 CUDA 时使用 CPU。结果自动保存到 `experiments/insight_a2/results/<时间戳>/`，可以重复运行。

只分析 Day 0/30/90/270，并指定显卡：

```bash
python experiments/insight_a2/run.py \
  --checkpoint checkpoints/TemporalDrift/DMX/max_f1.pth \
  --period day0 datasets/TemporalDrift/valid.npz \
  --period day30 datasets/TemporalDrift/day30.npz \
  --period day90 datasets/TemporalDrift/day90.npz \
  --period day270 datasets/TemporalDrift/day270.npz \
  --device cuda:3
```

`--datasets` 也接受多个 NPZ 文件。原始数据与同目录 `tam_` 文件遵循 A1 的约定。各期类别编号必须一致。模型不更新权重、不训练分类器；eval 模式冻结 BatchNorm 和 Dropout，使用 inference_mode，全部参数关闭梯度。

## 默认实验设计

- **特征层**：`features.18`，即 DMX/RF 最后类别投影 `features.19` 之前的 ReLU。对时间维全局平均池化，得到每条流量 512 维特征。模型 forward 返回的第二项是类别投影之后的展平表示，不作为这里的默认特征。可用 `--feature-layer` 显式更改 hook 层；高于二维的输出统一对空间/时间维平均。
- **样本**：每期最多均匀随机抽取 10000 条，不按真实类别分层，所有 K/种子共用这批样本。`--max-samples 0` 使用全部样本。原始数据仍整体载入内存。
- **表征处理**：先 L2 归一化，再在各期独立拟合无监督 PCA，默认最多 64 维，不白化；输出解释方差比例。`--pca-dim 0` 关闭 PCA。PCA 不接收标签，但会改变距离几何；跨期 PCA 基底不同，不能把不同期的坐标直接相减。建议用关闭 PCA 的结果做敏感性检查。
- **K 扫描**：默认 `round(C × [0.5, 0.75, 1, 1.25, 1.5, 2])`，102 类对应 51/76/102/128/153/204。C 来自源模型，绝不使用目标实际类别数选择 K。跳过 `K >= N` 并记录；可用 `--ks 32 64 102 160 204` 指定。
- **多次聚类**：默认 MiniBatchKMeans、3 个种子 2024/2025/2026，每次 `n_init=5`、最多 300 轮。MiniBatch 适合多期、多 K 扫描；用 `--method kmeans` 切换完整 KMeans。两者都不能保证找到全局最优聚类。
- **稳定性**：同一批样本、同一 K 的不同种子划分，两两计算 ARI；报告全部值、均值和标准差。它度量初始化/批次随机性稳定性，不代表重新采样、跨时间或更换算法后的稳定性。种子至少两个，重复种子去重。

其他参数：`--seeds 11 22 33 44 55`、`--n-init 10`、`--batch-size 128`、`--threads 4`、`--seed 2024`（采样和 PCA）、`--output-dir PATH`。目录已存在时创建时间戳子目录。

## 指标含义

| 指标 | 计算与解释 |
| --- | --- |
| NMI | 聚类与网站标签的归一化互信息（arithmetic 归一化），越高关联越强；不作随机一致性校正，可能偏好较多簇。 |
| ARI | 聚类与真实标签的调整兰德指数，随机一致性校正后可为负。 |
| Silhouette | PCA 后聚类空间（禁用 PCA 时为 L2 特征）中，按预测簇计算欧氏距离轮廓系数，无需真实标签。每期固定、与标签无关的最多 2000 条子集，各次运行共用；可用 `--silhouette-samples` 调整。子集仅有一个簇或每点一簇时为 null。 |
| Oracle Cluster Accuracy | 聚类编号到源域真实类别的最大权一一匹配准确率，使用目标标签，只是离线诊断。 |

Oracle 使用 K×C 列联表做矩形 Hungarian 匹配：K>C 时多余簇无法匹配，计为错误；K<C 时部分真实类别无法匹配。未匹配簇的映射为 -1。这里**不做多簇到同类的多数投票合并**，以免 K 增大时人为提高准确率。因此不同 K 的 Oracle 分数受到一一匹配约束，应结合 NMI、ARI、轮廓系数判断。也记录实际非空簇数、每簇样本数和 inertia；请求 K 不等于证明存在 K 个自然簇。

每期另在原始池化特征和 L2 特征上分别报告：每个有样本类别的样本数、类内均方欧氏距离（总体方差定义，单样本为 0）、加权类内方差、总方差、类间方差和类内/总方差比例。方差在 PCA 前计算，以便 PCA 截断不掩盖类内分散；零总方差时比例为 null。网站组成、采样量不同会影响跨期比较。

A1 分析原模型的输出编号；A2 聚类的是分类头前的目标特征。较高 A2 分数仅支持该表征在当前预处理和聚类设置下保留类别结构，不证明可无标签恢复网站身份。低分也不能证明所有类别信息丢失。不要用最高标签指标选择 K 后宣称无监督模型性能；本程序保留全部 K 的结果，不自动选择“最佳 K”。

## 产物与验证

- `config.json`：参数和实际加载路径。
- `results.json`、`period_*/report.json`：每次指标、各 K 的均值/标准差、种子间稳定性、方差、实际簇数与映射。
- `summary.csv`：每时期、K、种子各一行，包含四个指标。
- `metrics_vs_k.png/pdf`：各时期四项指标随 K 变化（误差线为种子间标准差，不是置信区间）。
- `metrics_vs_period.png/pdf`：固定各 K 的跨时期曲线（横轴为等距时期，不代表真实天数间距）。
- `structure.png/pdf`：各 K 的稳定性，以及各期 L2 特征类内/总方差比例。
- `period_*/features.npz`：池化特征、用于聚类的变换后特征、标签、原文件样本索引及轮廓子集索引（相对抽样后的特征行）。
- `period_*/clusters.npz`、`contingencies.npz`：每个 K/种子的聚类分配和列联表（行=簇、列=真实类别）。

依赖：本项目 PyTorch 环境，以及 numpy、scipy、scikit-learn、threadpoolctl、matplotlib。

```bash
python -m unittest discover -s experiments/insight_a2 -p 'test_*.py'
```
