# A2.5：冻结特征下的分类规则可恢复性（实验三）

固定源模型，在互不重叠的目标训练/验证/测试划分上比较 Source Head、Logistic Regression、kNN。额外提供 Day 0 同结构对照和迁移线性探针。**目标监督探针是诊断实验，使用了额外标签，不是最终 Hybrid 性能；G_probe 不是纯粹的分类头漂移占比。**

## 运行

在项目 `WF` 环境、仓库根目录执行，只需权重和数据集：

```bash
python experiments/insight_a25/run.py \
  --checkpoint checkpoints/TemporalDrift/DMX/max_f1.pth \
  --datasets datasets/TemporalDrift \
  --device cuda:3
```

显卡参数可省略，默认可见 cuda:0 或 CPU。与 A1/A2 一样，自动识别 DMX/RF、源域类别数、TAM（1800 长度）及各时期；结果自动写入 `experiments/insight_a25/results/<时间戳>/`。也支持 `--period day0 PATH --period day30 PATH`，或 `--datasets PATH1 PATH2 ...`。

整个目录以 valid 作为 Day 0。若只能找到源训练文件作为 Day 0，会明确报错，避免把源模型已经训练过的样本作为独立测试。显式文件以 day0/d0/valid/tam_valid 命名时识别为 Day 0；若没有 Day 0，仍运行目标探针，但不生成 Day 0 迁移对照。若 valid 曾用于 checkpoint 选择，它并非完全独立的源域最终测试集，正式实验最好传入另一份源域留出数据并标记 `--period day0 ...`。

## 划分与公平性

所有时期先提取冻结特征，然后确定统一评估协议：

1. 取各期共同出现、且每期至少有 5 条样本的类别集合。其余类别及各期原始支持数记录在 `protocol.json`。结果针对这一共同类别子集，不等于原始全量闭集准确率。
2. 令 m 为所保留类别在所有时期中的最小支持数。每类训练数为 `min(100, floor(0.5*m))`（至少 2），验证数为 `min(40, floor(0.2*m))`（至少 1），测试数为 `min(100, m - 训练数 - 验证数)`。
3. 各时期使用完全相同的类别集合和每类训练/验证/测试数，类内随机抽样、三者不重叠；剩余样本不用。这样 Day 0 的较小 valid 数据不会与目标域大规模训练预算混为一谈，但也可能使所有期的训练预算较小。实际预算会打印并保存，可提供更大的独立 Day 0 集。
4. 默认种子为 2024/2025/2026，每次重新划分。跨种子的样本可能重叠，因此种子间标准差不是独立重复实验的置信区间。
5. Source Head 与所有探针都在**完全相同的测试索引**上评估。Source Head 保留完整输出空间，不根据共同类别集合屏蔽其余输出。

可用 `--train-per-class`、`--val-per-class`、`--test-per-class` 修改上限；数据不足时自动采用上述共同支持数预算。程序不使用测试准确率选择样本、特征、C 或 k。读取全量标签只用于建立分层划分及共同支持协议；分类器拟合只读取 train，超参选择只读取 val。

划分单位是 NPZ 中的样本行。若文件含重复采集或同会话相关样本，行不重叠不等于会话独立；本入口无会话 ID，正式研究需先按会话/采集组构造独立数据。也需确保各时期本身没有复制或重叠样本。

## 表示和分类器

源模型全程 eval、inference_mode、关闭梯度，不更新 BatchNorm。默认 hook `features.18`，即类别投影前的 512 通道表示，对时间维平均池化得到 512 维向量，与 A2 默认一致。可通过 `--feature-layer` 更改。原始 Source Head 在完整特征图上继续前向；探针仅接收池化向量，因此差异不仅可能来自分类头容量，也可能包含池化丢失空间/时间信息。不能据探针低分就断言完整冻结表征已丢失类别信息。

| 输出名 | 训练和选择方式 |
| --- | --- |
| `source_head` | 原始冻结源模型，同一目标测试集上的 Raw Accuracy。 |
| `linear_fixed` | StandardScaler + L2 Logistic Regression，所有期固定 `C=1`，用于同结构、同规模、同正则化对照。 |
| `linear_tuned` | 同一线性结构，仅按各期验证准确率在 C=[0.01,0.1,1,10,100] 中选择；正则化可能跨期不同，须与 fixed 分开解释。 |
| `knn` | L2 归一化特征、欧氏距离、均匀投票，验证集从 k=[1,3,5,11,21] 中选择；提供非参数非线性决策对照。 |
| `day0_linear_transfer` | 把同种子 Day 0 的 `linear_fixed`（包括源域训练划分拟合的 scaler）直接迁移到后续时期；在 Day 0 上与 linear_fixed 相同。与 target linear_fixed 的对比进一步控制分类器结构、训练数量和固定 C。 |

StandardScaler 只 fit train；kNN 的 L2 归一化按样本进行，无跨样本统计。**不使用 A2 已经在全目标域拟合的 PCA，也不在 train+val 上重新拟合。** 线性模型默认 lbfgs、max_iter=2000；每个候选的收敛状态和验证分数会记录，未收敛时打印提示，可增加 `--max-iter`。C 平分时选择较小 C，k 平分时选择较小 k，均不看测试分数。

参数可覆盖：`--fixed-c 1 --c-grid 0.1 1 10 --neighbors 1 3 5 --seeds 2024 2025 2026 --threads 4 --batch-size 128`。

## 输出和解释

- `probe_curves.png/pdf`：各期 Source Head、线性 fixed/tuned、kNN、Day 0 迁移探针的测试准确率及 G_probe。误差线为划分种子间标准差；横轴为等距时期。
- `summary.csv`：时期/种子/分类器准确率、G_probe、增益的配对 Bootstrap 95% 百分位区间。
- `results.json`、`period_*/report.json`：每个划分规模、全部验证搜索结果、选中 C/k、测试指标。
- `config.json`、`protocol.json`：参数、实际加载文件、各期类别支持数、排除类别及统一预算。
- `period_*/features.npz`：全量池化特征、真实标签及 Source Head 预测。
- `seed_*_predictions.npz`：train/val/test 原始文件行索引、y_test 及全部分类器测试预测。
- `seed_*_linear_fixed.npz` / `linear_tuned.npz`：scaler 均值/尺度、线性权重、截距和类别顺序。kNN 可由保存的训练索引、特征、选定 k 复现。

`G_probe = A_probe - A_source_head`，各分类器分别报告。Bootstrap 对同一测试样本的正确性差值配对重采样（默认 1000 次，可用 `--bootstrap` 修改），条件于已拟合分类器和当前划分，不包含重训练不确定性，也不处理会话相关性。测试为固定每类样本数的平衡诊断集；当前 bootstrap 是对整个测试集作 IID 重采样。

判断分支时同时看固定线性、调参线性和 kNN：明显增益支持冻结特征中有可重新利用的信息；kNN 明显优于线性提示判别边界复杂性可能重要；都接近 Raw 仅说明当前特征读出、样本预算和探针不能恢复更多，不能排除其他读出方式。Day 0 同结构对照是必要参照，不预设随时间单调下降。

验证：

```bash
python -m unittest discover -s experiments/insight_a25 -p 'test_*.py'
```
