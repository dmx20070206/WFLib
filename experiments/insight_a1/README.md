# Insight A1：混淆还是类别身份错配？

冻结源域分类器，在每个目标时期计算 `A_raw`、一一置换诊断上界 `A_perm` 和差值 `G_perm`。不训练新模型，不做聚类。所有数据、模型、权重和输出路径通过命令行传入。

## 按本项目直接运行

在已安装本项目依赖的 `WF` 环境、仓库根目录执行。只需输入预训练权重与数据集：

```bash
python experiments/insight_a1/run.py \
  --checkpoint checkpoints/TemporalDrift/DMX/max_f1.pth \
  --datasets datasets/TemporalDrift
```

也可以只分析指定文件（顺序即曲线顺序）：

```bash
python experiments/insight_a1/run.py \
  --checkpoint checkpoints/TemporalDrift/DMX/max_f1.pth \
  --datasets datasets/TemporalDrift/valid.npz datasets/TemporalDrift/day270.npz
```

默认自动选择可见的 `cuda:0`，无 CUDA 时用 CPU。共享机器上可加 `--device cuda:3` 选择卡。

自动配置遵循当前项目约定：

- 从权重父目录名识别 `DMX` / `RF`，从分类层权重推断完整类别数（这份 DMX 权重为 102 类）。权重移到其他目录后可显式加 `--model DMX`。
- DMX / RF 使用 TAM、长度 1800，与 `scripts/TemporalDrift/DMX.sh` 一致。输入 `train.npz` 等原始文件时优先加载同目录 `tam_train.npz`，并核对标签；没有预生成 TAM 时，复用项目算法对前 5000 个包生成 TAM，不修改源数据。
- 数据集目录自动选 `valid.npz` 为 day0，缺少时回退到 train；支持仅有 `tam_` 文件的目录。按数值顺序读取所有 day 文件，并去除 raw/TAM 重复项。显式提供 train 时仍使用 train，但这是训练集基线，解读时应区别于独立测试集。
- 接受项目中以 float64 存储的整数标签，安全转为 int64；仍拒绝小数、NaN、越界标签或多标签数组。各时期必须沿用同一套源域编号。
- 默认保存到本目录 `results/<时间戳>/`。指定 `--output-dir` 已存在时自动新建时间戳子目录，失败后可直接重跑原命令。

原有 `--period NAME PATH`、`--feature`、`--seq-len`、`--num-classes` 等参数仍可用来覆盖默认配置；`--num-classes` 必须与权重一致。自动配置仅覆盖本仓库现有 DMX/RF 约定；其他训练预处理需显式指定。

已有预测也可单独分析，无需 PyTorch：

```bash
python experiments/insight_a1/run.py \
  --input-mode predictions --num-classes 102 \
  --datasets /external/D0.npz /external/D270.npz
```

预测 NPZ 包含一维 `y_true`、`y_pred`，可用 `--true-key`、`--pred-key` 更改字段名。分析依赖 `numpy scipy matplotlib`。

外部模型接口仍保留：`--model-factory module:callable --num-classes C --model-kwargs '{...}'`；工厂接受 `num_classes` 并返回 PyTorch 模型。`--loader module:callable --loader-kwargs '{...}'` 返回预处理的 `(X, y)`。模块需位于 PYTHONPATH。模型输出 `[N,C]` 分数，tuple/list 默认取第 0 项（可用 `--logits-index` 调整）。默认加载纯 state_dict，也自动识别 `state_dict` / `model_state_dict` 包装及 `module.` 前缀；其他包装用 `--checkpoint-key`。权重严格加载，使用 `weights_only=True`。

`--repeats` 默认 1000，`--confidence` 默认 0.95，`--seed` 默认 2024。更多参数见 `--help`。本实验针对统一标签空间的单标签闭集分类，不训练或更新模型。

## 统计设计与解释

混淆矩阵行是预测类别、列是真实类别。Hungarian 最大权匹配产生完整一一映射 `mapping_pred_to_true[j]`，表示源输出 j 应对应的真实类别。恒等映射是候选，因此同样本上的 `A_perm >= A_raw`。零支持类别和并列最优解的映射不可作身份解释；支持数随结果保存。

每时期独立计算以下基线（`--repeats` 控制次数）：

- **随机预测**：在完整 C 类中均匀独立抽样预测，再重新求最优匹配，报告三个指标的分布。
- **随机类别置换**：对实际预测随机抽取一一映射，报告映射后准确率分布。
- **标签打乱**：保留真实类别和预测类别的边际频数，打乱样本对应关系，每次重新求最优匹配。报告三个指标分布及 `A_perm` 的单侧 Monte Carlo p 值 `(1 + 超过或等于观测值次数)/(B+1)`。该检验评估打乱零假设下的关联，不能证明漂移机制。
- **配对 Bootstrap**：对 `(y_true, y_pred)` 成对 IID 重采样，每次重新优化映射，给出百分位区间。实现对混淆矩阵作等价的多项式抽样，避免反复建立样本数组。
- **留出映射检查**：固定种子随机划分样本，仅在拟合部分求映射，在留出部分计算 `A_mapped`、`G_mapped`；`--holdout` 默认 0.5。这不是留出部分的最优置换上界，增益可以为负；单样本时不可用。单次划分只作辅助检查，少样本类别可能未进入拟合部分。

Bootstrap 区间不是偏差校正；随机基线的 interval 是模拟分布范围，不是均值的置信区间。若采样存在同一会话/采集批次相关性，应使用分组采样设计，本脚本的 IID 区间不能覆盖该依赖。跨时期比较还需控制网站组成和采样方式，不能仅据曲线下降断言漂移。

**`A_perm` 使用目标真实标签，是离线诊断上界，不是 Hybrid 方法或可部署攻击的准确率。** 大增益仅表明存在可由一一映射解释的错误结构；小增益不单独证明信息丢失。留出映射同样使用部分目标标签，不代表无监督方法。有限样本、类别不平衡和缺失类别都需结合基线和支持数解释。

## 输出

每次运行保留独立结果，包含：

- `config.json`：全部参数；`results.json`：所有时期完整统计；`summary.csv`：主指标。
- `curves.png/pdf`：原始/置换准确率及 Bootstrap 区间、增益及随机预测/标签打乱均值。时期依照输入顺序排列，横轴为等距类别，不表示真实天数间距。
- `period_000/` 等：`predictions.npz`、`confusion.npy`、`report.json`（映射、支持数、留出结果等）和 `samples.npz`（每次模拟的原始统计量，三列依次为 A_raw/A_perm/G_perm）。

所有数值为 0–1 比例。大 C 的匹配代价较高，建议先用 `--repeats 100` 检查数据，再增加到 1000 或更多。未附带真实数据或实验结论。

验证：`python -m unittest discover -s experiments/insight_a1 -p 'test_*.py'`。
