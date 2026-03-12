# Private-DE 中文说明

本文档是当前仓库中 `private_de_v2/` 新实现路径的中文说明，重点覆盖三部分：

1. 这次重构到底改了什么
2. 现在应该怎么运行代码
3. 当前代码实际实现的算法逻辑是什么

注意：

- `private_de_v2/` 是新的主实现路径
- 顶层旧脚本 `main.py`、`primary.py`、`mygenerator.py`、`Pi.py` 和 `backup_code/` 仍然保留，但仅作为 legacy / baseline / experimental 参考
- 新实现以 `SPEC.md` 和 `ASSUMPTIONS.md` 为准，不再默认继承旧代码中的历史行为

## 1. 关键改动

### 1.1 新建了独立实现路径 `private_de_v2/`

旧仓库的问题主要是：

- 顶层脚本过于单体化
- JAX / Torch / CLI / 算法逻辑混在一起
- 有硬编码实验参数
- 有 placeholder 和不可达分支
- 很多逻辑难以单测

这次重构没有直接覆盖旧代码，而是新建了 `private_de_v2/`：

- `config.py`
- `data.py`
- `queries.py`
- `privacy.py`
- `selection.py`
- `measurement.py`
- `fitness.py`
- `mutation.py`
- `crossover.py`
- `generator.py`
- `evaluation.py`
- `cli.py`
- `tests/`

这样做的目的很明确：

- 保留旧代码作为基线参考
- 把新实现做成可测试、可复现、可解释的版本

### 1.2 去掉了旧代码中的硬编码实验入口

旧代码里存在这种行为：

```python
args = parser.parse_args(args=[...])
```

这会导致：

- 实验路径绑定在某个人机器上
- CLI 形同虚设
- 无法稳定复现实验

新实现改为：

- 从 YAML 配置文件读取参数
- 允许 CLI 覆盖配置
- 所有关键实验参数都显式暴露

入口现在统一为：

```bash
python -m private_de_v2
```

### 1.3 数据表示统一为“离散整数编码空间”

旧实现里一个大问题是：

- 有的逻辑在变换后的内部空间做
- 有的逻辑在反变换后的原始空间做
- 最后还会再 inverse transform 一次

新实现把内部表示统一成：

- 类别特征：离散类别编码
- 数值特征：先离散化，再编码为整数 bin

也就是说，在 `private_de_v2/` 中：

- 查询构造
- 查询评估
- fitness
- mutation
- crossover

全部在离散整数空间里进行。

最终输出 `synthetic.csv` 时，才把编码映射回可读标签。

### 1.4 查询工作负载和 query vector 变成显式对象

旧代码更多是 workload / marginal 的隐式组织方式，不是清晰的 Private-DE query vector 设计。

新实现中，`private_de_v2/queries.py` 显式引入了：

- `Query`
- `QueryVector`
- `QueryWorkload`

支持的 workload family 包括：

- `1way`
- `2way`
- `3way`
- `range`
- `prefix`
- `conditional_prefix`
- `halfspace`

并且支持消融：

- `--no-orthogonal-grouping`

这会把原本成组的正交 query vector 拆成 singleton vector。

### 1.5 隐私会计改成显式 zCDP accountant

旧代码中 privacy accounting 分散在多个地方，且有不一致的 sigma 写法。

新实现中：

- `private_de_v2/privacy.py` 负责 zCDP accountant
- `private_de_v2/measurement.py` 负责 Gaussian measurement

现在会明确记录：

- 初始化花了多少 `rho`
- 每轮 selection 花了多少 `rho`
- 每轮 measurement 花了多少 `rho`
- 累计已经花了多少 `rho`

并输出到：

- `metrics.jsonl`
- `summary.yaml`

### 1.6 修正并固定 signed error 约定

新实现中，统一使用：

```text
Err(q) = noisy_answer - synthetic_answer
```

解释如下：

- `Err(q) > 0`：合成数据低估了 noisy target
- `Err(q) < 0`：合成数据高估了 noisy target

这个约定现在在以下模块中是一致的：

- `selection`
- `measurement`
- `fitness`
- `mutation`
- `crossover`
- `evaluation`
- 单元测试

### 1.7 record-level fitness 独立成模块

旧代码里行级别评分混有 heuristic dominance 逻辑。

新实现中，`private_de_v2/fitness.py` 明确实现：

```text
f_V(x) = sum_{q in V} Err(q) * phi_q(x)
```

对正交 query vector 来说：

- 每条记录只匹配 vector 中的一个状态
- 因此记录 fitness 就是该状态对应的 signed error

### 1.8 directed mutation 和 crossover 被明确拆开

旧代码中 mutation / crossover / diversity injection 混杂较多。

现在：

- `mutation.py` 只负责 donor/recipient directed mutation
- `crossover.py` 只负责人口级别 crossover

这样做的好处是：

- 更容易单测
- 更容易做 ablation
- 更容易检查每一轮到底发生了什么

### 1.9 增加了测试、最小配置和可复现实验命令

新增内容：

- `configs/private_de_v2_minimal.yaml`
- `TEST_PLAN.md`
- `private_de_v2/tests/`

目前已经覆盖的核心测试包括：

- signed error 方向
- donor / recipient mutation 方向
- orthogonal vector mutation effect
- privacy monotonicity
- fixed seed determinism
- exponential mechanism 数值稳定性
- workload construction correctness
- inverse-variance weighting correctness

## 2. 目录说明

当前最重要的目录和文件如下：

- `private_de_v2/`
  - 新实现主路径
- `configs/private_de_v2_minimal.yaml`
  - 最小可运行配置
- `README.md`
  - 英文简版说明
- `Readme_zh.md`
  - 当前中文说明
- `ASSUMPTIONS.md`
  - 论文/规范模糊点的最终落地选择
- `SPEC_GAP_REPORT.md`
  - 当前实现与 spec 的剩余差距
- `TEST_PLAN.md`
  - 测试说明

legacy 参考代码仍然保留：

- `main.py`
- `primary.py`
- `mygenerator.py`
- `Pi.py`
- `backup_code/`
- `src/genetic_sd/`

## 3. 如何运行

## 3.1 最小运行方式

先准备一个 CSV 数据文件，然后运行：

```bash
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --output-dir outputs/run_example
```

如果你在 Windows PowerShell 下运行，也可以直接这样写：

```powershell
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path .\data.csv --output-dir .\outputs\run_example
```

运行结束后，会在 `output_dir` 下生成：

- `synthetic.csv`
- `metrics.jsonl`
- `summary.yaml`

### 各输出文件含义

- `synthetic.csv`
  - 最终合成数据

- `metrics.jsonl`
  - 每轮一条日志
  - 包含：
    - round index
    - selected vector id
    - selection rho
    - measurement rho
    - cumulative rho
    - vector loss before
    - vector loss after
    - best global loss
    - sigma
    - runtime

- `summary.yaml`
  - 整个实验的汇总
  - 包括：
    - 数据集路径
    - 记录数
    - 使用的总 `rho`
    - 对应 `(epsilon, delta)` 解释值
    - 轮数
    - population size
    - evaluation 指标

## 3.2 用 CLI 覆盖配置

除了配置文件外，也可以直接覆盖关键参数，例如：

```bash
python -m private_de_v2 \
  --config configs/private_de_v2_minimal.yaml \
  --dataset-path path/to/data.csv \
  --output-dir outputs/run_debug \
  --rounds 20 \
  --population-size 8 \
  --synthetic-size 5000 \
  --epsilon 2.0 \
  --delta 1e-6 \
  --seed 42 \
  --families 1way 2way prefix
```

常用 CLI 参数包括：

- 数据与输出
  - `--dataset-path`
  - `--output-dir`

- 隐私参数
  - `--epsilon`
  - `--delta`
  - `--total-rho`
  - `--initialization-rho-fraction`
  - `--selection-rho-fraction`

- 算法参数
  - `--rounds`
  - `--population-size`
  - `--synthetic-size`
  - `--max-mutations-per-round`
  - `--crossover-rate`
  - `--crossover-rows`
  - `--seed`
  - `--device`

- workload 参数
  - `--families`
  - `--max-vector-size`
  - `--range-widths`
  - `--prefix-thresholds-per-feature`
  - `--conditional-prefix-thresholds-per-feature`
  - `--conditional-prefix-max-condition-values`
  - `--halfspace-thresholds-per-pair`
  - `--halfspace-pair`

## 3.3 消融实验运行方式

### 关闭 directed mutation

```bash
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-directed-mutation
```

### 关闭 crossover

```bash
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-crossover
```

### 关闭 orthogonal grouping

```bash
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-orthogonal-grouping
```

### 关闭 inverse-variance weighting

```bash
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-inverse-variance-weighting
```

## 3.4 运行测试

```bash
python -m unittest discover -s private_de_v2/tests -v
```

## 4. 配置文件说明

最小配置文件在：

```text
configs/private_de_v2_minimal.yaml
```

主要字段如下：

- `data`
  - 数据集路径、输出目录、离散化配置

- `workload`
  - query family、vector 大小、prefix/range/halfspace 等构造参数

- `privacy`
  - `epsilon` / `delta` 或 `total_rho`
  - 初始化预算占比
  - 每轮 selection 预算占比

- `selection`
  - 当前使用的 utility score 配置

- `algorithm`
  - 轮数
  - population size
  - synthetic size
  - mutation / crossover 参数
  - seed

- `ablations`
  - 四个主要消融开关

- `output`
  - 输出文件名

## 5. 当前代码实现的算法逻辑

下面不是论文表述，而是当前代码真实执行的逻辑。

核心主循环在：

```text
private_de_v2/generator.py
```

## 5.1 第一步：加载数据并离散化

模块：

```text
private_de_v2/data.py
```

流程：

1. 读取 CSV
2. 如果 `drop_missing=true`，则删除缺失值行
3. 对每一列建立 `DiscreteColumn`
4. 对数值列执行离散化
5. 把整张表变成整数编码矩阵

当前内部表示是：

- 每行是一条离散记录
- 每列是一个整数编码后的属性值

## 5.2 第二步：构造 workload 和 query vector

模块：

```text
private_de_v2/queries.py
```

流程：

1. 先构造初始化用的所有 `1-way` vectors
2. 再根据 `workload.families` 构造本轮候选 workload
3. 每个 workload family 都会生成显式的 `QueryVector`

例如：

- `1way`
  - 一个属性的所有类别组成一个 vector

- `2way` / `3way`
  - 多个属性的笛卡尔积状态组成一个 vector

- `range`
  - 对有序列按宽度切分成不重叠区间

- `prefix`
  - 构造成对的 `<= t` 和 `> t`

- `conditional_prefix`
  - 构造成三元组：
    - condition 不成立
    - condition 成立且 `<= t`
    - condition 成立且 `> t`

- `halfspace`
  - 当前实现是两个有序特征离散值求和后的阈值二分

如果设置：

```text
--no-orthogonal-grouping
```

则每个 query 会被拆成 singleton vector。

## 5.3 第三步：初始化阶段先私有测量所有 1-way marginals

这一步是当前实现中很重要的改动。

旧 PT 代码是随机初始化；新代码改成：

1. 对每个 `1-way` vector 分配初始化预算
2. 使用 Gaussian mechanism 测量 noisy answers
3. 把 noisy 结果裁剪到非负并归一化成概率分布
4. 对每一列独立采样，生成初始 synthetic population

对应代码：

- 初始化 vector 构造：`build_initialization_vectors(...)`
- 测量：`gaussian_measure_vector(...)`
- 初始化采样：`_sample_initial_population(...)`

当前初始化策略是：

- product of private 1-way marginals

这是一个明确的、可解释的、可复现的初始化方案。

## 5.4 第四步：建立 zCDP accountant 并分配预算

模块：

```text
private_de_v2/privacy.py
private_de_v2/generator.py
```

流程：

1. 如果配置里给了 `total_rho`，直接使用
2. 否则由 `(epsilon, delta)` 推导 `rho`
3. 先拿出 `initialization_rho_fraction` 做初始化
4. 剩余预算平均分给所有 round
5. 每轮再拆成：
   - `selection_rho`
   - `measurement_rho`

所以当前预算结构是：

```text
total_rho
  -> initialization
  -> round_0
       -> selection
       -> measurement
  -> round_1
       -> selection
       -> measurement
  ...
```

## 5.5 第五步：每轮私有选择一个 query vector

模块：

```text
private_de_v2/selection.py
```

当前实现不是直接照搬某个复杂 paper 公式，而是采用一个更简单可测试的 utility：

```text
utility(V) = clipped L1 gap between real answers and synthetic answers
```

具体过程：

1. 对每个候选 vector，计算：
   - 真实数据在该 vector 上的答案
   - 当前 synthetic best population 在该 vector 上的答案
2. 计算两者的 L1 gap
3. 经过 size weight 和 clip 得到 utility
4. 用 exponential mechanism 进行私有采样

稳定性处理：

- 在概率计算时做了 logits 平移，避免数值溢出

## 5.6 第六步：对被选中的 vector 做 Gaussian measurement

模块：

```text
private_de_v2/measurement.py
```

流程：

1. 计算真实数据在该 vector 上的频率答案
2. 根据 `rho` 和 sensitivity 计算 `sigma`
3. 加 Gaussian noise
4. 把 noisy answers 记录到 `MeasurementStore`

当前 sensitivity 约定：

- singleton vector：`1 / N`
- 多 query 的 grouped orthogonal vector：`sqrt(2) / N`

噪声尺度：

```text
sigma = sensitivity / sqrt(2 * rho)
```

## 5.7 第七步：如果某个 query 被重复测量，则做 inverse-variance weighting

模块：

```text
private_de_v2/measurement.py
```

当前实现支持：

- 同一个 query id 被重复直接测量时
- 用 inverse-variance weighting 合并

也就是权重为：

```text
weight = 1 / variance
```

如果关闭：

```text
--no-inverse-variance-weighting
```

则会改成保留最新值。

## 5.8 第八步：计算 signed errors 和 record-level fitness

模块：

```text
private_de_v2/fitness.py
```

当前实现固定：

```text
Err(q) = noisy_answer - synthetic_answer
```

然后对选中的 vector 计算每条记录的 fitness：

```text
f_V(x) = sum_{q in V} Err(q) * phi_q(x)
```

由于当前 vector 通常是正交的，因此一条记录只会匹配其中一个 query state。

所以记录 fitness 的含义非常直观：

- 如果落在 underrepresented state，fitness 为正
- 如果落在 overrepresented state，fitness 为负

## 5.9 第九步：做 directed mutation

模块：

```text
private_de_v2/mutation.py
```

逻辑如下：

1. 找到 `Err(q) > 0` 的 recipient states
2. 找到 `Err(q) < 0` 的 donor states
3. 根据误差大小决定本轮要做多少次 mutation
4. 从 donor state 中选记录
5. 把该记录投影到 recipient state

当前“closest / projection”的定义是显式的：

- equality query：直接把对应属性改成目标值
- range / prefix：把值 clamp 到合法区间
- categorical 退出某个 state：选最近的备选离散值
- halfspace：通过调整两个有序特征使其满足目标阈值条件

也就是说，当前 mutation 是：

- record-space
- inspectable
- deterministic projection based

不是黑箱式的连续参数优化。

## 5.10 第十步：做 population-level crossover

模块：

```text
private_de_v2/crossover.py
```

逻辑如下：

1. 先根据所有已测 query 的 MSE，找出当前最好的 population
2. 在其他 population 中，找出 fitness 最差的记录
3. 从最好 population 中抽 donor rows
4. 用 donor rows 替换掉差的 rows

当前 crossover 是一个清晰的独立操作，但它仍然是相对简单的版本：

- donor population：全局 measured-query loss 最小
- donor rows：优先来自 active vector 中的正误差状态
- recipient rows：active vector 下最差的记录

## 5.11 第十一步：记录每轮日志并输出最终结果

每轮日志写入：

```text
metrics.jsonl
```

最终汇总写入：

```text
summary.yaml
```

最终最佳 population 会被解码后输出为：

```text
synthetic.csv
```

## 6. 当前实现中的几个重要约定

这些约定已经写入代码，不是“未来可能”的设想。

### 6.1 使用频率，不使用 raw count

当前所有 query answer 都是 frequency。

优点：

- sensitivity 更清晰
- selection / measurement / evaluation 一致

### 6.2 当前没有 projection placeholder

旧代码里有一个假函数：

```python
project_2way_to_consistent_1way(...)
```

新实现没有保留它。

因为 placeholder 如果没有真正实现，就不应该伪装成算法的一部分。

### 6.3 当前 halfspace 是“简化版”

并不是泛化线性分类器意义上的 halfspace。

现在的实现只是：

- 两个有序属性
- 单位权重
- 对离散值和做阈值二分

这是一个明确、可解释、可测试的版本，但不是最通用形式。

### 6.4 当前 selection score 是简化版本

不是某种复杂的 paper 常数乘积形式，而是：

- clipped L1 frequency gap

这样做的原因是：

- 更稳定
- 更容易单测
- sensitivity 更容易写清楚

## 7. 推荐阅读顺序

如果你要继续开发，建议按下面顺序读代码：

1. `private_de_v2/config.py`
2. `private_de_v2/cli.py`
3. `private_de_v2/data.py`
4. `private_de_v2/queries.py`
5. `private_de_v2/privacy.py`
6. `private_de_v2/selection.py`
7. `private_de_v2/measurement.py`
8. `private_de_v2/fitness.py`
9. `private_de_v2/mutation.py`
10. `private_de_v2/crossover.py`
11. `private_de_v2/generator.py`
12. `private_de_v2/evaluation.py`

## 8. 当前如何判断运行是否正常

一次正常运行后，至少应该检查：

1. `synthetic.csv` 是否生成
2. `metrics.jsonl` 是否每轮都有记录
3. `summary.yaml` 里的：
   - `total_rho_spent` 是否没有超过预算
   - `rounds_completed` 是否符合预期
   - `evaluation` 是否存在

如果想确认可复现性：

1. 用相同 seed 跑两次
2. 比较：
   - `synthetic.csv`
   - `metrics.jsonl`
   - `summary.yaml`

## 9. 测试

运行方式：

```bash
python -m unittest discover -s private_de_v2/tests -v
```

当前已经覆盖的测试包括：

- `test_fitness.py`
  - signed error
  - record-level fitness

- `test_mutation.py`
  - donor/recipient 方向
  - orthogonal vector mutation effect

- `test_privacy_selection_measurement.py`
  - privacy monotonicity
  - exponential mechanism stability
  - inverse-variance weighting

- `test_queries.py`
  - workload construction
  - no-orthogonal-grouping 行为

- `test_generator_smoke.py`
  - fixed seed 下的端到端 deterministic smoke test

## 10. 仍然保留但不建议继续扩展的代码

以下文件依然存在，但不建议再往上堆新逻辑：

- `main.py`
- `primary.py`
- `mygenerator.py`
- `Pi.py`
- `backup_code/`

原因：

- 结构混杂
- 历史行为不稳定
- 部分逻辑与新 spec 不一致
- 有 placeholder、重复实现和不可达分支

如果后续要继续开发 Private-DE，建议只在：

```text
private_de_v2/
```

上继续推进。
