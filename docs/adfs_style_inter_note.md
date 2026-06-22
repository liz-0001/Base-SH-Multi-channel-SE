# grouping-inter-sds / ADFS-style 分支修改说明

## 实验定位

本分支是 `grouping-inter` 的升级版，不是推翻重来。它保留：

```text
order-wise SH grouping
+ low/high SH order split
+ adjacent-order interaction
+ 原 TFGridNetV2 主干
+ 原训练流程、loss、推理和测评脚本框架
```

主要替换的是 `HighLowMutualGuidance` 的交互方式：

```text
grouping-inter v1: 单边上下文 gate

grouping-inter-sds / ADFS-style: SDS-Net ADSF-style low/high correlation gate
```

## 为什么要改

旧版 inter 的低阶指导高阶大致是：

```text
H_h_new = H_h + H_h * Gate(H_low)
```

这个更像用低阶上下文对高阶做缩放，低阶信息没有真正作为补充特征注入高阶，也没有显式建模 low/high 之间的相关性。

SDS-Net 中 ADSF 的核心思想是：

```text
shallow/deep 分别做全局描述
-> 建模两者通道相关性
-> 根据相关性生成自适应融合权重
-> 再做融合
```

因此本分支将 SH 高低阶交互改为类似 SDS-Net ADSF 的相关性驱动方式。

## 分组方式

8mic 当前 SH 最大阶数为 4，总通道数为：

```text
(4 + 1)^2 = 25
```

按阶分组：

```text
order 0: 1 个 SH 分量
order 1: 3 个 SH 分量
order 2: 5 个 SH 分量
order 3: 7 个 SH 分量
order 4: 9 个 SH 分量
```

real/imag 沿通道拼接后，每阶输入通道为：

```text
order 0: 2
order 1: 6
order 2: 10
order 3: 14
order 4: 18
```

高低阶划分：

```text
low orders  = {0, 1}
high orders = {2, 3, 4}
```

## 新的 ADFS-style HighLowMutualGuidance

位置：`networks/tfgridnetv2.py`

模块名仍为：

```text
HighLowMutualGuidance
```

但内部实现已经改为 ADFS-style correlation guidance。

### 1. 每阶独立编码

```text
B_l -> Encoder_l -> H_l
```

每个 `H_l` 的 shape：

```text
[B, hidden_dim, T, F]
```

默认：

```text
hidden_dim = emb_dim = 32
```

### 2. 融合 low/high context

```text
H_low  = Fuse(H_0, H_1)
H_high = Fuse(H_2, H_3, H_4)
```

代码中分别是：

```python
self.low_fuse
self.high_fuse
```

### 3. 做全局通道描述

```text
U_low  = GAP(H_low)
U_high = GAP(H_high)
```

shape：

```text
[B, hidden_dim, 1, 1]
```

代码中使用：

```python
F.adaptive_avg_pool2d(low_context, 1)
F.adaptive_avg_pool2d(high_context, 1)
```

### 4. 建模 low/high 相关性

```text
Corr = Refine(Desc_low * Desc_high)
```

这里的 `*` 是逐通道相乘，表示低阶和高阶在每个通道上的匹配程度。

代码中是：

```python
low_desc = self.low_descriptor(GAP(low_context))
high_desc = self.high_descriptor(GAP(high_context))
corr = self.correlation_refine(low_desc * high_desc)
```

### 5. 根据相关性生成 gate

低阶到高阶：

```text
G_h = Gate_h(Corr)
```

高阶到低阶：

```text
G_l = Gate_l(Corr)
```

注意：gate 不再只来自单边特征，而是来自 low/high 相关性。

### 6. 双向残差更新

低阶指导高阶：

```text
H_h_new = H_h + alpha_h * G_h * Proj_h(H_low)
```

高阶反哺低阶：

```text
H_l_new = H_l + alpha_l * G_l * Proj_l(H_high)
```

其中：

```text
alpha_l / alpha_h 是可学习残差缩放参数，初始为 0
```

这样训练刚开始时模型接近原来的 grouping/inter 输入，不会一开始就强行注入跨组信息。随着训练进行，模型会自己学习是否需要增强高低阶交互。

## 和旧 grouping-inter v1 的区别

| 项目 | grouping-inter v1 | grouping-inter-sds / ADFS-style |
|---|---|---|
| low/high 分组 | 有 | 有 |
| low/high context fuse | 有 | 有 |
| GAP 全局通道描述 | 无 | 有 |
| low-high 相关性建模 | 无 | 有 |
| gate 来源 | 单边 low 或 high | low/high correlation |
| 低阶注入高阶 | 主要是缩放高阶自身 | `Proj(H_low)` 注入高阶 |
| 高阶注入低阶 | 有 | 有 |
| 稳定残差 alpha | 无 | 有，初始 0 |
| 与 SDS-Net ADSF 相似度 | 较低 | 更高 |

## 完整前向流程

```text
SH waveform input
-> STFT
-> real/imag channel concat
-> order-wise SH grouping encoder
-> ADFS-style high-low correlation guidance
-> adjacent-order interaction
-> concat all order features
-> 1x1 Conv fuse 回 emb_dim
-> 原 TFGridNetV2 backbone
-> 原输出头
```

## 代码修改位置

- `networks/tfgridnetv2.py`
  - 替换 `HighLowMutualGuidance` 为 ADFS-style correlation guidance
  - 增加 `low_descriptor`、`high_descriptor`、`correlation_refine`
  - 增加 `low_to_high_projs`
  - 增加每阶 `cross_update_scale`，初始为 0

- `train.py`
  - 默认模型目录改为 `model_grouping_inter_adfs_mse_sisdr_stft_8mic`
  - 日志中显示 ADFS-style high-low correlation guidance

- `inference.py`
  - 默认读取 `model_grouping_inter_adfs_mse_sisdr_stft_8mic`
  - 推理时默认构建 ADFS-style inter 模型

- `evaluation_fixed.py`
  - profile 时默认构建 ADFS-style inter 模型

- `scripts/test_order_grouping_shape.py`
  - 检查 ADFS-style inter 输出 shape
  - 检查 `cross_update_scale` 数量和阶数一致

## 实验对比建议

建议和以下实验做对比：

```text
baseline
Group-only
Grouping + adjacent
Grouping-inter v1
Grouping-inter-sds
```

这个分支要回答的问题是：

```text
同样是高低阶交互，基于 low/high 相关性的自适应融合是否比简单单边 gate 更有效？
```


## 本版与上一版 SDS-like 的关键差异

上一版为了稳定，使用的是同通道相关：

```text
corr = low_desc * high_desc
corr shape = [B, C]
```

本版改成更完整的 ADFS-style 跨通道相关矩阵：

```text
M = outer(low_desc, high_desc)
M shape = [B, C, C]
```

其中 `M[:, i, j]` 表示低阶第 `i` 个隐藏通道与高阶第 `j` 个隐藏通道之间的相关性。随后分别沿 high/low 方向聚合：

```text
W_low  = sum_j M[:, i, j]
W_high = sum_i M[:, i, j]
```

再通过可学习因子 `beta = sigmoid(fusion_factor)` 平衡两侧权重，并生成低阶和高阶各自的 gate。这样比简单同通道乘法更接近 SDS-Net ADSF 的完整相关性建模思想。

注意：本实验仍然在统一 hidden_dim=32 的工程特征空间中做相关矩阵，而不是直接在原始 low/high SH 通道数上做非对称矩阵。这样改动更小、训练更稳，也方便和前面实验公平比较。
