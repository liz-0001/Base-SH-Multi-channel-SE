# grouping-inter 分支修改说明

## 实验目标

本分支在现有 8mic TFG-serial baseline 的 SH 输入特征编码部分加入完整的球谐阶间交互前端：

```text
order-wise SH grouping
+ high-low mutual guidance
+ adjacent-order interaction
```

主干网络、输出头、loss 计算方式、训练循环、推理输出格式和测评指标没有重写。本分支默认启用这些新增模块，训练、推理和测评命令里不需要再额外写 `--enable_order_grouping` 或 `--enable_adjacent_interaction`。

## 输入和分组

baseline 的 SH 输入通道数为 25，对应最大阶数 `N=4`：

```text
order 0: [0]                  1 个 SH 分量
order 1: [1, 2, 3]            3 个 SH 分量
order 2: [4, 5, 6, 7, 8]      5 个 SH 分量
order 3: [9 ... 15]           7 个 SH 分量
order 4: [16 ... 24]          9 个 SH 分量
```

STFT 后代码保持原来的 complex 处理方式，将 real 和 imag 沿通道维拼接。因此每阶 encoder 的实际输入通道为：

```text
order 0: 2
order 1: 6
order 2: 10
order 3: 14
order 4: 18
```

每阶特征单独进入轻量 encoder：

```text
B_l -> Conv2d -> GroupNorm -> PReLU -> H_l
```

每个 `H_l` 的 shape 都是：

```text
[B, order_hidden_dim, T, F]
```

默认 `order_hidden_dim = emb_dim = 32`。

## 新增模块一：HighLowMutualGuidance

位置：`networks/tfgridnetv2.py`

模块名：`HighLowMutualGuidance`

默认低阶和高阶划分：

```text
low orders  = {0, 1}
high orders = {2, 3, 4}
```

### 低阶指导高阶 L -> H

先融合低阶上下文：

```text
H_low = Fuse(H_0, H_1)
```

然后对每个高阶生成 gate：

```text
G_h = sigmoid(Conv2d(H_low))
```

用残差调制高阶：

```text
H_h_new = H_h + H_h * G_h
```

含义：低阶提供更稳定的全局空间结构，用来调节高阶方向细节。

### 高阶反哺低阶 H -> L

先融合高阶上下文：

```text
H_high = Fuse(H_2, H_3, H_4)
```

然后分别更新 order 0 和 order 1：

```text
H_0_new = H_0 + Gate_0(H_high) * Proj_0(H_high)
H_1_new = H_1 + Gate_1(H_high) * Proj_1(H_high)
```

含义：高阶提供更细的方向性信息，反过来补充低阶的空间表达。

## 新增模块二：AdjacentOrderInteraction

High-low guidance 后继续沿用相邻阶门控残差交互：

```text
H_l_adj = H_l
        + Gate_left_l(H_l, H_{l-1}) * Proj_left_l(H_{l-1})
        + Gate_right_l(H_l, H_{l+1}) * Proj_right_l(H_{l+1})
```

边界阶只接收存在的邻居：

```text
order 0 只接收 order 1
order 4 只接收 order 3
```

## 本分支完整执行顺序

```text
SH waveform input
-> STFT
-> real/imag channel concat
-> order-wise SH grouping encoder
-> high-low mutual guidance
-> adjacent-order interaction
-> concat all orders
-> 1x1 Conv fuse 回 emb_dim
-> 原 TFGridNetV2 backbone
-> 原输出头
```

## 代码修改位置

- `networks/tfgridnetv2.py`
  - 新增 `HighLowMutualGuidance`
  - `OrderWiseSHGroupingEncoder` 中加入 high-low guidance，并保持 adjacent interaction
  - `TFGridNetV2` 默认启用 grouping、high-low guidance 和 adjacent interaction

- `train.py`
  - 删除命令行里的模块开关参数
  - 默认启用完整 SH interaction frontend
  - 训练日志和 `run_config.json` 记录模块状态

- `inference.py`
  - 默认按 grouping-inter 模型构建网络
  - 推理命令不需要模块开关

- `evaluation_fixed.py`
  - 默认按 grouping-inter 模型做参数量和计算量统计
  - 测评命令不需要模块开关

- `scripts/test_order_grouping_shape.py`
  - 增加 high-low guidance 的 shape test
  - 检查 high-low 和 adjacent 后每阶 shape 不变

## 和前面实验的关系

```text
baseline: 不按阶建模

grouping-only: 只做按阶独立编码

grouping-adjacent: 按阶编码后，只做相邻阶交互

grouping-inter: 按阶编码后，先做高低阶互相指导，再做相邻阶交互
```

本分支主要验证：

```text
低阶全局空间结构和高阶方向细节之间的双向指导，是否能进一步提升多通道语音增强效果。
```
