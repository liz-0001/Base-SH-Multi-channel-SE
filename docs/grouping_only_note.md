# Grouping-only 实验笔记

## 1. 实验目的

本次修改的目标是在现有球谐域多通道语音增强 baseline 上加入 **order-wise SH grouping**。

baseline 原始输入已经是球谐域特征，包含 0 阶到 4 阶的 SH coefficients，总 SH 通道数为：

```text
(4 + 1)^2 = 25
```

原 baseline 在 STFT 后会把 real 和 imag 沿通道维拼接：

```text
[real SH channels, imag SH channels]
```

然后直接使用一个普通 `Conv2d` 将 `2 * n_imics` 个通道映射到 `emb_dim`，再送入 TFGridNetV2 主干。

本次 grouping-only 修改的核心思想是：

```text
先按照 SH order 对通道进行分组；
每一阶单独经过一个轻量 encoder；
最后将各阶表示 concat 后投影回 emb_dim；
后续 TFGridNetV2 主干、loss、训练流程和输出头保持不变。
```

这样可以验证一个问题：

```text
显式按 SH 阶数建模，是否比 baseline 直接混合所有 SH 通道更适合球谐域多通道语音增强？
```

## 2. SH order 分组方式

对于最大阶数 `N`，第 `l` 阶 SH 分量数量为：

```text
2l + 1
```

第 `l` 阶对应的通道索引为：

```text
start = l^2
end   = (l + 1)^2 - 1
```

当前 8mic baseline 使用 4 阶 SH，总通道数为 25，具体分组为：

| Order | SH indices | 通道数 |
|---|---|---:|
| 0 | `[0]` | 1 |
| 1 | `[1, 2, 3]` | 3 |
| 2 | `[4, 5, 6, 7, 8]` | 5 |
| 3 | `[9, 10, 11, 12, 13, 14, 15]` | 7 |
| 4 | `[16, 17, 18, 19, 20, 21, 22, 23, 24]` | 9 |

由于当前模型在 STFT 后使用 real/imag 通道拼接，因此每阶实际输入 encoder 的通道数为：

| Order | Real 通道数 | Imag 通道数 | Encoder 输入通道数 |
|---|---:|---:|---:|
| 0 | 1 | 1 | 2 |
| 1 | 3 | 3 | 6 |
| 2 | 5 | 5 | 10 |
| 3 | 7 | 7 | 14 |
| 4 | 9 | 9 | 18 |

## 3. 修改的代码位置

### 3.1 `networks/tfgridnetv2.py`

新增模块：

```python
OrderWiseSHGroupingEncoder
```

该模块完成：

```text
1. 检查 n_imics 是否等于 (N + 1)^2；
2. 根据 order 构造通道切片；
3. 将 real 和 imag 使用同样的 order index 分组；
4. 每一阶经过单独 encoder；
5. concat 所有 order features；
6. 使用 1x1 Conv 融合并投影回 emb_dim。
```

每阶 encoder 的结构为：

```text
Conv2d -> GroupNorm -> PReLU
```

最后融合结构为：

```text
Conv2d(1x1) -> GroupNorm
```

同时在 `TFGridNetV2` 中新增参数：

```python
enable_order_grouping=False
sh_order=None
order_hidden_dim=None
```

默认情况下：

```python
enable_order_grouping=False
```

模型仍然使用原 baseline 输入投影：

```python
Conv2d(2 * n_imics, emb_dim)
```

开启 grouping-only 时，才替换为：

```python
OrderWiseSHGroupingEncoder(...)
```

因此 baseline 路径仍然兼容。

### 3.2 `train.py`

新增训练参数：

```bash
--enable_order_grouping
--sh_order
--order_hidden_dim
```

其中：

```text
--enable_order_grouping  开启 order-wise SH grouping
--sh_order               SH 最大阶数，当前默认 4
--order_hidden_dim       每个 order encoder 的 hidden dim，默认等于 emb_dim
```

训练日志和 `run_config.json` 会记录这些配置，方便后续整理实验表格。

### 3.3 `inference.py`

新增推理参数：

```bash
--enable_order_grouping
--sh_order
--order_hidden_dim
```

注意：如果推理 grouping-only 训练出的权重，必须加：

```bash
--enable_order_grouping
```

否则模型结构和权重不匹配。

### 3.4 `scripts/test_order_grouping_shape.py`

新增 shape test，用于检查：

```text
1. N = 3 时，分组结果是否为 [1, 3, 5, 7]；
2. 每阶 encoder 输出后 shape 是否不改变时间频率维度；
3. concat + fuse 后输出能否回到 [B, emb_dim, T, F]。
```

## 4. 先运行 shape test

在服务器训练环境中进入项目目录：

```bash
cd ~/lizhe/SH_injection-main/TFG
```

运行：

```bash
python scripts/test_order_grouping_shape.py
```

期望输出：

```text
Order grouping shape test passed.
order widths: [1, 3, 5, 7]
input shape : (2, 32, 9, 17)
output shape: (2, 32, 9, 17)
```

如果该测试通过，说明 order index 和输出 shape 没问题。

## 5. Grouping-only 训练指令

当前实验仍然使用 baseline 的训练设置，只额外开启 order grouping。

推荐训练命令：这是默认mse的跑法

```bash
python -u train.py \
  --gpus 0,1,2,3 \
  --enable_order_grouping \
  --sh_order 4 \
  --num_epoch 100 \
  --batch_size 8 \
  --num_worker 8 \
  --lr 1e-3 \
  --model_dir model_grouping_only_8mic
```

这是组合loss的跑法
```bash
python -u train.py \
  --gpus 0,1,2,3 \
  --enable_order_grouping \
  --sh_order 4 \
  --loss_type mse_sisdr_stft \
  --sisdr_weight 0.01 \
  --stft_weight 0.5 \
  --num_epoch 100 \
  --batch_size 8 \
  --num_worker 4 \
  --lr 1e-3 \
  --model_dir model_grouping_only_mse_sisdr_stft_8mic
```

如果显存不足，可以先将 batch size 降到 4：

```bash
python -u train.py \
  --gpus 0,1,2,3 \
  --enable_order_grouping \
  --sh_order 4 \
  --num_epoch 100 \
  --batch_size 4 \
  --num_worker 4 \
  --lr 1e-3 \
  --model_dir model_grouping_only_8mic
```

训练结果会保存到：

```text
model_grouping_only_8mic/
```

主要关注：

```text
model_grouping_only_8mic/model_best.pth
model_grouping_only_8mic/run_config.json
model_grouping_only_8mic/epoch_metrics.csv
logs/对应时间戳/train.log
```

## 6. Grouping-only 推理指令

推理 grouping-only 模型时，需要和训练时保持同样的模型结构开关：

```bash
CUDA_VISIBLE_DEVICES=0 python -u inference.py \
  --modelpath model_grouping_only_8mic \
  --model_name model_best.pth \
  --test_name mic_8 \
  --mic_prefix mic \
  --enable_order_grouping \
  --sh_order 4 \
  --profile_model
```

推理输出包括：

```text
1. PESQ/STOI 平均结果；
2. 增强后的 wav 文件；
3. 参数量和计算量统计；
4. metrics mat 文件。
```

默认增强语音保存位置：

```text
/data/lizhe/SH_data/Mic8_2s_gpurir/predictions_tfg_serial_test_mic_8
```

平均指标保存位置：

```text
model_grouping_only_8mic/result_model_best/mic_8_metrics.mat
```

## 7. 和 baseline 对比时需要注意

为了公平比较，建议保持以下设置一致：

```text
数据集一致；
训练轮数一致；
batch size 一致；
学习率一致；
loss 一致；
推理 test set 一致；
评价脚本一致。
```

实验表中可以记录：

| Model | Grouping | Adjacent | High-low | PESQ | STOI | Params | MACs |
|---|---:|---:|---:|---:|---:|---:|---:|
| TFG-serial baseline | No | No | No | - | - | - | - |
| Grouping-only | Yes | No | No | - | - | - | - |

## 8. 本次实验结论目标

本次 grouping-only 实验主要回答：

```text
只加入 SH order-wise grouping，不加入 adjacent interaction 和 high-low guidance，
是否能提升 8mic TFG-serial baseline 的 PESQ/STOI？
```

如果 grouping-only 有提升，说明：

```text
SH 按阶建模这个物理先验本身有效。
```

如果提升不明显，也可以继续观察：

```text
1. adjacent-order interaction 是否能补充阶间信息；
2. high-low mutual guidance 是否能让低阶全局信息和高阶细节信息互相增强。
```
