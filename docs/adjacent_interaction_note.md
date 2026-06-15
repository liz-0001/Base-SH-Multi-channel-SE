# Grouping + Adjacent Interaction 实验笔记

## 1. 实验目标

本分支用于实现第二个结构消融：

```text
order-wise SH grouping + adjacent-order interaction
```

它建立在 `grouping-only` 的基础上，不改变 TFGridNetV2 主干、输出头、数据读取方式和训练流程，只在 SH 输入前端中加入相邻阶之间的轻量交互。

本实验要回答的问题是：

```text
在已经按 SH order 分组建模的基础上，只让相邻阶之间交换信息，是否能进一步提升增强效果？
```

## 2. 插入位置

原 grouping-only 流程：

```text
SH input
-> 按阶分组 B_0, B_1, ..., B_N
-> 每阶 encoder 得到 H_0, H_1, ..., H_N
-> concat
-> 1x1 fuse 到 emb_dim
-> TFGridNetV2 backbone
```

现在修改为：

```text
SH input
-> 按阶分组 B_0, B_1, ..., B_N
-> 每阶 encoder 得到 H_0, H_1, ..., H_N
-> adjacent-order interaction 得到 H_0_adj, ..., H_N_adj
-> concat
-> 1x1 fuse 到 emb_dim
-> TFGridNetV2 backbone
```

也就是说，adjacent interaction 插在：

```text
每阶 encoder 之后，concat + fuse 之前。
```

## 3. 相邻阶交互公式

对第 `l` 阶，只接收相邻阶的信息：

```text
l = 0:
    H_0_adj = H_0 + Gate_right_0(H_0, H_1) * Proj_right_0(H_1)

0 < l < N:
    H_l_adj = H_l
            + Gate_left_l(H_l, H_{l-1}) * Proj_left_l(H_{l-1})
            + Gate_right_l(H_l, H_{l+1}) * Proj_right_l(H_{l+1})

l = N:
    H_N_adj = H_N + Gate_left_N(H_N, H_{N-1}) * Proj_left_N(H_{N-1})
```

其中：

```text
Gate = sigmoid(Conv1x1(concat(H_l, H_neighbor)))
Proj = Conv1x1(H_neighbor)
```

每个 `H_l` 的 shape 为：

```text
[B, hidden_dim, T, F]
```

因此：

```text
concat(H_l, H_neighbor): [B, 2 * hidden_dim, T, F]
Gate:                    [B, hidden_dim, T, F]
Proj:                    [B, hidden_dim, T, F]
H_l_adj:                 [B, hidden_dim, T, F]
```

输出 shape 与输入 shape 完全一致。

## 4. 修改的代码位置

### 4.1 `networks/tfgridnetv2.py`

新增模块：

```python
class AdjacentOrderInteraction(nn.Module):
```

作用：

```text
输入:  [H_0, H_1, ..., H_N]
输出:  [H_0_adj, H_1_adj, ..., H_N_adj]
```

同时修改：

```python
class OrderWiseSHGroupingEncoder(nn.Module):
```

新增参数：

```python
enable_adjacent_interaction=False
```

当该参数为 `True` 时：

```python
order_features = self.adjacent_interaction(order_features)
```

然后再进行：

```python
return self.fuse(torch.cat(order_features, dim=1))
```

### 4.2 `TFGridNetV2`

新增参数：

```python
enable_adjacent_interaction=False
```

并在启用 order grouping 时传入：

```python
self.conv = OrderWiseSHGroupingEncoder(
    ...
    enable_adjacent_interaction=enable_adjacent_interaction,
)
```

默认关闭，因此不会影响 baseline 或 grouping-only。

### 4.3 `train.py`

新增训练参数：

```bash
--enable_adjacent_interaction
```

如果开启 adjacent 但忘记开启 grouping，代码会自动开启 grouping：

```python
if args.enable_adjacent_interaction and not args.enable_order_grouping:
    args.enable_order_grouping = True
```

训练日志和 `run_config.json` 中会记录：

```text
enable_order_grouping
enable_adjacent_interaction
sh_order
order_hidden_dim
```

### 4.4 `inference.py`

新增推理参数：

```bash
--enable_adjacent_interaction
```

推理 grouping+adjacent 权重时，必须同时保持模型结构一致。

### 4.5 `evaluation_fixed.py`

新增模型构造参数：

```bash
--enable_adjacent_interaction
```

用于正确加载 grouping+adjacent 权重并统计参数量、MACs/FLOPs。

### 4.6 `scripts/test_order_grouping_shape.py`

扩展 shape test：

```text
1. 检查 N=3 时 order widths 是否为 [1, 3, 5, 7]
2. 检查 grouping-only 输出 shape
3. 检查 grouping+adjacent 输出 shape
4. 检查 AdjacentOrderInteraction 中每阶更新前后 shape 不变
```

## 5. 训练命令

本分支建议继续使用和 grouping-only 相同的联合 loss，这样可以比较：

```text
grouping-only + 联合 loss
vs
grouping+adjacent + 联合 loss
```

训练命令：

```bash
python -u train.py \
  --gpus 0,1,2,3 \
  --enable_order_grouping \
  --enable_adjacent_interaction \
  --sh_order 4 \
  --loss_type mse_sisdr_stft \
  --sisdr_weight 0.01 \
  --stft_weight 0.5 \
  --num_epoch 100 \
  --batch_size 8 \
  --num_worker 4 \
  --lr 1e-3 \
  --model_dir model_grouping_adjacent_mse_sisdr_stft_8mic
```

如果显存不够，可以把 batch size 改成 4。

## 6. 推理命令

```bash
CUDA_VISIBLE_DEVICES=0 python -u inference.py \
  --modelpath model_grouping_adjacent_mse_sisdr_stft_8mic \
  --model_name model_best.pth \
  --test_name mic_8 \
  --mic_prefix mic \
  --enable_order_grouping \
  --enable_adjacent_interaction \
  --sh_order 4 \
  --prediction_path /data/lizhe/SH_data/Mic8_2s_gpurir/predictions_grouping_adjacent_mic_8
```

## 7. 测评命令

```bash
CUDA_VISIBLE_DEVICES=0 python -u evaluation_fixed.py \
  --dataset_root /data/lizhe/SH_data/Mic8_2s_gpurir \
  --prediction_path /data/lizhe/SH_data/Mic8_2s_gpurir/predictions_grouping_adjacent_mic_8 \
  --test_name mic_8 \
  --modelpath model_grouping_adjacent_mse_sisdr_stft_8mic \
  --model_name model_best.pth \
  --enable_order_grouping \
  --enable_adjacent_interaction \
  --sh_order 4 \
  --profile_model \
  --save_dir model_grouping_adjacent_mse_sisdr_stft_8mic/eval_mic_8
```

## 8. Shape Test

在服务器训练环境中运行：

```bash
python scripts/test_order_grouping_shape.py
```

期望看到：

```text
Order grouping shape test passed.
order widths: [1, 3, 5, 7]
input shape : (2, 32, 9, 17)
output shape: (2, 32, 9, 17)
adjacent output shape: (2, 32, 9, 17)
```

## 9. 消融对比方式

推荐比较：

| Model | Loss | Grouping | Adjacent |
|---|---|---:|---:|
| TFG-serial | MSE | No | No |
| TFG-serial | MSE+SI-SDR+STFT | No | No |
| Grouping-only | MSE+SI-SDR+STFT | Yes | No |
| Grouping+Adjacent | MSE+SI-SDR+STFT | Yes | Yes |

这样可以说明：

```text
在相同联合 loss 下，相邻阶交互是否相比 grouping-only 带来进一步提升。
```

