# -*- coding: utf-8 -*-
"""
TFG / IGCRN style training script

功能：
1. 读取 train / val 数据
2. 训练模型
3. 使用 tqdm 显示训练/验证进度，并保持官方化日志输出
4. 自动保存每个 epoch 模型与最佳模型
"""

import os
import gc
import time
import random
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from loader.IGCRN_dataloader import make_fix_loader
from networks.tfgridnetv2 import TFGridNetV2
# 地址复用 本地：/data/lizhe/SH_data 服务器：/autodl-tmp
from pathlib import Path
DATA_ROOT = Path("/autodl-tmp")

warnings.filterwarnings("ignore")

# =========================
# 1. 固定随机种子，保证复现性
# =========================
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# =========================
# 1.2. 导入自定义模块切换训练模式
#==========================
def set_trainable(model, module_name="adfs_optimizer", trainable=False):
    # 检查是否使用了 DataParallel
    real_model = model.module if hasattr(model, 'module') else model
    
    for name, param in real_model.named_parameters():
        if module_name in name:
            param.requires_grad = trainable
            # 打印一下确认状态（可选）
            # print(f"Setting {name} trainable={trainable}")
# =========================
# 2. 参数设置
# =========================
parser = argparse.ArgumentParser("TFGridNetV2 training")

# -------- 训练集路径 --------
parser.add_argument(
    "--train_wav_scp",
    type=str,
    default=DATA_ROOT / "Mic8_2s_gpurir/loader_txt/wav_scp/wav_scp_train.txt",
    help="训练集 wav_scp 列表"
)
parser.add_argument(
    "--train_mix_dir",
    type=str,
    default=DATA_ROOT / "Mic8_2s_gpurir/generated_data/train/mix",
    help="训练集 mix 目录"
)
parser.add_argument(
    "--train_ref_dir",
    type=str,
    default=DATA_ROOT / "Mic8_2s_gpurir/generated_data/train/noreverb_ref",
    help="训练集参考干净语音目录"
)
parser.add_argument(
    "--train_mic_dir",
    type=str,
    default=DATA_ROOT / "Mic8_2s_gpurir/RIR/cir_uniform_8/train_val_rir/MIC",
    help="训练集麦克风阵列信息目录"
)

# -------- 验证集路径 --------
parser.add_argument(
    "--val_wav_scp",
    type=str,
    default=DATA_ROOT / "Mic8_2s_gpurir/loader_txt/wav_scp/wav_scp_val.txt",
    help="验证集 wav_scp 列表"
)
parser.add_argument(
    "--val_mix_dir",
    type=str,
    default=DATA_ROOT / "Mic8_2s_gpurir/generated_data/val/mix",
    help="验证集 mix 目录"
)
parser.add_argument(
    "--val_ref_dir",
    type=str,
    default=DATA_ROOT / "Mic8_2s_gpurir/generated_data/val/noreverb_ref",
    help="验证集参考干净语音目录"
)
parser.add_argument(
    "--val_mic_dir",
    type=str,
    default=DATA_ROOT / "Mic8_2s_gpurir/RIR/cir_uniform_8/train_val_rir/MIC",
    help="验证集麦克风阵列信息目录"
)

# -------- 训练超参数 --------
parser.add_argument("--gpuid", type=int, default=0, help="使用哪张 GPU")
parser.add_argument("--num_epoch", type=int, default=100, help="训练轮数")
parser.add_argument("--num_worker", type=int, default=4, help="DataLoader worker 数")
parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--fft_len", type=int, default=512)
parser.add_argument("--channel", type=int, default=8)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--chunk", type=int, default=2)
parser.add_argument("--sample_rate", type=int, default=16000)
parser.add_argument("--resume", action="store_true", help="是否从已有模型继续训练")
parser.add_argument(
    "--resume_model",
    type=str,
    default="model_test",
    help="继续训练时加载的模型路径"
)

args = parser.parse_args()

# =========================
# 3. 设备设置
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NowTime = time.localtime()


def log_info(msg: str):
    """统一日志输出格式"""
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{now}] {msg}", flush=True)


def count_parameters(network):
    """统计模型可训练参数量"""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def check_path_exists(path_name, path_value):
    """训练开始前检查路径是否存在，避免跑到一半才报错"""
    if not os.path.exists(path_value):
        raise FileNotFoundError(f"{path_name} 不存在: {path_value}")


if __name__ == "__main__":
    log_info("Training start")
    log_info(f"Using device: {device}")

    # =========================
    # 4. 检查关键路径
    # =========================
    check_path_exists("train_wav_scp", args.train_wav_scp)
    check_path_exists("train_mix_dir", args.train_mix_dir)
    check_path_exists("train_ref_dir", args.train_ref_dir)
    check_path_exists("train_mic_dir", args.train_mic_dir)

    check_path_exists("val_wav_scp", args.val_wav_scp)
    check_path_exists("val_mix_dir", args.val_mix_dir)
    check_path_exists("val_ref_dir", args.val_ref_dir)
    check_path_exists("val_mic_dir", args.val_mic_dir)

    # =========================
    # 5. 读取参数
    # =========================
    iter_count = 0
    batch_size = args.batch_size
    num_worker = args.num_worker
    fft_len = args.fft_len
    repeat = args.repeat
    chunk = args.chunk
    sample_rate = args.sample_rate

    train_wav_scp = args.train_wav_scp
    train_mix_dir = args.train_mix_dir
    train_ref_dir = args.train_ref_dir
    train_mic_dir = args.train_mic_dir

    val_wav_scp = args.val_wav_scp
    val_mix_dir = args.val_mix_dir
    val_ref_dir = args.val_ref_dir
    val_mic_dir = args.val_mic_dir

    resume = args.resume

    # =========================
    # 6. 构建模型
    # =========================
    network = TFGridNetV2(
        input_dim=None,
        n_srcs=1,
        n_fft=fft_len,
        stride=256,
        n_imics=25,
        n_layers=3,
        lstm_hidden_units=128,
        attn_approx_qk_dim=256,
        emb_dim=32
    )

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        log_info(f"Use {torch.cuda.device_count()} GPUs for DataParallel")
        network = torch.nn.DataParallel(network)

    log_info(f"Trainable parameters: {count_parameters(network):,}")

    if resume:
        log_info(f"Resume training from: {args.resume_model}")
        state_dict = torch.load(args.resume_model, map_location=device)
        network.load_state_dict(state_dict)

    network = network.to(device)
    # --- 在主循环开始前的临时测试代码 ---
    log_info("--- Running ADFS Integration Check ---")

    # 1. 检查 ADFS 是否在网络中
    if hasattr(network, "adfs_optimizer") or (hasattr(network, "module") and hasattr(network.module, "adfs_optimizer")):
        log_info("Success: ADFS module found in the network.")
    else:
        log_info("Error: ADFS module NOT found!")

    # 2. 检查参数冻结状态 (假设现在是第 1 个 Epoch)
    for name, param in network.named_parameters():
        if "adfs_optimizer" in name:
            log_info(f"Param: {name} | Requires_Grad: {param.requires_grad}")

    log_info("--- Check Done ---")    

    # =========================
    # 7. 优化器、损失函数、日志
    # =========================
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, network.parameters()), 
        lr=args.lr
    )
    loss_function = nn.MSELoss()

    writer = SummaryWriter(
        "runs/Fine_tuning_{}/".format(time.strftime("%Y-%m-%d-%H-%M-%S", NowTime))
    )

    modelpath = "model_test/"
    os.makedirs(modelpath, exist_ok=True)

    loss_train_epoch = []
    loss_val_epoch = []

    min_val_loss = float("inf")
    val_no_impv = 0

    # =========================
    # 8. 开始训练
    # =========================
    for epoch in range(args.num_epoch):
        epoch_id = epoch + 1
        # === 新增：分阶段训练逻辑 ===
        # 假设前 10 个 Epoch 冻结 ADFS，第 11 个 Epoch 开始解冻训练
        if epoch_id <= 10:
            # 冻结 ADFS
            set_trainable(network, module_name="adfs_optimizer", trainable=False)
            if epoch_id == 1:
                log_info("Stage 1: ADFS is frozen. Training backbone only.")
        else:
            # 解冻 ADFS
            set_trainable(network, module_name="adfs_optimizer", trainable=True)
            if epoch_id == 11:
                log_info("Stage 2: ADFS is unfrozen. Training all modules.")
                # 注意：如果解冻了新参数，有些情况下需要重新把新参数告知优化器
                # 但如果你想简单处理，可以在这里重新定义一次 optimizer (可选)
                # optimizer = optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=args.lr)
        # =========================

        train_loader = make_fix_loader(
            wav_scp=train_wav_scp,
            mix_dir=train_mix_dir,
            ref_dir=train_ref_dir,
            mic_dir=train_mic_dir,
            batch_size=batch_size,
            repeat=repeat,
            num_workers=num_worker,
            chunk=chunk,
            sample_rate=sample_rate,
        )

        val_loader = make_fix_loader(
            wav_scp=val_wav_scp,
            mix_dir=val_mix_dir,
            ref_dir=val_ref_dir,
            mic_dir=val_mic_dir,
            batch_size=batch_size,
            repeat=repeat,
            num_workers=num_worker,
            chunk=chunk,
            sample_rate=sample_rate,
        )

        # =========================
        # 8.1 训练阶段
        # =========================
        log_info(f"Epoch {epoch_id}/{args.num_epoch} | Train start | batches={len(train_loader)}")

        network.train()
        train_loss_sum = 0.0
        train_step_count = 0
        train_start_time = time.time()

        train_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Train {epoch_id}/{args.num_epoch}",
            dynamic_ncols=True,
            leave=True
        )

        for idx, egs in train_bar:
            stft_input = egs["stft_input"][:, 0, :].unsqueeze(1).to(device, non_blocking=True)
            shc_input = egs["shc_input"].to(device, non_blocking=True)
            target = egs["target"][:, 0, :].to(device, non_blocking=True)

            ilens = shc_input.size(2) * torch.ones(
                shc_input.size(0), dtype=torch.int64, device=device
            )

            optimizer.zero_grad()

            outputs = network(shc_input.transpose(1, 2), ilens)
            loss = loss_function(outputs[0][0], target)

            loss.backward()
            # 在 train.py 的 optimizer.step() 之前加入
            if epoch_id > 10:  # 假设 10 轮后解冻
                for name, param in network.named_parameters():
                    if "adfs_optimizer" in name and param.grad is not None:
                        # 如果能打印出大于 0 的数值，说明 ADFS 正在学习
                        print(f"ADFS Grad Norm: {param.grad.norm().item()}")
            optimizer.step()

            loss_item = loss.item()
            train_loss_sum += loss_item
            train_step_count += 1

            writer.add_scalars("Loss", {"Train": loss_item}, iter_count)
            iter_count += 1

            if idx % 100 == 0:
                train_bar.set_postfix(
                    loss=f"{loss_item:.6f}",
                    avg=f"{train_loss_sum / train_step_count:.6f}"
                )

            del stft_input, shc_input, target, ilens, outputs, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        epoch_train_loss = train_loss_sum / max(train_step_count, 1)
        loss_train_epoch.append(epoch_train_loss)
        train_time = time.time() - train_start_time

        log_info(
            f"Epoch {epoch_id}/{args.num_epoch} | Train done | "
            f"train_loss={epoch_train_loss:.6f} | time={train_time/60:.2f} min"
        )

        # =========================
        # 8.2 验证阶段
        # =========================
        log_info(f"Epoch {epoch_id}/{args.num_epoch} | Val start   | batches={len(val_loader)}")

        network.eval()
        val_loss_sum = 0.0
        val_step_count = 0
        val_start_time = time.time()

        val_bar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Val   {epoch_id}/{args.num_epoch}",
            dynamic_ncols=True,
            leave=True
        )

        with torch.no_grad():
            for idx, egs in val_bar:
                stft_input = egs["stft_input"][:, 0, :].unsqueeze(1).to(device, non_blocking=True)
                shc_input = egs["shc_input"].to(device, non_blocking=True)
                target = egs["target"][:, 0, :].to(device, non_blocking=True)

                ilens = shc_input.size(2) * torch.ones(
                    shc_input.size(0), dtype=torch.int64, device=device
                )

                outputs = network(shc_input.transpose(1, 2), ilens)
                loss_val = loss_function(outputs[0][0], target)

                loss_val_item = loss_val.item()
                val_loss_sum += loss_val_item
                val_step_count += 1

                if idx % 100 == 0:
                    val_bar.set_postfix(
                        loss=f"{loss_val_item:.6f}",
                        avg=f"{val_loss_sum / val_step_count:.6f}"
                    )

                del stft_input, shc_input, target, ilens, outputs, loss_val
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        epoch_val_loss = val_loss_sum / max(val_step_count, 1)
        loss_val_epoch.append(epoch_val_loss)
        val_time = time.time() - val_start_time

        log_info(
            f"Epoch {epoch_id}/{args.num_epoch} | Val done   | "
            f"val_loss={epoch_val_loss:.6f} | time={val_time/60:.2f} min"
        )

        writer.add_scalars(
            "Epoch_Loss",
            {
                "Train_Epoch": adepoch_train_loss,
                "Validation_Epoch": epoch_val_loss
            },
            epoch_id
        )

        # =========================
        # 8.3 保存模型
        # =========================
        torch.save(network.state_dict(), os.path.join(modelpath, f"network_epoch{epoch_id}.pth"))

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch_val_loss <= min_val_loss:
            min_val_loss = epoch_val_loss
            val_no_impv = 0
            torch.save(network.state_dict(), os.path.join(modelpath, "model_best.pth"))
            log_info(
                f"Epoch {epoch_id}/{args.num_epoch} | "
                f"lr={current_lr:.8f} | best_val={epoch_val_loss:.6f} | saved=model_best.pth"
            )
        else:
            val_no_impv += 1

            optim_state = optimizer.state_dict()
            optim_state["param_groups"][0]["lr"] = optim_state["param_groups"][0]["lr"] / 2.0
            optimizer.load_state_dict(optim_state)

            new_lr = optimizer.param_groups[0]["lr"]
            log_info(
                f"Epoch {epoch_id}/{args.num_epoch} | "
                f"val not improved | lr adjusted to {new_lr:.8f} | no_impv={val_no_impv}"
            )

            if val_no_impv >= 5:
                log_info("No improvements for 5 epochs, early stopping.")
                break

        # =========================
        # 8.4 保存 loss 曲线
        # =========================
        np.save(os.path.join(modelpath, "loss_val_epoch.npy"), loss_val_epoch)
        np.save(os.path.join(modelpath, "loss_train_epoch.npy"), loss_train_epoch)

        plt.figure(figsize=(8, 6))
        plt.title("MSE Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.plot(loss_train_epoch, label="train_loss")
        plt.plot(loss_val_epoch, label="val_loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(modelpath, "Network_loss.png"))
        plt.close()

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    writer.close()
    log_info("Training finished.")
