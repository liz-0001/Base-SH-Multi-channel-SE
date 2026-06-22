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
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
#数据
from loader.IGCRN_dataloader import make_fix_loader
from networks.tfgridnetv2 import TFGridNetV2
# 地址复用 本地：/data/lizhe/SH_data 服务器：/root/autodl-tmp/SH_data
from pathlib import Path
DATA_ROOT = Path("/root/autodl-tmp/SH_data")

warnings.filterwarnings("ignore")

import sys
import logging




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
parser.add_argument("--gpuid", type=int, default=0, help="兼容旧参数；单卡时可用")
parser.add_argument(
    "--gpus",
    type=str,
    default="0,1,2,3",
    help="可见 GPU 编号，例如 0 或 0,1,2,3；空字符串表示使用当前环境设置",
)
parser.add_argument("--num_epoch", type=int, default=100, help="训练轮数")
parser.add_argument("--num_worker", type=int, default=4, help="DataLoader worker 数")
parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
parser.add_argument("--batch_size", type=int, default=8, help="global batch size")
parser.add_argument("--log_interval", type=int, default=20, help="进度条刷新 loss 的 step 间隔")
parser.add_argument("--disable_tqdm", action="store_true", help="关闭控制台进度条")
parser.add_argument(
    "--loss_type",
    type=str,
    default="mse",
    choices=["mse", "mse_sisdr", "mse_stft", "mse_sisdr_stft"],
    help="训练损失类型；论文 baseline 用 mse",
)
parser.add_argument("--sisdr_weight", type=float, default=0.01, help="SI-SDR loss 权重")
parser.add_argument("--stft_weight", type=float, default=0.5, help="STFT loss 权重")
parser.add_argument("--stft_loss_fft", type=int, default=512, help="STFT loss 的 FFT 长度")
parser.add_argument("--stft_loss_hop", type=int, default=256, help="STFT loss 的 hop 长度")
parser.add_argument("--fft_len", type=int, default=512)
parser.add_argument("--channel", type=int, default=8)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--chunk", type=int, default=2)
parser.add_argument("--sample_rate", type=int, default=16000)
parser.add_argument("--sh_order", type=int, default=4, help="SH 最大阶数；8mic baseline 当前为 4 阶，通道数 25")
parser.add_argument(
    "--order_hidden_dim",
    type=int,
    default=None,
    help="每个 SH 阶 encoder 的 hidden dim；默认等于 emb_dim",
)
parser.add_argument("--resume", action="store_true", help="是否从已有模型继续训练")
parser.add_argument(
    "--resume_model",
    type=str,
    default="model_tfg_serial_8mic/model_best.pth",
    help="继续训练时加载的模型路径"
)
parser.add_argument(
    "--model_dir",
    type=str,
    default="model_grouping_inter_adfs_mse_sisdr_stft_8mic",
    help="模型保存目录"
)

args = parser.parse_args()

# grouping-inter-sds branch: the full SH interaction frontend is enabled by default.
args.enable_order_grouping = True
args.enable_high_low_guidance = True
args.enable_low_to_high = True
args.enable_high_to_low = True
args.enable_adjacent_interaction = True

if args.gpus.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()

# =========================
# 3. 设备设置
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NowTime = time.localtime()


class TqdmLoggingHandler(logging.Handler):
    """避免 logging 打乱 tqdm 进度条"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logger(log_file):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = None


def log_info(msg: str):
    """同时输出到终端和干净的 train.log"""
    if logger is not None:
        logger.info(msg)
    else:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{now}] {msg}", flush=True)



def count_parameters(network):
    """统计模型可训练参数量"""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def check_path_exists(path_name, path_value):
    """训练开始前检查路径是否存在，避免跑到一半才报错"""
    if not os.path.exists(path_value):
        raise FileNotFoundError(f"{path_name} 不存在: {path_value}")


class EnhancementLoss(nn.Module):
    def __init__(
        self,
        loss_type="mse",
        sisdr_weight=0.01,
        stft_weight=0.5,
        stft_fft=512,
        stft_hop=256,
        eps=1.0e-8,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.sisdr_weight = sisdr_weight
        self.stft_weight = stft_weight
        self.stft_fft = stft_fft
        self.stft_hop = stft_hop
        self.eps = eps
        self.mse = nn.MSELoss()
        self.register_buffer("window", torch.hann_window(stft_fft), persistent=False)

    def forward(self, estimate, target):
        estimate = estimate.float()
        target = target.float()

        mse_loss = self.mse(estimate, target)
        sisdr_loss = estimate.new_tensor(0.0)
        stft_loss = estimate.new_tensor(0.0)

        total_loss = mse_loss
        if "sisdr" in self.loss_type:
            sisdr_loss = self.si_sdr_loss(estimate, target)
            total_loss = total_loss + self.sisdr_weight * sisdr_loss
        if "stft" in self.loss_type:
            stft_loss = self.log_stft_mag_loss(estimate, target)
            total_loss = total_loss + self.stft_weight * stft_loss

        stats = {
            "total": total_loss.detach(),
            "mse": mse_loss.detach(),
            "sisdr": sisdr_loss.detach(),
            "stft": stft_loss.detach(),
        }
        return total_loss, stats

    def si_sdr_loss(self, estimate, target):
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)

        dot = torch.sum(estimate * target, dim=-1, keepdim=True)
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        projection = dot * target / target_energy
        noise = estimate - projection

        ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + self.eps)
        si_sdr = 10 * torch.log10(ratio + self.eps)
        return -torch.mean(si_sdr)

    def log_stft_mag_loss(self, estimate, target):
        estimate_spec = torch.stft(
            estimate,
            n_fft=self.stft_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_fft,
            window=self.window.to(estimate.device),
            return_complex=True,
        )
        target_spec = torch.stft(
            target,
            n_fft=self.stft_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_fft,
            window=self.window.to(target.device),
            return_complex=True,
        )
        estimate_mag = torch.abs(estimate_spec)
        target_mag = torch.abs(target_spec)
        return F.l1_loss(torch.log1p(estimate_mag), torch.log1p(target_mag))


def log_training_config():
    rows = [
        ("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")),
        ("cuda_available", str(torch.cuda.is_available())),
        ("visible_gpu_count", str(torch.cuda.device_count())),
        ("model", "TFGridNetV2 serial"),
        ("mic", "8"),
        ("sh_order", str(args.sh_order)),
        ("sh_channels", "25"),
        ("enable_order_grouping", str(args.enable_order_grouping)),
        ("enable_high_low_guidance", str(args.enable_high_low_guidance)),
        ("enable_low_to_high", str(args.enable_low_to_high)),
        ("enable_high_to_low", str(args.enable_high_to_low)),
        ("enable_adjacent_interaction", str(args.enable_adjacent_interaction)),
        ("order_hidden_dim", str(args.order_hidden_dim)),
        ("epochs", str(args.num_epoch)),
        ("batch_size", str(args.batch_size)),
        ("num_worker", str(args.num_worker)),
        ("lr", str(args.lr)),
        ("fft_len", str(args.fft_len)),
        ("stride", "256"),
        ("chunk_seconds", str(args.chunk)),
        ("sample_rate", str(args.sample_rate)),
        ("optimizer", "Adam"),
        ("loss_type", args.loss_type),
        ("sisdr_weight", str(args.sisdr_weight)),
        ("stft_weight", str(args.stft_weight)),
        ("stft_loss_fft", str(args.stft_loss_fft)),
        ("stft_loss_hop", str(args.stft_loss_hop)),
        ("model_dir", str(args.model_dir)),
    ]
    log_info("========== Training Config ==========")
    for key, value in rows:
        log_info(f"{key:<22}: {value}")
    if torch.cuda.is_available():
        for gpu_idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_idx)
            mem_gb = props.total_memory / 1024 ** 3
            log_info(f"{'gpu_' + str(gpu_idx):<22}: {props.name} | {mem_gb:.1f} GB")
    log_info("---------- Data Paths ----------")
    for key in [
        "train_wav_scp",
        "train_mix_dir",
        "train_ref_dir",
        "train_mic_dir",
        "val_wav_scp",
        "val_mix_dir",
        "val_ref_dir",
        "val_mic_dir",
    ]:
        log_info(f"{key:<22}: {getattr(args, key)}")
    log_info("=====================================")


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def make_progress_bar(iterable, total, desc):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=True,
        mininterval=1,
        file=sys.stderr,
        disable=args.disable_tqdm,
    )


def save_run_config(log_dir, modelpath):
    config = {key: str(value) for key, value in vars(args).items()}
    config.update(
        {
            "seed": SEED,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "visible_gpu_count": torch.cuda.device_count(),
            "model": "TFGridNetV2 serial",
            "n_imics": 25,
            "enable_order_grouping": args.enable_order_grouping,
            "enable_high_low_guidance": args.enable_high_low_guidance,
            "enable_low_to_high": args.enable_low_to_high,
            "enable_high_to_low": args.enable_high_to_low,
            "enable_adjacent_interaction": args.enable_adjacent_interaction,
            "sh_order": args.sh_order,
            "order_hidden_dim": args.order_hidden_dim,
            "n_layers": 3,
            "lstm_hidden_units": 128,
            "attn_approx_qk_dim": 256,
            "emb_dim": 32,
            "optimizer": "Adam",
            "loss_type": args.loss_type,
            "sisdr_weight": args.sisdr_weight,
            "stft_weight": args.stft_weight,
            "stft_loss_fft": args.stft_loss_fft,
            "stft_loss_hop": args.stft_loss_hop,
        }
    )
    for out_dir in [log_dir, Path(modelpath)]:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def append_epoch_metrics(csv_path, row):
    is_new_file = not os.path.exists(csv_path)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "lr",
        "best_val_loss",
        "train_minutes",
        "val_minutes",
        "train_steps",
        "val_steps",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer_csv.writeheader()
        writer_csv.writerow(row)


if __name__ == "__main__":
    exp_name = time.strftime("TFG_%Y%m%d_%H%M%S", time.localtime())

    log_dir = Path("logs") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    global_logger_file = log_dir / "train.log"
    logger = setup_logger(global_logger_file)

    log_info(f"Log file: {global_logger_file}")

    log_info("Training start")
    log_info(f"Using device: {device}")
    log_training_config()

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
        n_imics=25,#球谐通道纬度
        n_layers=3,
        lstm_hidden_units=128,
        attn_approx_qk_dim=256,
        emb_dim=32,
        enable_order_grouping=args.enable_order_grouping,
        sh_order=args.sh_order,
        order_hidden_dim=args.order_hidden_dim,
        enable_adjacent_interaction=args.enable_adjacent_interaction,
        enable_high_low_guidance=args.enable_high_low_guidance,
        enable_low_to_high=args.enable_low_to_high,
        enable_high_to_low=args.enable_high_to_low,
    )
    log_info(
        "Model: TFGridNetV2 serial + order-wise SH grouping + ADFS-style high-low correlation guidance "
        "+ adjacent-order interaction "
        f"(sh_order={args.sh_order}, order_hidden_dim={args.order_hidden_dim or 32})"
    )

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        log_info(f"Use {torch.cuda.device_count()} GPUs for DataParallel")
        network = torch.nn.DataParallel(network)

    log_info(f"Trainable parameters: {count_parameters(network):,}")

    if resume:
        log_info(f"Resume training from: {args.resume_model}")
        state_dict = torch.load(args.resume_model, map_location=device)
        network.load_state_dict(state_dict, strict=False)
        log_info("Successfully loaded checkpoint.")
    network = network.to(device)

    # =========================
    # 7. 优化器、损失函数、日志
    # =========================
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, network.parameters()), 
        lr=args.lr
    )
    loss_function = EnhancementLoss(
        loss_type=args.loss_type,
        sisdr_weight=args.sisdr_weight,
        stft_weight=args.stft_weight,
        stft_fft=args.stft_loss_fft,
        stft_hop=args.stft_loss_hop,
    ).to(device)
    log_info(
        f"Loss function: {args.loss_type} | "
        f"sisdr_weight={args.sisdr_weight} | stft_weight={args.stft_weight}"
    )

    writer = SummaryWriter(
        "runs/Fine_tuning_{}/".format(time.strftime("%Y-%m-%d-%H-%M-%S", NowTime))
    )

    modelpath = args.model_dir
    os.makedirs(modelpath, exist_ok=True)
    log_info(f"Model save dir: {modelpath}")
    save_run_config(log_dir, modelpath)
    log_info(f"Run config saved to: {log_dir / 'run_config.json'}")
    log_info(f"Run config saved to: {Path(modelpath) / 'run_config.json'}")

    loss_train_epoch = []
    loss_val_epoch = []

    min_val_loss = float("inf")
    val_no_impv = 0

    # =========================
    # 8. 开始训练
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

    log_info("========== Dataset Summary ==========")
    log_info(f"train utterances={len(train_loader.dataset)} | train batches={len(train_loader)}")
    log_info(f"val utterances={len(val_loader.dataset)} | val batches={len(val_loader)}")
    log_info("=====================================")

    for epoch in range(args.num_epoch):
        epoch_id = epoch + 1

        # =========================
        # 8.1 训练阶段
        # =========================
        log_info(
            f"Epoch {epoch_id}/{args.num_epoch} | Train start | "
            f"batches={len(train_loader)} | lr={current_lr(optimizer):.8f}"
        )

        network.train()
        train_loss_sum = 0.0
        train_step_count = 0
        train_start_time = time.time()

        train_bar = make_progress_bar(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Train {epoch_id}/{args.num_epoch}",
        )

        for idx, egs in train_bar:
            shc_input = egs["shc_input"].to(device, non_blocking=True)
            target = egs["target"][:, 0, :].to(device, non_blocking=True)

            ilens = shc_input.size(2) * torch.ones(
                shc_input.size(0), dtype=torch.int64, device=device
            )

            optimizer.zero_grad()

            outputs = network(shc_input.transpose(1, 2), ilens)
            loss, loss_stats = loss_function(outputs[0][0], target)

            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            train_loss_sum += loss_item
            train_step_count += 1

            writer.add_scalars("Loss", {"Train": loss_item}, iter_count)
            writer.add_scalars(
                "Loss_Components_Train",
                {
                    "mse": loss_stats["mse"].item(),
                    "sisdr": loss_stats["sisdr"].item(),
                    "stft": loss_stats["stft"].item(),
                },
                iter_count,
            )
            iter_count += 1

            if idx % args.log_interval == 0 or idx == len(train_loader) - 1:
                train_bar.set_postfix(
                    loss=f"{loss_item:.6f}",
                    avg=f"{train_loss_sum / train_step_count:.6f}",
                    mse=f"{loss_stats['mse'].item():.3e}",
                    sisdr=f"{loss_stats['sisdr'].item():.3f}",
                    stft=f"{loss_stats['stft'].item():.3e}",
                    lr=f"{current_lr(optimizer):.2e}",
                )

            del shc_input, target, ilens, outputs, loss
            #if device.type == "cuda":
                #torch.cuda.empty_cache()

        epoch_train_loss = train_loss_sum / max(train_step_count, 1)
        loss_train_epoch.append(epoch_train_loss)
        train_time = time.time() - train_start_time

        log_info(
            f"Epoch {epoch_id}/{args.num_epoch} | Train done | "
            f"train_loss={epoch_train_loss:.6f} | time={train_time/60:.2f} min | "
            f"steps={train_step_count}"
        )

        # =========================
        # 8.2 验证阶段
        # =========================
        log_info(f"Epoch {epoch_id}/{args.num_epoch} | Val start   | batches={len(val_loader)}")

        network.eval()
        val_loss_sum = 0.0
        val_step_count = 0
        val_start_time = time.time()

        val_bar = make_progress_bar(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Val   {epoch_id}/{args.num_epoch}",
        )

        with torch.no_grad():
            for idx, egs in val_bar:
                shc_input = egs["shc_input"].to(device, non_blocking=True)
                target = egs["target"][:, 0, :].to(device, non_blocking=True)

                ilens = shc_input.size(2) * torch.ones(
                    shc_input.size(0), dtype=torch.int64, device=device
                )

                outputs = network(shc_input.transpose(1, 2), ilens)
                loss_val, loss_val_stats = loss_function(outputs[0][0], target)

                loss_val_item = loss_val.item()
                val_loss_sum += loss_val_item
                val_step_count += 1

                if idx % args.log_interval == 0 or idx == len(val_loader) - 1:
                    val_bar.set_postfix(
                        loss=f"{loss_val_item:.6f}",
                        avg=f"{val_loss_sum / val_step_count:.6f}",
                        mse=f"{loss_val_stats['mse'].item():.3e}",
                        sisdr=f"{loss_val_stats['sisdr'].item():.3f}",
                        stft=f"{loss_val_stats['stft'].item():.3e}",
                    )

                del shc_input, target, ilens, outputs, loss_val
                #if device.type == "cuda":
                    #torch.cuda.empty_cache()

        epoch_val_loss = val_loss_sum / max(val_step_count, 1)
        loss_val_epoch.append(epoch_val_loss)
        val_time = time.time() - val_start_time

        log_info(
            f"Epoch {epoch_id}/{args.num_epoch} | Val done   | "
            f"val_loss={epoch_val_loss:.6f} | time={val_time/60:.2f} min | "
            f"steps={val_step_count}"
        )

        writer.add_scalars(
            "Epoch_Loss",
            {
                "Train_Epoch": epoch_train_loss,
                "Validation_Epoch": epoch_val_loss
            },
            epoch_id
        )

        # =========================
        # 8.3 保存模型
        # =========================
        torch.save(network.state_dict(), os.path.join(modelpath, f"network_epoch{epoch_id}.pth"))

        lr_before_update = current_lr(optimizer)
        stop_training = False

        if epoch_val_loss <= min_val_loss:
            min_val_loss = epoch_val_loss
            val_no_impv = 0
            torch.save(network.state_dict(), os.path.join(modelpath, "model_best.pth"))
            log_info(
                f"Epoch {epoch_id}/{args.num_epoch} | "
                f"lr={lr_before_update:.8f} | best_val={epoch_val_loss:.6f} | saved=model_best.pth"
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
                stop_training = True

        metric_row = {
            "epoch": epoch_id,
            "train_loss": f"{epoch_train_loss:.8f}",
            "val_loss": f"{epoch_val_loss:.8f}",
            "lr": f"{lr_before_update:.10f}",
            "best_val_loss": f"{min_val_loss:.8f}",
            "train_minutes": f"{train_time / 60:.4f}",
            "val_minutes": f"{val_time / 60:.4f}",
            "train_steps": train_step_count,
            "val_steps": val_step_count,
        }
        append_epoch_metrics(log_dir / "epoch_metrics.csv", metric_row)
        append_epoch_metrics(Path(modelpath) / "epoch_metrics.csv", metric_row)

        # =========================
        # 8.4 保存 loss 曲线
        # =========================
        np.save(os.path.join(modelpath, "loss_val_epoch.npy"), loss_val_epoch)
        np.save(os.path.join(modelpath, "loss_train_epoch.npy"), loss_train_epoch)

        epochs_axis = np.arange(1, len(loss_train_epoch) + 1)
        plt.figure(figsize=(9, 6))
        plt.title("TFG-serial 8Mic Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.plot(epochs_axis, loss_train_epoch, marker="o", linewidth=2, label="train")
        plt.plot(epochs_axis, loss_val_epoch, marker="o", linewidth=2, label="val")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(modelpath, "Network_loss.png"))
        plt.close()

        gc.collect()
        #if device.type == "cuda":
            #torch.cuda.empty_cache()

        if stop_training:
            break

    writer.close()
    log_info("Training finished.")
