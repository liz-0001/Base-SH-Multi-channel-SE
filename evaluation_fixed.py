# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:09:25 2020
Modified for Mic8_2s_gpurir evaluation

@author: admin
"""

import os
import fnmatch
import argparse
import numpy as np
import pandas as pd
import scipy.io as io
import scipy.io.wavfile
import torch
from pesq import pesq
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources
from networks.tfgridnetv2 import TFGridNetV2


def get_args():
    parser = argparse.ArgumentParser("Evaluation for enhanced wavs")
    parser.add_argument('--dataset_root', type=str,
                        default='/data/lizhe/SH_data/Mic8_2s_gpurir',
                        help='dataset root path')
    parser.add_argument('--prediction_path', type=str,
                        required=True,
                        help='directory of enhanced wav files')
    parser.add_argument('--test_name', type=str,
                        default='mic_8',
                        help='test set name, e.g. mic_8 / mic_4 / mic_12 / mic_16')
    parser.add_argument('--save_dir', type=str,
                        default='',
                        help='directory to save metrics; default: prediction_path/results')
    parser.add_argument('--ref_channel', type=int,
                        default=0,
                        help='reference channel index when wav is multi-channel')
    parser.add_argument('--modelpath', type=str,
                        default='',
                        help='model directory for parameter/MAC profiling; empty means skip model profile')
    parser.add_argument('--model_name', type=str,
                        default='model_best.pth',
                        help='checkpoint name under modelpath')
    parser.add_argument('--profile_model', action='store_true',
                        help='estimate MACs/FLOPs using thop')
    parser.add_argument('--gpus', type=str,
                        default='0',
                        help='visible GPU id for model profiling')
    parser.add_argument('--sh_order', type=int,
                        default=4,
                        help='maximum SH order')
    parser.add_argument('--order_hidden_dim', type=int,
                        default=None,
                        help='hidden dim for each order encoder; default equals emb_dim')
    parser.add_argument('--sample_rate', type=int,
                        default=16000,
                        help='sample rate for MAC/FLOP profiling input')
    parser.add_argument('--chunk', type=int,
                        default=2,
                        help='seconds for MAC/FLOP profiling input')
    return parser.parse_args()


def format_number(value):
    if value is None:
        return "N/A"
    value = float(value)
    if value >= 1e9:
        return f"{value / 1e9:.3f}G"
    if value >= 1e6:
        return f"{value / 1e6:.3f}M"
    if value >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.0f}"


def load_checkpoint_state_dict(modelname):
    state_dict = torch.load(modelname, map_location='cpu')

    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def build_model(args):
    return TFGridNetV2(
        input_dim=None,
        n_srcs=1,
        n_fft=512,
        stride=256,
        n_imics=25,
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


def compute_model_profile(args):
    if args.modelpath.strip() == '':
        return {}

    if args.gpus.strip():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus.strip()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modelname = os.path.join(args.modelpath, args.model_name)
    if not os.path.isfile(modelname):
        raise FileNotFoundError(f"model checkpoint not found: {modelname}")

    model = build_model(args)
    model.load_state_dict(load_checkpoint_state_dict(modelname), strict=False)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    profile_info = {
        'parameters_total': float(total_params),
        'parameters_trainable': float(trainable_params),
        'parameters_frozen': float(total_params - trainable_params),
        'macs': np.nan,
        'flops_approx': np.nan,
        'thop_params': np.nan,
        'profile_seconds': float(args.chunk),
        'profile_samples': float(args.sample_rate * args.chunk),
        'profile_channels': 25.0,
    }

    if args.profile_model:
        try:
            from thop import profile

            n_samples = args.sample_rate * args.chunk
            dummy_input = torch.randn(1, n_samples, 25, device=device)
            dummy_ilens = torch.full((1,), n_samples, dtype=torch.int64, device=device)
            with torch.no_grad():
                macs, params_from_thop = profile(
                    model,
                    inputs=(dummy_input, dummy_ilens),
                    verbose=False,
                )
            profile_info.update(
                {
                    'macs': float(macs),
                    'flops_approx': float(2 * macs),
                    'thop_params': float(params_from_thop),
                }
            )
        except Exception as exc:
            print(f"[Warn] MACs/FLOPs profiling failed: {type(exc).__name__}: {exc}")

    return profile_info


def print_model_profile(profile_info):
    if not profile_info:
        print("Model profile: skipped")
        return

    print("========== Model Profile ==========")
    print(f"{'parameters_total':<22}: {profile_info['parameters_total']:,.0f} ({format_number(profile_info['parameters_total'])})")
    print(f"{'parameters_trainable':<22}: {profile_info['parameters_trainable']:,.0f} ({format_number(profile_info['parameters_trainable'])})")
    print(f"{'parameters_frozen':<22}: {profile_info['parameters_frozen']:,.0f} ({format_number(profile_info['parameters_frozen'])})")
    if np.isnan(profile_info['macs']):
        print("MACs/FLOPs estimate   : skipped or failed; add --profile_model to enable")
    else:
        print(f"{'profile_input':<22}: batch=1 | seconds={profile_info['profile_seconds']:.0f} | samples={profile_info['profile_samples']:.0f} | channels={profile_info['profile_channels']:.0f}")
        print(f"{'MACs':<22}: {profile_info['macs']:,.0f} ({format_number(profile_info['macs'])})")
        print(f"{'FLOPs_approx':<22}: {profile_info['flops_approx']:,.0f} ({format_number(profile_info['flops_approx'])})")
        print(f"{'thop_params':<22}: {profile_info['thop_params']:,.0f} ({format_number(profile_info['thop_params'])})")
    print("===================================")



def SDR(reference, estimation):
    sdr, _, _, _ = bss_eval_sources(reference[None, :], estimation[None, :])
    return float(sdr[0])


def SI_SDR(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [T]
        estimation: numpy.ndarray, [T]
    Returns:
        float
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    if np.all(reference_energy == 0):
        return -np.inf

    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / (reference_energy + 1e-8)
    projection = optimal_scaling * reference
    noise = estimation - projection

    denom = np.sum(noise ** 2, axis=-1) + 1e-8
    ratio = np.sum(projection ** 2, axis=-1) / denom
    return float(10 * np.log10(ratio + 1e-8))


def STOI(ref, est, sr=16000):
    return float(stoi(ref, est, sr, extended=False))


def WB_PESQ(ref, est, sr=16000):
    return float(pesq(sr, ref, est, "wb"))


def NB_PESQ(ref, est, sr=16000):
    return float(pesq(sr, ref, est, "nb"))


def read_wav(path, ref_channel=0):
    sr, data = scipy.io.wavfile.read(path)

    if len(data.shape) != 1:
        data = data[:, ref_channel]

    data = np.asarray(data, dtype=np.float32)
    return sr, data


def align_length(*signals):
    min_len = min(len(x) for x in signals)
    return [x[:min_len] for x in signals]


def safe_metric_compute(clean, mix, est, sr=16000):
    results = {}

    # mixed vs clean
    results['pesq_mix'] = NB_PESQ(clean, mix, sr)
    results['stoi_mix'] = STOI(clean, mix, sr)
    results['sdr_mix'] = SDR(clean, mix)
    results['si_sdr_mix'] = SI_SDR(clean, mix)

    # enhanced vs clean
    results['pesq_est'] = NB_PESQ(clean, est, sr)
    results['stoi_est'] = STOI(clean, est, sr)
    results['sdr_est'] = SDR(clean, est)
    results['si_sdr_est'] = SI_SDR(clean, est)

    results['pesq_improvement'] = results['pesq_est'] - results['pesq_mix']
    results['stoi_improvement'] = results['stoi_est'] - results['stoi_mix']
    results['sdr_improvement'] = results['sdr_est'] - results['sdr_mix']
    results['si_sdr_improvement'] = results['si_sdr_est'] - results['si_sdr_mix']

    return results


if __name__ == "__main__":
    args = get_args()
    # grouping-inter-sds branch: always profile the full SH interaction frontend.
    args.enable_order_grouping = True
    args.enable_high_low_guidance = True
    args.enable_low_to_high = True
    args.enable_high_to_low = True
    args.enable_adjacent_interaction = True

    dataset_root = args.dataset_root
    prediction_path = args.prediction_path
    test_name = args.test_name
    ref_channel = args.ref_channel

    mix_dir = os.path.join(dataset_root, 'generated_data', f'test_{test_name}', 'mix')
    ref_dir = os.path.join(dataset_root, 'generated_data', f'test_{test_name}', 'noreverb_ref')
    wav_scp = os.path.join(dataset_root, 'loader_txt', 'wav_scp', f'wav_scp_test_{test_name}.txt')

    if args.save_dir.strip() == '':
        save_dir = os.path.join(prediction_path, 'results')
    else:
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    print("========== Evaluation Config ==========")
    print("dataset_root   :", dataset_root)
    print("prediction_path:", prediction_path)
    print("test_name      :", test_name)
    print("mix_dir        :", mix_dir)
    print("ref_dir        :", ref_dir)
    print("wav_scp        :", wav_scp)
    print("save_dir       :", save_dir)
    print("ref_channel    :", ref_channel)
    print("modelpath      :", args.modelpath if args.modelpath.strip() else "<skip profile>")
    print("model_name     :", args.model_name)
    print("order_grouping :", args.enable_order_grouping)
    print("high_low_guide :", args.enable_high_low_guidance)
    print("low_to_high    :", args.enable_low_to_high)
    print("high_to_low    :", args.enable_high_to_low)
    print("adjacent_inter :", args.enable_adjacent_interaction)
    print("=======================================")

    profile_info = compute_model_profile(args)
    print_model_profile(profile_info)

    if not os.path.isdir(prediction_path):
        raise FileNotFoundError(f"prediction_path not found: {prediction_path}")
    if not os.path.isdir(mix_dir):
        raise FileNotFoundError(f"mix_dir not found: {mix_dir}")
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"ref_dir not found: {ref_dir}")
    if not os.path.isfile(wav_scp):
        raise FileNotFoundError(f"wav_scp not found: {wav_scp}")

    records = []

    with open(wav_scp, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total utterances in wav_scp: {len(lines)}")

    for idx, line in enumerate(lines):
        utt_id = line.strip().split('/')[-1]
        if not utt_id.endswith('.wav'):
            utt_id_wav = utt_id + '.wav'
        else:
            utt_id_wav = utt_id

        mix_path = os.path.join(mix_dir, utt_id_wav)
        ref_path = os.path.join(ref_dir, utt_id_wav)
        est_path = os.path.join(prediction_path, utt_id_wav)

        if not os.path.isfile(mix_path):
            print(f"[Skip] mix not found: {mix_path}")
            continue
        if not os.path.isfile(ref_path):
            print(f"[Skip] ref not found: {ref_path}")
            continue
        if not os.path.isfile(est_path):
            print(f"[Skip] est not found: {est_path}")
            continue

        try:
            sr_s, s = read_wav(ref_path, ref_channel=ref_channel)
            sr_y, y = read_wav(mix_path, ref_channel=ref_channel)
            sr_x, x = read_wav(est_path, ref_channel=0)

            if not (sr_s == sr_y == sr_x == 16000):
                print(f"[Warn] sample rate mismatch for {utt_id_wav}: ref={sr_s}, mix={sr_y}, est={sr_x}")

            s, y, x = align_length(s, y, x)

            metrics = safe_metric_compute(s, y, x, sr=16000)
            metrics['utt_id'] = utt_id_wav
            records.append(metrics)

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(lines)}")

        except Exception as e:
            print(f"[Error] utt_id: {utt_id_wav}")
            print(f"        mix_path: {mix_path}")
            print(f"        ref_path: {ref_path}")
            print(f"        est_path: {est_path}")
            print(f"        info: {e}")
            continue

    if len(records) == 0:
        raise RuntimeError("No valid utterances were evaluated. Please check prediction_path and file names.")

    df_detail = pd.DataFrame(records)

    summary = {
        'pesq_mix': df_detail['pesq_mix'].mean(),
        'pesq_est': df_detail['pesq_est'].mean(),
        'stoi_mix': df_detail['stoi_mix'].mean(),
        'stoi_est': df_detail['stoi_est'].mean(),
        'sdr_mix': df_detail['sdr_mix'].mean(),
        'sdr_est': df_detail['sdr_est'].mean(),
        'si_sdr_mix': df_detail['si_sdr_mix'].mean(),
        'si_sdr_est': df_detail['si_sdr_est'].mean(),
        'pesq_improvement': df_detail['pesq_improvement'].mean(),
        'stoi_improvement': df_detail['stoi_improvement'].mean(),
        'sdr_improvement': df_detail['sdr_improvement'].mean(),
        'si_sdr_improvement': df_detail['si_sdr_improvement'].mean(),
    }
    summary.update(profile_info)

    print("\n========== Final Average Results ==========")
    print(f"PESQ  mix: {summary['pesq_mix']:.6f}")
    print(f"PESQ  est: {summary['pesq_est']:.6f}")
    print(f"PESQ  imp: {summary['pesq_improvement']:.6f}")
    print(f"STOI  mix: {summary['stoi_mix']:.6f}")
    print(f"STOI  est: {summary['stoi_est']:.6f}")
    print(f"STOI  imp: {summary['stoi_improvement']:.6f}")
    print(f"SDR   mix: {summary['sdr_mix']:.6f}")
    print(f"SDR   est: {summary['sdr_est']:.6f}")
    print(f"SDR   imp: {summary['sdr_improvement']:.6f}")
    print(f"SI-SDR mix: {summary['si_sdr_mix']:.6f}")
    print(f"SI-SDR est: {summary['si_sdr_est']:.6f}")
    print(f"SI-SDR imp: {summary['si_sdr_improvement']:.6f}")
    print("===========================================\n")

    # 保存逐条结果
    detail_csv = os.path.join(save_dir, f'{test_name}_detail_metrics.csv')
    df_detail.to_csv(detail_csv, index=False, encoding='utf-8-sig')

    # 保存汇总结果
    df_summary = pd.DataFrame([summary], index=[test_name])
    summary_csv = os.path.join(save_dir, f'{test_name}_summary_metrics.csv')
    df_summary.to_csv(summary_csv, encoding='utf-8-sig')

    # 保存 mat
    mat_path = os.path.join(save_dir, f'{test_name}_metrics.mat')
    io.savemat(
        mat_path,
        {
            'pesq_mix': np.array([summary['pesq_mix']], dtype=np.float32),
            'pesq_est': np.array([summary['pesq_est']], dtype=np.float32),
            'stoi_mix': np.array([summary['stoi_mix']], dtype=np.float32),
            'stoi_est': np.array([summary['stoi_est']], dtype=np.float32),
            'sdr_mix': np.array([summary['sdr_mix']], dtype=np.float32),
            'sdr_est': np.array([summary['sdr_est']], dtype=np.float32),
            'si_sdr_mix': np.array([summary['si_sdr_mix']], dtype=np.float32),
            'si_sdr_est': np.array([summary['si_sdr_est']], dtype=np.float32),
            'pesq_improvement': np.array([summary['pesq_improvement']], dtype=np.float32),
            'stoi_improvement': np.array([summary['stoi_improvement']], dtype=np.float32),
            'sdr_improvement': np.array([summary['sdr_improvement']], dtype=np.float32),
            'si_sdr_improvement': np.array([summary['si_sdr_improvement']], dtype=np.float32),
            'parameters_total': np.array([summary.get('parameters_total', np.nan)], dtype=np.float32),
            'parameters_trainable': np.array([summary.get('parameters_trainable', np.nan)], dtype=np.float32),
            'parameters_frozen': np.array([summary.get('parameters_frozen', np.nan)], dtype=np.float32),
            'macs': np.array([summary.get('macs', np.nan)], dtype=np.float32),
            'flops_approx': np.array([summary.get('flops_approx', np.nan)], dtype=np.float32),
            'thop_params': np.array([summary.get('thop_params', np.nan)], dtype=np.float32),
        }
    )

    print("Saved files:")
    print(detail_csv)
    print(summary_csv)
    print(mat_path)
