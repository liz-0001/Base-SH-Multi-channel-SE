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
from pesq import pesq
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources


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
    return parser.parse_args()



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

    return results


if __name__ == "__main__":
    args = get_args()
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
    print("=======================================")

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
    }

    print("\n========== Final Average Results ==========")
    print(f"PESQ  mix: {summary['pesq_mix']:.6f}")
    print(f"PESQ  est: {summary['pesq_est']:.6f}")
    print(f"STOI  mix: {summary['stoi_mix']:.6f}")
    print(f"STOI  est: {summary['stoi_est']:.6f}")
    print(f"SDR   mix: {summary['sdr_mix']:.6f}")
    print(f"SDR   est: {summary['sdr_est']:.6f}")
    print(f"SI-SDR mix: {summary['si_sdr_mix']:.6f}")
    print(f"SI-SDR est: {summary['si_sdr_est']:.6f}")
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
        }
    )

    print("Saved files:")
    print(detail_csv)
    print(summary_csv)
    print(mat_path)
