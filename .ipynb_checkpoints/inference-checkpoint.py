# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:43:18 2020

@author: admin
"""

import shutil
import argparse
import librosa
import spaudiopy
import torch
import os
import fnmatch
import numpy as np
import soundfile as sf
from scipy import signal, io
from tqdm import tqdm
from evaluation_fixed import NB_PESQ, STOI
from networks.IGCRN import IGCRN
import warnings
from multiprocessing import Pool
from config.train_config import *
from networks.enhancer import PonderEnhancer
from networks.tfgridnetv2 import TFGridNetV2

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data


def calculate_metrics(ref, mix, est):
    return NB_PESQ(ref, mix), NB_PESQ(ref, est), STOI(ref, mix), STOI(ref, est)


def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / (r + 1e-8))
    return r, theta, phi


def microphone_positions_spherical(cartesian_positions):
    positions_spherical = np.zeros(cartesian_positions.shape)
    for i, (x, y, z) in enumerate(cartesian_positions):
        positions_spherical[i] = cart2sph(x, y, z)
    return positions_spherical


def wav_generator(mix_path, ref_path, mic_path):
    mix = audioread(mix_path)
    ref = audioread(ref_path)

    mic_data = np.load(mic_path)  # shape: (C, 3)
    center = np.mean(mic_data, axis=0)
    transformed_coords = mic_data - center
    mic_positions_spherical = microphone_positions_spherical(transformed_coords)

    colat = mic_positions_spherical[:, 2]
    azi = mic_positions_spherical[:, 1]

    sh_type = 'real'
    sph_order = 4
    mix_coeffs = spaudiopy.sph.src_to_sh(mix.T, azi, colat, sph_order, sh_type)

    shc_input = torch.tensor(np.float32(mix_coeffs)).unsqueeze(0).to(device)
    ilens = torch.full((shc_input.size(0),), shc_input.size(2), dtype=torch.int64, device=device)

    with torch.no_grad():
        outputs = load_network(shc_input.transpose(1, 2), ilens)

    outputs = outputs[0][0]
    est = outputs.squeeze().T.detach().cpu().numpy()

    # 保证 est 是 1 维
    est = np.asarray(est).squeeze()

    # 参考和混合取第 0 通道
    if len(ref.shape) != 1:
        ref_eval = ref[:, 0]
    else:
        ref_eval = ref

    if len(mix.shape) != 1:
        mix_eval = mix[:, 0]
    else:
        mix_eval = mix

    min_len = min(len(ref_eval), len(mix_eval), len(est))
    ref_eval = ref_eval[:min_len]
    mix_eval = mix_eval[:min_len]
    est = est[:min_len]

    pesq_mix, pesq_est, stoi_mix, stoi_est = calculate_metrics(ref_eval, mix_eval, est)
    pesq_mix = np.mean(np.array(pesq_mix))
    pesq_est = np.mean(np.array(pesq_est))
    stoi_mix = np.mean(np.array(stoi_mix))
    stoi_est = np.mean(np.array(stoi_est))

    return pesq_mix, pesq_est, stoi_mix, stoi_est, est


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    modelpath = 'model_miso_new_data/'

    # test_list = ['mic_8', 'mic_4', 'mic_12', 'mic_16']
    test_list = ['mic_8']

    file_path = '/autodl-tmp/Mic8_2s_gpurir'
    mic_path_root = '/autodl-tmp/Mic8_2s_gpurir/RIR/cir_uniform_8/test_rir'

    for test_name in test_list:
        test_wav_scp = os.path.join(file_path, 'loader_txt', 'wav_scp', 'wav_scp_test_' + test_name + '.txt')
        wav_path = os.path.join(file_path, 'generated_data', 'test_' + test_name, 'mix')
        ref_dir = os.path.join(file_path, 'generated_data', 'test_' + test_name, 'noreverb_ref')
        mic_dir = os.path.join(mic_path_root, test_name, 'MIC')

        modelname = os.path.join(modelpath, 'network_epoch18.pth')

        # 和 evaluation_fixed.py 的 prediction_path 保持一致
        pred_save_dir = os.path.join(file_path, 'predictions_tfg_serial_test_' + test_name)
        os.makedirs(pred_save_dir, exist_ok=True)

        print(str(modelname))
        print("Processing test ..." + str(test_name))
        print("Using device:", device)
        print("Prediction wav save dir:", pred_save_dir)

        load_network = TFGridNetV2(
            input_dim=None,
            n_srcs=1,
            n_fft=512,
            stride=256,
            n_imics=25,
            n_layers=3,
            lstm_hidden_units=128,
            attn_approx_qk_dim=256,
            emb_dim=32
        )

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
        state_dict = new_state_dict

        load_network.load_state_dict(state_dict)
        load_network = load_network.to(device)

        if torch.cuda.device_count() > 1:
            load_network = torch.nn.DataParallel(load_network)

        load_network.eval()

        pesq_mix_list = []
        pesq_est_list = []
        stoi_mix_list = []
        stoi_est_list = []

        with open(test_wav_scp, 'r', encoding='utf-8') as infile:
            data = infile.readlines()
            for i in tqdm(range(len(data))):
                utt_id = data[i].strip("\n").split('/')[-1]

                def ensure_wav(name):
                    return name if name.endswith('.wav') else name + '.wav'

                utt_id_wav = ensure_wav(utt_id)

                mix_path = os.path.join(wav_path, utt_id_wav)
                ref_path = os.path.join(ref_dir, utt_id_wav)

                mic_id = utt_id.split('#')[2].split('rir')[-1]
                mic_path = os.path.join(mic_dir, 'mic' + mic_id + '.npy')

                try:
                    pesq_mix, pesq_est, stoi_mix, stoi_est, est = wav_generator(mix_path, ref_path, mic_path)

                    pesq_mix_list.append(pesq_mix)
                    pesq_est_list.append(pesq_est)
                    stoi_mix_list.append(stoi_mix)
                    stoi_est_list.append(stoi_est)

                    # 保存增强后的 wav，文件名与 evaluation_fixed.py 一致
                    est_save_path = os.path.join(pred_save_dir, utt_id_wav)
                    sf.write(est_save_path, est, 16000)

                except Exception as e:
                    print(f'Error utterance: {utt_id}')
                    print(f'mix_path: {mix_path}')
                    print(f'ref_path: {ref_path}')
                    print(f'mic_path: {mic_path}')
                    print(f'Error info: {e}')
                    continue

        if len(pesq_mix_list) == 0:
            print(f"{test_name}: no valid samples processed.")
            continue

        pesq_mix = np.mean(np.array(pesq_mix_list))
        pesq_est = np.mean(np.array(pesq_est_list))
        stoi_mix = np.mean(np.array(stoi_mix_list))
        stoi_est = np.mean(np.array(stoi_est_list))

        print(test_name + "_result:")
        print(
            'pesq_mix:' + str(pesq_mix) + '   ' +
            'pesq_est:' + str(pesq_est) + '   ' +
            'stoi_mix:' + str(stoi_mix) + '   ' +
            'stoi_est:' + str(stoi_est)
        )

        res1path = os.path.join(modelpath, 'result_model_epoch18')
        if not os.path.isdir(res1path):
            os.makedirs(res1path)

        io.savemat(
            os.path.join(res1path, test_name + '_metrics.mat'),
            {
                'pesq_mix': pesq_mix,
                'pesq_est': pesq_est,
                'stoi_mix': stoi_mix,
                'stoi_est': stoi_est
            }
        )

        print("Saved average metrics to:", os.path.join(res1path, test_name + '_metrics.mat'))
        print("Saved enhanced wavs to:", pred_save_dir)
