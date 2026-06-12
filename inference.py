# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:43:18 2020

@author: admin
"""

import librosa
import spaudiopy
import torch
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import warnings
from networks.tfgridnetv2 import TFGridNetV2

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mic_path(mic_dir, mic_prefix, utt_id):
    mic_id = utt_id.split('#')[2].split('rir', 1)[-1]
    mic_id = mic_id.removesuffix('.wav').removesuffix('.npy')
    return os.path.join(mic_dir, mic_prefix + mic_id + '.npy')


def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data


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


def wav_generator(mix_path, mic_path):
    mix = audioread(mix_path)

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

    return est


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser("TFG inference")
    _parser.add_argument("--modelpath", type=str, default="model_tfg_serial_8mic",
                         help="模型目录，默认使用 8 麦串行 TFG baseline")
    _parser.add_argument("--model_name", type=str, default="model_best.pth")
    _parser.add_argument("--file_path", type=str, default="/data/lizhe/SH_data/Mic8_2s_gpurir")
    _parser.add_argument("--mic_path_root", type=str,
                         default="/data/lizhe/SH_data/Mic8_2s_gpurir/RIR/cir_uniform_8/test_rir")
    _parser.add_argument("--mic_prefix", type=str, default="mic_array_pos")
    _parser.add_argument("--test_name", type=str, default="mic_8")
    _parser.add_argument("--gpus", type=str, default="0")
    _parser.add_argument("--prediction_path", type=str, default="", help="增强 wav 保存目录；默认保存到 dataset root 下")
    _parser.add_argument("--enable_order_grouping", action="store_true", help="推理 grouping-only 模型时开启")
    _parser.add_argument("--sh_order", type=int, default=4)
    _parser.add_argument("--order_hidden_dim", type=int, default=None)
    _inf_args = _parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = _inf_args.gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    modelpath = _inf_args.modelpath

    # test_list = ['mic_8', 'mic_4', 'mic_12', 'mic_16']
    test_list = [_inf_args.test_name]

    file_path = _inf_args.file_path
    mic_path_root = _inf_args.mic_path_root
    mic_prefix = _inf_args.mic_prefix

    for test_name in test_list:
        test_wav_scp = os.path.join(file_path, 'loader_txt', 'wav_scp', 'wav_scp_test_' + test_name + '.txt')
        wav_path = os.path.join(file_path, 'generated_data', 'test_' + test_name, 'mix')
        mic_dir = os.path.join(mic_path_root, test_name, 'MIC')

        modelname = os.path.join(modelpath, _inf_args.model_name)

        if _inf_args.prediction_path.strip():
            pred_save_dir = _inf_args.prediction_path
        else:
            model_tag = os.path.basename(os.path.normpath(modelpath))
            pred_save_dir = os.path.join(file_path, f'predictions_{model_tag}_test_{test_name}')
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
            emb_dim=32,
            enable_order_grouping=_inf_args.enable_order_grouping,
            sh_order=_inf_args.sh_order,
            order_hidden_dim=_inf_args.order_hidden_dim,
        )
        if _inf_args.enable_order_grouping:
            print(
                "Model: TFGridNetV2 serial + order-wise SH grouping "
                f"(sh_order={_inf_args.sh_order}, order_hidden_dim={_inf_args.order_hidden_dim or 32})"
            )
        else:
            print("Baseline model: TFGridNetV2 serial SHC input, no ADFS")

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

        load_network.load_state_dict(state_dict,strict=False)
        load_network = load_network.to(device)

        load_network.eval()
        print("Inference only: enhanced wavs will be saved. Run evaluation_fixed.py for metrics/profile.")

        with open(test_wav_scp, 'r', encoding='utf-8') as infile:
            data = infile.readlines()
            for i in tqdm(range(len(data))):
                utt_id = data[i].strip("\n").split('/')[-1]

                def ensure_wav(name):
                    return name if name.endswith('.wav') else name + '.wav'

                utt_id_wav = ensure_wav(utt_id)

                mix_path = os.path.join(wav_path, utt_id_wav)
                mic_path = get_mic_path(mic_dir, mic_prefix, utt_id)

                try:
                    est = wav_generator(mix_path, mic_path)

                    est_save_path = os.path.join(pred_save_dir, utt_id_wav)
                    sf.write(est_save_path, est, 16000)

                except Exception as e:
                    print(f'Error utterance: {utt_id}')
                    print(f'mix_path: {mix_path}')
                    print(f'mic_path: {mic_path}')
                    print(f'Error info: {e}')
                    continue

        print("Saved enhanced wavs to:", pred_save_dir)
