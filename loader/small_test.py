"""
子集 DataLoader：接口与 IGCRN_dataloader 一致，仅在内存中取原数据集约 1/10 用于训练。
不修改、不生成任何数据集文件。
"""

import random
import os

import numpy as np
import spaudiopy
import torch.utils.data as tud
from torch.utils.data import Dataset

from loader.IGCRN_dataloader import audioread, parse_scp

DEFAULT_FRACTION = 0.1
DEFAULT_SEED = 123


def subsample_wav_list(wav_list, fraction=DEFAULT_FRACTION, seed=DEFAULT_SEED):
    """从列表中确定性随机抽取 fraction 比例样本，不写入磁盘。"""
    total = len(wav_list)
    if total == 0:
        return []

    n_keep = max(1, int(total * fraction))
    if n_keep >= total:
        return list(wav_list)

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total), n_keep))
    return [wav_list[i] for i in indices]


class FixDataset(Dataset):
    """与 IGCRN_dataloader.FixDataset 相同输入输出，初始化时仅保留 fraction 子集。"""

    def __init__(
        self,
        wav_scp,
        mix_dir,
        ref_dir,
        mic_dir,
        repeat=1,
        chunk=4,
        sample_rate=16000,
        fraction=DEFAULT_FRACTION,
        seed=DEFAULT_SEED,
    ):
        super(FixDataset, self).__init__()

        full_list = []
        parse_scp(wav_scp, full_list)
        self.wav_list = subsample_wav_list(full_list, fraction=fraction, seed=seed)

        print(
            f"[small_test] wav_scp={wav_scp} | "
            f"total={len(full_list)} -> subset={len(self.wav_list)} "
            f"({fraction * 100:.0f}%, seed={seed})"
        )

        self.mix_dir = mix_dir
        self.ref_dir = ref_dir
        self.mic_dir = mic_dir
        self.segment_length = chunk * sample_rate
        self.wav_list *= repeat

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        utt_id = self.wav_list[index].strip()

        file_name = utt_id if utt_id.endswith(".wav") else utt_id + ".wav"

        mix_path = os.path.join(self.mix_dir, file_name)
        ref_path = os.path.join(self.ref_dir, file_name)

        mix = audioread(mix_path)
        ref = audioread(ref_path)

        mic_file = os.path.join(
            self.mic_dir,
            'mic_array_pos' + utt_id.split('#')[2].split('rir')[-1] + '.npy'
        )

        mic_data = np.load(mic_file)
        center = np.mean(mic_data, axis=0)
        transformed_coords = mic_data - center
        mic_positions_spherical = self.microphone_positions_spherical(transformed_coords)
        colat = mic_positions_spherical[:, 2]
        azi = mic_positions_spherical[:, 1]
        sh_type = 'real'
        sph_order = 4

        mix_coeffs = spaudiopy.sph.src_to_sh(mix.T, azi, colat, sph_order, sh_type)

        egs = {
            "stft_input": np.float32(mix.T),
            "shc_input": np.float32(mix_coeffs),
            "target": np.float32(ref.T),
            "file_name": utt_id,
        }
        return egs

    def cart2sph(self, x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        return r, theta, phi

    def microphone_positions_spherical(self, cartesian_positions):
        positions_spherical = np.zeros(cartesian_positions.shape)
        for i, (x, y, z) in enumerate(cartesian_positions):
            positions_spherical[i] = self.cart2sph(x, y, z)
        return positions_spherical


def make_fix_loader(
    wav_scp,
    mix_dir,
    ref_dir,
    mic_dir,
    batch_size=8,
    repeat=1,
    num_workers=16,
    chunk=4,
    sample_rate=16000,
    fraction=DEFAULT_FRACTION,
    seed=DEFAULT_SEED,
):
    """与 IGCRN_dataloader.make_fix_loader 参数一致，额外支持 fraction / seed。"""
    dataset = FixDataset(
        wav_scp=wav_scp,
        mix_dir=mix_dir,
        ref_dir=ref_dir,
        mic_dir=mic_dir,
        repeat=repeat,
        chunk=chunk,
        sample_rate=sample_rate,
        fraction=fraction,
        seed=seed,
    )

    loader = tud.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )
    return loader


def test_loader():
    wav_scp = '/data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_scp/wav_scp_train.txt'
    mix_dir = '/data/lizhe/SH_data/Mic8_2s_gpurir/generated_data/train/mix'
    ref_dir = '/data/lizhe/SH_data/Mic8_2s_gpurir/generated_data/train/noreverb_ref'
    mic_dir = '/data/lizhe/SH_data/Mic8_2s_gpurir/RIR/cir_uniform_8/train_val_rir/MIC'

    loader = make_fix_loader(
        wav_scp=wav_scp,
        mix_dir=mix_dir,
        ref_dir=ref_dir,
        mic_dir=mic_dir,
        batch_size=2,
        repeat=1,
        num_workers=0,
        chunk=4,
        sample_rate=16000,
        fraction=0.1,
        seed=123,
    )

    print('batches:', len(loader))
    for idx, egs in enumerate(loader):
        print(
            f"batch {idx}: stft_input={egs['stft_input'].shape}, "
            f"shc_input={egs['shc_input'].shape}, target={egs['target'].shape}"
        )
        if idx >= 2:
            break
    print('done!')


if __name__ == "__main__":
    test_loader()
