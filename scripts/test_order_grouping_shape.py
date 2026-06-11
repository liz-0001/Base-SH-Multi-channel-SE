import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from networks.tfgridnetv2 import OrderWiseSHGroupingEncoder


def main():
    batch_size = 2
    sh_order = 3
    n_imics = (sh_order + 1) ** 2
    emb_dim = 32
    n_frames = 9
    n_freqs = 17

    module = OrderWiseSHGroupingEncoder(
        n_imics=n_imics,
        emb_dim=emb_dim,
        sh_order=sh_order,
    )
    x = torch.randn(batch_size, 2 * n_imics, n_frames, n_freqs)
    y = module(x)

    expected_widths = [1, 3, 5, 7]
    actual_widths = [end - start for start, end in module.order_slices]

    assert actual_widths == expected_widths, actual_widths
    assert y.shape == (batch_size, emb_dim, n_frames, n_freqs), y.shape

    print("Order grouping shape test passed.")
    print(f"order widths: {actual_widths}")
    print(f"input shape : {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
