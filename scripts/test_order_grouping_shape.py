import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from networks.tfgridnetv2 import AdjacentOrderInteraction, OrderWiseSHGroupingEncoder


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
    adjacent_module = OrderWiseSHGroupingEncoder(
        n_imics=n_imics,
        emb_dim=emb_dim,
        sh_order=sh_order,
        enable_adjacent_interaction=True,
    )
    x = torch.randn(batch_size, 2 * n_imics, n_frames, n_freqs)
    y = module(x)
    y_adj = adjacent_module(x)

    expected_widths = [1, 3, 5, 7]
    actual_widths = [end - start for start, end in module.order_slices]

    assert actual_widths == expected_widths, actual_widths
    assert y.shape == (batch_size, emb_dim, n_frames, n_freqs), y.shape
    assert y_adj.shape == (batch_size, emb_dim, n_frames, n_freqs), y_adj.shape

    hidden_dim = 16
    order_features = [
        torch.randn(batch_size, hidden_dim, n_frames, n_freqs)
        for _ in expected_widths
    ]
    adjacent = AdjacentOrderInteraction(num_orders=len(order_features), hidden_dim=hidden_dim)
    updated_features = adjacent(order_features)
    assert len(updated_features) == len(order_features)
    for before, after in zip(order_features, updated_features):
        assert after.shape == before.shape, (before.shape, after.shape)

    print("Order grouping shape test passed.")
    print(f"order widths: {actual_widths}")
    print(f"input shape : {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"adjacent output shape: {tuple(y_adj.shape)}")


if __name__ == "__main__":
    main()
