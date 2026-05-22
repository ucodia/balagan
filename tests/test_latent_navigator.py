"""Tests for balagan.core.latent_navigator: seed-grid bilinear z->w."""

import numpy as np
import pytest
import torch

from balagan.core.latent_navigator import LatentNavigator, _corner_seeds


class StubMapping(torch.nn.Module):
    """Minimal stand-in for a StyleGAN2 MappingNetwork.

    With z_dim == w_dim the forward pass is an identity broadcast of z across
    num_ws, then truncation toward w_avg exactly as the real network does.
    """

    def __init__(self, z_dim=4, c_dim=0, w_dim=4, num_ws=2, w_avg=None):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.register_buffer(
            "w_avg", torch.zeros(w_dim) if w_avg is None else w_avg
        )

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None):
        x = z.to(torch.float32).unsqueeze(1).repeat(1, self.num_ws, 1)
        if truncation_psi != 1:
            x = self.w_avg.lerp(x, truncation_psi)
        return x


def make_navigator(w_avg=None, step_y=100) -> LatentNavigator:
    mapping = StubMapping(z_dim=4, c_dim=0, w_dim=4, num_ws=2, w_avg=w_avg)
    return LatentNavigator(mapping, z_dim=4, step_y=step_y)


def z_for(seed: int) -> torch.Tensor:
    """The deterministic Z for a seed, generated as the spec prescribes."""
    return torch.from_numpy(np.random.RandomState(seed).randn(4).astype(np.float32))


# --- seed grid ---------------------------------------------------------------


def test_corner_seeds_integer_position_is_single_seed():
    assert _corner_seeds(0.0, 0.0, 100) == [(0, 1.0)]
    assert _corner_seeds(3.0, 0.0, 100) == [(3, 1.0)]
    assert _corner_seeds(0.0, 2.0, 100) == [(200, 1.0)]
    assert _corner_seeds(5.0, 7.0, 100) == [(705, 1.0)]


def test_corner_seeds_fractional_position_is_bilinear():
    corners = dict(_corner_seeds(2.25, 3.75, 100))
    assert set(corners) == {302, 303, 402, 403}
    assert corners[302] == pytest.approx(0.1875)
    assert corners[303] == pytest.approx(0.0625)
    assert corners[402] == pytest.approx(0.5625)
    assert corners[403] == pytest.approx(0.1875)
    assert sum(corners.values()) == pytest.approx(1.0)


def test_corner_seeds_drops_zero_weight_corners():
    corners = _corner_seeds(2.5, 4.0, 100)  # fractional x, integer y
    assert len(corners) == 2
    assert {seed for seed, _ in corners} == {402, 403}


def test_corner_seeds_apply_32bit_mask_to_negative_position():
    assert _corner_seeds(-1.0, 0.0, 100) == [((1 << 32) - 1, 1.0)]


# --- navigator ---------------------------------------------------------------


def test_navigator_returns_num_ws_by_w_dim():
    navigator = make_navigator()
    assert tuple(navigator(1.5, 2.5, 0.7).shape) == (2, 4)


def test_integer_position_maps_a_single_seed():
    navigator = make_navigator()
    w = navigator(3.0, 0.0, 1.0)
    expected = z_for(3).unsqueeze(0).repeat(2, 1)
    assert torch.allclose(w, expected)


def test_fractional_position_blends_corner_seeds():
    navigator = make_navigator()
    w = navigator(2.5, 0.0, 1.0)  # seeds 2 and 3, weights 0.5 / 0.5
    expected = (0.5 * (z_for(2) + z_for(3))).unsqueeze(0).repeat(2, 1)
    assert torch.allclose(w, expected)


def test_truncation_psi_scales_output_toward_w_avg():
    navigator = make_navigator()  # w_avg is zero
    full = navigator(2.5, 1.5, 1.0)
    half = navigator(2.5, 1.5, 0.5)
    assert torch.allclose(half, 0.5 * full)


def test_w_avg_offset_cancels_at_full_truncation():
    plain = make_navigator(w_avg=torch.zeros(4))
    offset = make_navigator(w_avg=torch.tensor([1.0, -2.0, 3.0, -4.0]))
    # truncation_psi == 1 means no truncation; the w_avg subtract/add must
    # cancel exactly because the bilinear weights sum to 1.
    assert torch.allclose(plain(2.5, 1.5, 1.0), offset(2.5, 1.5, 1.0))


def test_navigator_is_deterministic():
    navigator = make_navigator()
    assert torch.equal(navigator(2.5, 1.5, 0.7), navigator(2.5, 1.5, 0.7))
