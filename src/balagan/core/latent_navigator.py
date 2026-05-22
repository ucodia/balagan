"""Seed-grid bilinear z->w navigation through the canonical mapping network."""

import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SEED_MASK = (1 << 32) - 1
_BILINEAR_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))


def _corner_seeds(
    latent_x: float, latent_y: float, step_y: int
) -> list[tuple[int, float]]:
    """Return the (seed, weight) corners bilinearly bracketing a latent position.

    Mirrors the seed grid of NVIDIA's StyleGAN3 visualizer (viz/latent_widget.py):
    four integer-seed corners around the continuous (latent_x, latent_y), each
    weighted by bilinear proximity. Zero-weight corners are dropped; the
    surviving weights sum to 1.
    """
    base_x = math.floor(latent_x)
    base_y = math.floor(latent_y)
    corners: list[tuple[int, float]] = []
    for offset_x, offset_y in _BILINEAR_CORNERS:
        seed_x = base_x + offset_x
        seed_y = base_y + offset_y
        seed = (seed_x + seed_y * step_y) & _SEED_MASK
        weight = (1.0 - abs(latent_x - seed_x)) * (1.0 - abs(latent_y - seed_y))
        if weight > 0.0:
            corners.append((seed, weight))
    return corners


class LatentNavigator:
    """Maps a continuous seed-space position to a single W latent.

    Each integer point of seed space owns a deterministic Z (a seeded normal
    draw). A continuous position bilinearly blends the four surrounding seeds in
    W space, mapping every Z through the resident canonical mapping network --
    so the latent geometry is fixed by one model while synthesis varies, which
    is the BalaGAN-faithful chimera behavior.

    The w_avg subtract/combine/add and truncation follow NVIDIA's StyleGAN3
    visualizer (viz/renderer.py).
    """

    def __init__(
        self, canonical_mapping: torch.nn.Module, z_dim: int, step_y: int = 100
    ) -> None:
        self._mapping = canonical_mapping
        self._z_dim = z_dim
        self._c_dim = int(canonical_mapping.c_dim)
        self._step_y = step_y
        self._w_avg = canonical_mapping.w_avg

    def __call__(
        self, latent_x: float, latent_y: float, truncation_psi: float
    ) -> torch.Tensor:
        """Return the blended W latent of shape [num_ws, w_dim] for a position."""
        corners = _corner_seeds(latent_x, latent_y, self._step_y)
        unique_seeds = sorted({seed for seed, _ in corners})

        zs = np.zeros([len(unique_seeds), self._z_dim], dtype=np.float32)
        cs = np.zeros([len(unique_seeds), self._c_dim], dtype=np.float32)
        for index, seed in enumerate(unique_seeds):
            rnd = np.random.RandomState(seed)
            zs[index] = rnd.randn(self._z_dim)
            if self._c_dim > 0:
                cs[index, rnd.randint(self._c_dim)] = 1

        device = self._w_avg.device
        z_batch = torch.from_numpy(zs).to(device)
        c_batch = torch.from_numpy(cs).to(device)
        mapped = self._mapping(
            z=z_batch, c=c_batch, truncation_psi=truncation_psi, truncation_cutoff=None
        )
        mapped = mapped - self._w_avg
        w_by_seed = dict(zip(unique_seeds, mapped))

        blended = torch.stack(
            [w_by_seed[seed] * weight for seed, weight in corners]
        ).sum(dim=0)
        return blended + self._w_avg
