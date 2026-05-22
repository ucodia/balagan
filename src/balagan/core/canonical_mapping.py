"""Loads the always-resident canonical mapping network.

Z->W mapping uses this one network for the engine's whole lifetime, which fixes
the latent geometry while synthesis varies across snapshots -- the
BalaGAN-faithful chimera behavior.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_network_pkl(pkl_path: Path) -> dict:
    """Load a StyleGAN snapshot pickle via the stylegan3 submodule.

    The submodule imports are deferred: ``stylegan3/`` is only importable once
    cli.py has put it on ``sys.path``, and tests inject a stub loader instead.
    """
    import dnnlib
    import legacy

    with dnnlib.util.open_url(str(pkl_path)) as stream:
        return legacy.load_network_pkl(stream)


def load_canonical_mapping(
    pkl_path: Path | str,
    device: str | torch.device,
    *,
    pkl_loader: Callable[[Path], dict] | None = None,
) -> torch.nn.Module:
    """Load the canonical snapshot, extract its mapping network, and move it to
    the inference device.

    Only the mapping network is retained; the rest of the generator is dropped.
    """
    pkl_path = Path(pkl_path)
    load = pkl_loader if pkl_loader is not None else load_network_pkl
    generator = load(pkl_path)["G_ema"]
    mapping = generator.mapping.to(device)
    logger.info("Loaded canonical mapping from %s onto %s", pkl_path, device)
    return mapping
