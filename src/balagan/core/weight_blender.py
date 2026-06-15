"""In-place weight-space blending of training snapshots.

Ports the weight-interpolation strategy from the gen_balagan.py prototype:
cache each snapshot's state_dict at load time, pre-allocate one blend-target
network, and lerp cached tensors into the target in place per call -- never
calling state_dict() or constructing a module on the per-frame path.
"""

import copy
import logging

import torch

logger = logging.getLogger(__name__)


class WeightBlender:
    """Blends same-architecture networks in weight space.

    Floating-point tensors are linearly interpolated; non-floating-point buffers
    (e.g. ``num_batches_tracked``) are hard-copied from the dominant side, since
    lerp is meaningless for them. Safe only because all snapshots come from one
    training run and therefore share an architecture and loss basin.
    """

    def __init__(self) -> None:
        self._networks: dict[int, torch.nn.Module] = {}
        self._state_dicts: dict[int, dict[str, torch.Tensor]] = {}
        self._blend_net: torch.nn.Module | None = None
        self._blend_state: dict[str, torch.Tensor] | None = None

    def cache_snapshot(self, kimg: int, network: torch.nn.Module) -> None:
        """Cache a snapshot's network and a reference to its state_dict, keyed by
        kimg. The first cached snapshot also seeds the pre-allocated blend target.
        """
        self._networks[kimg] = network
        self._state_dicts[kimg] = {
            key: value.detach() for key, value in network.state_dict().items()
        }
        if self._blend_net is None:
            self._blend_net = copy.deepcopy(network)
            # dict(state_dict()) captures live references to the target's
            # tensors, so writing into them mutates the blend network.
            self._blend_state = dict(self._blend_net.state_dict())
            logger.info("Weight blender allocated its blend target from kimg %d", kimg)

    def evict_snapshot(self, kimg: int) -> None:
        """Drop a snapshot's cached network and state_dict."""
        self._networks.pop(kimg, None)
        self._state_dicts.pop(kimg, None)

    def is_cached(self, kimg: int) -> bool:
        """Whether a snapshot's state_dict is currently cached."""
        return kimg in self._state_dicts

    def __call__(self, kimg_a: int, kimg_b: int, alpha: float) -> torch.nn.Module:
        """Return the network for blend factor ``alpha`` between two snapshots.

        Fast paths return a cached network directly: ``alpha == 0`` (or equal
        snapshots) yields the lower network, ``alpha == 1`` the upper. Otherwise
        the weights are lerped in place into the pre-allocated blend target.
        """
        if kimg_a == kimg_b or alpha == 0.0:
            return self._networks[kimg_a]
        if alpha == 1.0:
            return self._networks[kimg_b]
        return self._blend_into_target(kimg_a, kimg_b, alpha)

    def blend_into_target(
        self, kimg_a: int, kimg_b: int, alpha: float
    ) -> torch.nn.Module:
        """Blend a pair into the persistent target network and return it.

        Unlike ``__call__``, this never returns a cached network -- even at
        ``alpha`` 0 or 1 it writes the result into the pre-allocated blend
        target. The target's tensors therefore keep fixed addresses across
        frames, which is what lets a captured CUDA graph read freshly-blended
        weights on replay (``torch.lerp`` at alpha 0/1 reduces to copying the
        dominant side, so no special-casing is needed).
        """
        return self._blend_into_target(kimg_a, kimg_b, alpha)

    def _blend_into_target(
        self, kimg_a: int, kimg_b: int, alpha: float
    ) -> torch.nn.Module:
        sd_lower = self._state_dicts[kimg_a]
        sd_upper = self._state_dicts[kimg_b]
        assert self._blend_net is not None and self._blend_state is not None
        for key, dst in self._blend_state.items():
            lower = sd_lower[key]
            upper = sd_upper[key]
            if dst.is_floating_point():
                torch.lerp(lower, upper, alpha, out=dst)
            else:
                dst.copy_(lower if alpha < 0.5 else upper)
        return self._blend_net
