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

    def cache_snapshot(self, index: int, network: torch.nn.Module) -> None:
        """Cache a snapshot's network and a reference to its state_dict, keyed by
        index. The first cached snapshot also seeds the pre-allocated blend target.
        """
        self._networks[index] = network
        self._state_dicts[index] = {
            key: value.detach() for key, value in network.state_dict().items()
        }
        if self._blend_net is None:
            self._blend_net = copy.deepcopy(network)
            # dict(state_dict()) captures live references to the target's
            # tensors, so writing into them mutates the blend network.
            self._blend_state = dict(self._blend_net.state_dict())
            logger.info("Weight blender allocated its blend target from index %d", index)

    def evict_snapshot(self, index: int) -> None:
        """Drop a snapshot's cached network and state_dict."""
        self._networks.pop(index, None)
        self._state_dicts.pop(index, None)

    def is_cached(self, index: int) -> bool:
        """Whether a snapshot's state_dict is currently cached."""
        return index in self._state_dicts

    def __call__(self, index_a: int, index_b: int, alpha: float) -> torch.nn.Module:
        """Return the network for blend factor ``alpha`` between two snapshots.

        Fast paths return a cached network directly: ``alpha == 0`` (or equal
        snapshots) yields the lower network, ``alpha == 1`` the upper. Otherwise
        the weights are lerped in place into the pre-allocated blend target.
        """
        if index_a == index_b or alpha == 0.0:
            return self._networks[index_a]
        if alpha == 1.0:
            return self._networks[index_b]

        sd_lower = self._state_dicts[index_a]
        sd_upper = self._state_dicts[index_b]
        assert self._blend_net is not None and self._blend_state is not None
        for key, dst in self._blend_state.items():
            lower = sd_lower[key]
            upper = sd_upper[key]
            if dst.is_floating_point():
                torch.lerp(lower, upper, alpha, out=dst)
            else:
                dst.copy_(lower if alpha < 0.5 else upper)
        return self._blend_net
