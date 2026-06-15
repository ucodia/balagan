"""Tests for balagan.core.weight_blender: in-place weight-space blending."""

import torch

from balagan.core.weight_blender import WeightBlender


class StubNet(torch.nn.Module):
    """Minimal network with float parameters plus a non-float buffer."""

    def __init__(self, value: float, batches: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((2, 2), value))
        self.bias = torch.nn.Parameter(torch.full((3,), value))
        self.register_buffer("num_batches_tracked", torch.tensor(batches))


def make_blender_with_pair(
    value_a=1.0, value_b=3.0, batches_a=10, batches_b=20
) -> tuple[WeightBlender, StubNet, StubNet]:
    net_a = StubNet(value_a, batches_a)
    net_b = StubNet(value_b, batches_b)
    blender = WeightBlender()
    blender.cache_snapshot(100, net_a)
    blender.cache_snapshot(200, net_b)
    return blender, net_a, net_b


def test_blend_lerps_float_parameters_at_midpoint():
    blender, net_a, net_b = make_blender_with_pair()
    blended = blender(100, 200, 0.5)
    assert torch.allclose(blended.weight, torch.full((2, 2), 2.0))
    assert torch.allclose(blended.bias, torch.full((3,), 2.0))
    # the cached source networks must be left untouched
    assert torch.allclose(net_a.weight, torch.full((2, 2), 1.0))
    assert torch.allclose(net_b.weight, torch.full((2, 2), 3.0))


def test_non_float_buffer_copied_from_lower_side():
    blender, _, _ = make_blender_with_pair(batches_a=10, batches_b=20)
    blended = blender(100, 200, 0.3)  # alpha < 0.5 -> dominant side is lower
    assert blended.num_batches_tracked.item() == 10


def test_non_float_buffer_copied_from_upper_side():
    blender, _, _ = make_blender_with_pair(batches_a=10, batches_b=20)
    blended = blender(100, 200, 0.7)  # alpha >= 0.5 -> dominant side is upper
    assert blended.num_batches_tracked.item() == 20


def test_alpha_zero_returns_lower_network_directly():
    blender, net_a, _ = make_blender_with_pair()
    assert blender(100, 200, 0.0) is net_a


def test_alpha_one_returns_upper_network_directly():
    blender, _, net_b = make_blender_with_pair()
    assert blender(100, 200, 1.0) is net_b


def test_equal_kimgs_returns_that_network_directly():
    blender, net_a, _ = make_blender_with_pair()
    assert blender(100, 100, 0.5) is net_a


def test_blend_reuses_one_preallocated_target_in_place():
    blender, net_a, net_b = make_blender_with_pair()
    first = blender(100, 200, 0.5)
    second = blender(100, 200, 0.7)
    assert first is second  # one pre-allocated blend target, reused
    assert first is not net_a and first is not net_b
    # the second call overwrote the same target in place
    assert torch.allclose(first.weight, torch.full((2, 2), 2.4))


def test_evict_snapshot_drops_the_cache_entry():
    blender, _, _ = make_blender_with_pair()
    assert blender.is_cached(100) and blender.is_cached(200)
    blender.evict_snapshot(100)
    assert not blender.is_cached(100)
    assert blender.is_cached(200)


def test_blend_into_target_never_returns_a_cached_network():
    """Graph mode needs the persistent target every call -- even at alpha 0/1,
    where __call__ would short-circuit to a cached net -- so its tensors keep
    fixed addresses for CUDA-graph replay."""
    blender, net_a, net_b = make_blender_with_pair()
    at_zero = blender.blend_into_target(100, 200, 0.0)
    at_one = blender.blend_into_target(100, 200, 1.0)
    assert at_zero is at_one  # same pre-allocated target both times
    assert at_zero is not net_a and at_zero is not net_b


def test_blend_into_target_matches_the_source_sides_at_alpha_extremes():
    """torch.lerp at alpha 0/1 reduces to copying a side, so the target holds
    that snapshot's exact weights without special-casing."""
    blender, _, _ = make_blender_with_pair(value_a=1.0, value_b=3.0)
    at_zero = blender.blend_into_target(100, 200, 0.0)
    assert torch.allclose(at_zero.weight, torch.full((2, 2), 1.0))
    at_one = blender.blend_into_target(100, 200, 1.0)
    assert torch.allclose(at_one.weight, torch.full((2, 2), 3.0))


def test_blends_the_requested_pair_among_several():
    blender = WeightBlender()
    blender.cache_snapshot(100, StubNet(1.0, 0))
    blender.cache_snapshot(200, StubNet(5.0, 0))
    blender.cache_snapshot(300, StubNet(9.0, 0))
    blended = blender(100, 300, 0.5)  # blends 1.0 and 9.0, ignores kimg 200
    assert torch.allclose(blended.weight, torch.full((2, 2), 5.0))
