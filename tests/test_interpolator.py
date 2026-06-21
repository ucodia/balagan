"""Tests for balagan.core.interpolator: naive linear t-mapping."""

from pathlib import Path

import pytest

from balagan.config import SnapshotInfo
from balagan.core.interpolator import Interpolator


def snap(index: int) -> SnapshotInfo:
    return SnapshotInfo(index=index, pkl_path=Path(f"snap-{index:04d}.pkl"))


def test_requires_at_least_two_snapshots():
    with pytest.raises(ValueError):
        Interpolator([snap(20)])


def test_t_zero_returns_the_first_pair():
    interp = Interpolator([snap(0), snap(1), snap(2)])
    assert interp(0.0) == (0, 1, 0.0)


def test_t_one_returns_the_last_pair_at_full_blend():
    interp = Interpolator([snap(0), snap(1), snap(2)])
    assert interp(1.0) == (1, 2, 1.0)


def test_snapshot_i_sits_at_t_equal_i_over_n_minus_one():
    snaps = [snap(i) for i in range(5)]
    interp = Interpolator(snaps)
    for i, expected_index in enumerate((0, 1, 2, 3)):
        index_a, _, blend = interp(i / 4)
        assert index_a == expected_index
        assert blend == pytest.approx(0.0, abs=1e-9)


def test_blend_interpolates_within_a_bracket():
    interp = Interpolator([snap(0), snap(1), snap(2)])  # stops at t = 0, 0.5, 1.0
    index_a, index_b, blend = interp(0.25)  # halfway through the first bracket
    assert (index_a, index_b) == (0, 1)
    assert blend == pytest.approx(0.5)


def test_blend_stays_in_unit_interval():
    interp = Interpolator([snap(i) for i in range(5)])
    for i in range(201):
        _, _, blend = interp(i / 200)
        assert 0.0 <= blend <= 1.0


def test_snap_a_index_is_non_decreasing_for_increasing_t():
    interp = Interpolator([snap(i) for i in range(5)])
    snap_as = [interp(i / 200)[0] for i in range(201)]
    assert snap_as == sorted(snap_as)


def test_unsorted_input_is_handled_in_sorted_order():
    snaps = [snap(4), snap(1), snap(2), snap(3), snap(0)]
    interp = Interpolator(snaps)
    assert interp(0.0) == (0, 1, 0.0)
    assert interp(1.0) == (3, 4, 1.0)


def test_out_of_range_t_is_clamped():
    interp = Interpolator([snap(0), snap(1)])
    assert interp(-0.5) == (0, 1, 0.0)
    assert interp(1.5) == (0, 1, 1.0)
