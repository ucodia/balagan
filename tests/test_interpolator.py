"""Tests for balagan.core.interpolator: naive linear t-mapping."""

from pathlib import Path

import pytest

from balagan.config import SnapshotInfo
from balagan.core.interpolator import Interpolator


def snap(kimg: int) -> SnapshotInfo:
    return SnapshotInfo(kimg=kimg, pkl_path=Path(f"network-snapshot-{kimg:06d}.pkl"))


def test_requires_at_least_two_snapshots():
    with pytest.raises(ValueError):
        Interpolator([snap(20)])


def test_t_zero_returns_the_first_pair():
    interp = Interpolator([snap(20), snap(100), snap(500)])
    assert interp(0.0) == (20, 100, 0.0)


def test_t_one_returns_the_last_pair_at_full_blend():
    interp = Interpolator([snap(20), snap(100), snap(500)])
    assert interp(1.0) == (100, 500, 1.0)


def test_snapshot_i_sits_at_t_equal_i_over_n_minus_one():
    snaps = [snap(k) for k in (10, 20, 30, 40, 50)]
    interp = Interpolator(snaps)
    for i, expected_kimg in enumerate((10, 20, 30, 40)):
        kimg_a, _, blend = interp(i / 4)
        assert kimg_a == expected_kimg
        assert blend == pytest.approx(0.0, abs=1e-9)


def test_blend_interpolates_within_a_bracket():
    interp = Interpolator([snap(0), snap(100), snap(200)])  # stops at t = 0, 0.5, 1.0
    kimg_a, kimg_b, blend = interp(0.25)  # halfway through the first bracket
    assert (kimg_a, kimg_b) == (0, 100)
    assert blend == pytest.approx(0.5)


def test_blend_stays_in_unit_interval():
    interp = Interpolator([snap(k) for k in (10, 20, 30, 40, 50)])
    for i in range(201):
        _, _, blend = interp(i / 200)
        assert 0.0 <= blend <= 1.0


def test_snap_a_kimg_is_non_decreasing_for_increasing_t():
    interp = Interpolator([snap(k) for k in (10, 20, 30, 40, 50)])
    snap_as = [interp(i / 200)[0] for i in range(201)]
    assert snap_as == sorted(snap_as)


def test_unsorted_input_is_handled_in_sorted_order():
    snaps = [snap(50), snap(20), snap(30), snap(40), snap(10)]
    interp = Interpolator(snaps)
    assert interp(0.0) == (10, 20, 0.0)
    assert interp(1.0) == (40, 50, 1.0)


def test_out_of_range_t_is_clamped():
    interp = Interpolator([snap(0), snap(100)])
    assert interp(-0.5) == (0, 100, 0.0)
    assert interp(1.5) == (0, 100, 1.0)
