"""Tests for balagan.core.interpolator: phase-aware FID-weighted t mapping."""

from pathlib import Path

import pytest

from balagan.config import Phase, PhaseConfig, SnapshotInfo
from balagan.core.interpolator import Interpolator

# The american-nightmare phase layout (t-space and kimg-space boundaries).
AMERICAN_NIGHTMARE_PHASES = (
    Phase(0.0, 0.25, 20, 200),
    Phase(0.25, 0.5, 200, 500),
    Phase(0.5, 0.75, 500, 2540),
    Phase(0.75, 0.9, 2540, 2880),
    Phase(0.9, 1.0, 2880, 6400),
)


def snap(kimg: int, fid: float) -> SnapshotInfo:
    return SnapshotInfo(
        kimg=kimg, fid_raw=fid, pkl_path=Path(f"network-snapshot-{kimg:06d}.pkl")
    )


def make_phase_config(
    smoothing_window: int = 5,
    floor: float = 1.0,
    phases: tuple[Phase, ...] = AMERICAN_NIGHTMARE_PHASES,
) -> PhaseConfig:
    return PhaseConfig(
        kimg_range=(phases[0].kimg_start, phases[-1].kimg_end),
        smoothing_window=smoothing_window,
        floor=floor,
        canonical_mapping_kimg=2544,
        phases=phases,
    )


# Snapshots at every phase boundary plus intermediates; FID falls over training.
SNAPSHOTS = [
    snap(20, 320.0),
    snap(100, 280.0),
    snap(200, 210.0),
    snap(350, 165.0),
    snap(500, 130.0),
    snap(1500, 72.0),
    snap(2540, 50.0),
    snap(2700, 46.0),
    snap(2880, 42.0),
    snap(5000, 38.0),
    snap(6400, 36.0),
]


# --- the five spec-required test cases ---------------------------------------


def test_t_zero_returns_first_pair():
    interp = Interpolator(SNAPSHOTS, make_phase_config())
    assert interp(0.0) == (20, 100, 0.0)


def test_t_one_returns_last_pair():
    interp = Interpolator(SNAPSHOTS, make_phase_config())
    assert interp(1.0) in [(6400, 6400, 0.0), (5000, 6400, 1.0)]


def test_t_half_straddles_third_phase_boundary():
    interp = Interpolator(SNAPSHOTS, make_phase_config())
    kimg_a, kimg_b, blend = interp(0.5)
    # Phase boundaries in t are 0.0, 0.25, 0.5, ...; the third is kimg 500.
    assert kimg_a <= 500 <= kimg_b
    assert 0.0 <= blend <= 1.0


def test_t_quarter_matches_phase_boundary_snapshot():
    interp = Interpolator(SNAPSHOTS, make_phase_config())
    kimg_a, kimg_b, blend = interp(0.25)
    # t=0.25 is phase boundary kimg 200: (200, 200, 0.0) or an adjacent pair
    # that resolves exactly to kimg 200.
    resolves_to_200 = (
        (kimg_a == 200 and kimg_b == 200)
        or (kimg_a == 200 and blend == 0.0)
        or (kimg_b == 200 and blend == 1.0)
    )
    assert resolves_to_200, (kimg_a, kimg_b, blend)


def test_snap_a_kimg_non_decreasing_for_increasing_t():
    interp = Interpolator(SNAPSHOTS, make_phase_config())
    snap_a = [interp(i / 200)[0] for i in range(201)]
    assert snap_a == sorted(snap_a)


# --- stated guarantees -------------------------------------------------------


def test_blend_always_within_unit_interval():
    interp = Interpolator(SNAPSHOTS, make_phase_config())
    for i in range(201):
        _, _, blend = interp(i / 200)
        assert 0.0 <= blend <= 1.0


def test_phase_boundaries_map_to_boundary_kimgs():
    interp = Interpolator(SNAPSHOTS, make_phase_config())
    for t, boundary_kimg in [
        (0.0, 20),
        (0.25, 200),
        (0.5, 500),
        (0.75, 2540),
        (0.9, 2880),
    ]:
        kimg_a, _, blend = interp(t)
        assert kimg_a == boundary_kimg and blend == 0.0


def test_flat_fid_phase_spaces_snapshots_evenly():
    # When every snapshot in a phase shares an FID, the floor forces equal
    # weights, so t-coordinates are spaced uniformly across the phase.
    config = make_phase_config(
        smoothing_window=1, floor=1.0, phases=(Phase(0.0, 1.0, 0, 100),)
    )
    snaps = [snap(kimg, 5.0) for kimg in (0, 25, 50, 75, 100)]
    interp = Interpolator(snaps, config)
    for t, expected_kimg_a in [(0.0, 0), (0.25, 25), (0.5, 50), (0.75, 75)]:
        kimg_a, _, blend = interp(t)
        assert kimg_a == expected_kimg_a and blend == 0.0


def test_requires_at_least_two_snapshots():
    with pytest.raises(ValueError):
        Interpolator([snap(20, 100.0)], make_phase_config())


def test_phase_without_two_snapshots_raises():
    # Phase 5 (kimg 2880-6400) is left with a single snapshot.
    snaps = [snap(k, 100.0) for k in (20, 100, 200, 350, 500, 1500, 2540, 2700, 2880)]
    with pytest.raises(ValueError):
        Interpolator(snaps, make_phase_config())
