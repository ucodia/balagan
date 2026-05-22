"""Tests for balagan.core.snapshot_manager: rolling window + loader thread."""

import time
from pathlib import Path

from balagan.config import SnapshotInfo
from balagan.core.snapshot_manager import SnapshotManager, _compute_window


class FakeSynthesis:
    """Stand-in for a loaded synthesis network; records its source path."""

    def __init__(self, pkl_path: Path):
        self.pkl_path = pkl_path


def fake_loader(pkl_path: Path) -> FakeSynthesis:
    return FakeSynthesis(pkl_path)


def make_snapshots(count: int) -> list[SnapshotInfo]:
    return [
        SnapshotInfo(
            kimg=index * 100,
            fid_raw=float(count - index),
            pkl_path=Path(f"network-snapshot-{index * 100:06d}.pkl"),
        )
        for index in range(count)
    ]


# --- _compute_window: the spec's required cases ------------------------------


def test_window_canonical_example():
    assert _compute_window(20, 6, 18, 19, 8) == {6, 13, 14, 15, 16, 17, 18, 19}


def test_window_pair_at_left_edge():
    assert _compute_window(20, 6, 0, 1, 8) == {0, 1, 2, 3, 4, 5, 6, 7}


def test_window_pair_in_middle():
    assert _compute_window(20, 6, 10, 11, 8) == {6, 8, 9, 10, 11, 12, 13, 14}


def test_window_canonical_coincides_with_pair_member():
    window = _compute_window(20, 6, 5, 6, 8)
    assert len(window) == 8  # canonical (index 6) must not be double-counted
    assert {5, 6}.issubset(window)
    assert window == {2, 3, 4, 5, 6, 7, 8, 9}


# --- _compute_window: other cases --------------------------------------------


def test_window_smaller_than_window_size_returns_all_snapshots():
    assert _compute_window(5, 2, 3, 4, 8) == {0, 1, 2, 3, 4}


def test_window_without_canonical_still_fills_window():
    window = _compute_window(20, None, 10, 11, 8)
    assert len(window) == 8
    assert {10, 11}.issubset(window)


# --- SnapshotManager ---------------------------------------------------------


def test_prime_loads_the_window_synchronously():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader, window_size=8
    )
    manager.prime(1800, 1900)  # indices 18 and 19, canonical index 6
    for kimg in (600, 1300, 1400, 1500, 1600, 1700, 1800, 1900):
        assert manager.get_synthesis(kimg) is not None
    assert manager.get_synthesis(0) is None  # index 0 is outside the window


def test_get_synthesis_returns_none_before_loading():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader
    )
    assert manager.get_synthesis(1800) is None


def test_is_pair_ready_reflects_loaded_state():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader, window_size=8
    )
    assert not manager.is_pair_ready(1800, 1900)
    manager.prime(1800, 1900)
    assert manager.is_pair_ready(1800, 1900)


def test_prime_evicts_snapshots_outside_the_new_window():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader, window_size=8
    )
    manager.prime(1800, 1900)
    assert manager.get_synthesis(1700) is not None
    manager.prime(0, 100)  # jump to the opposite edge
    assert manager.get_synthesis(1700) is None  # evicted
    assert manager.get_synthesis(0) is not None
    assert manager.get_synthesis(100) is not None


def test_loaded_network_matches_its_snapshot_path():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader, window_size=8
    )
    manager.prime(1800, 1900)
    assert manager.get_synthesis(1800).pkl_path == Path("network-snapshot-001800.pkl")


def test_background_thread_loads_after_set_active_pair():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader, window_size=8
    )
    manager.start()
    try:
        manager.set_active_pair(1800, 1900)
        deadline = time.monotonic() + 5.0
        while not manager.is_pair_ready(1800, 1900) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert manager.is_pair_ready(1800, 1900)
    finally:
        manager.stop()


def test_loaded_kimgs_reports_the_resident_snapshots():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader, window_size=8
    )
    assert manager.loaded_kimgs() == set()
    manager.prime(1800, 1900)
    assert manager.loaded_kimgs() == {600, 1300, 1400, 1500, 1600, 1700, 1800, 1900}


def test_pending_count_reports_snapshots_awaiting_load():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader, window_size=8
    )
    manager.prime(0, 100)
    manager.set_active_pair(1800, 1900)  # desired jumps; loader thread not started
    assert manager.pending_count() > 0


def test_loaded_networks_returns_the_resident_networks():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader, window_size=8
    )
    manager.prime(1800, 1900)
    networks = manager.loaded_networks()
    assert set(networks) == manager.loaded_kimgs()
    assert networks[1800].pkl_path == Path("network-snapshot-001800.pkl")


def test_loaded_networks_returns_a_copy_isolated_from_later_eviction():
    manager = SnapshotManager(
        make_snapshots(20), canonical_kimg=600, loader=fake_loader, window_size=8
    )
    manager.prime(1800, 1900)
    networks = manager.loaded_networks()
    manager.prime(0, 100)  # jump to the opposite edge, evicting the far window
    assert 1800 in networks  # the earlier caller's view is unaffected
    assert manager.get_synthesis(1800) is None  # though the manager has evicted it


def test_window_size_zero_keeps_every_snapshot_resident():
    snapshots = make_snapshots(20)
    manager = SnapshotManager(
        snapshots, canonical_kimg=600, loader=fake_loader, window_size=0
    )
    manager.prime(0, 100)
    assert manager.loaded_kimgs() == {snap.kimg for snap in snapshots}
