"""Tests for balagan.config: run-folder scan and canonical defaulting."""

from pathlib import Path

import pytest

from balagan.config import (
    ConfigError,
    EngineConfig,
    SnapshotInfo,
    load_run,
)


def make_snapshots_dir(tmp_path: Path, names: list[str]) -> Path:
    """Create a synthetic run folder with empty .pkl files."""
    snapshots_dir = tmp_path / "run"
    snapshots_dir.mkdir()
    for name in names:
        (snapshots_dir / name).touch()
    return snapshots_dir


def test_load_run_indexes_snapshot_files_sorted_by_name(tmp_path):
    snapshots_dir = make_snapshots_dir(
        tmp_path, ["snap-500.pkl", "snap-020.pkl", "snap-200.pkl", "snap-100.pkl"]
    )
    cfg = load_run(snapshots_dir)
    assert isinstance(cfg, EngineConfig)
    # Sorted by filename: snap-020, snap-100, snap-200, snap-500
    assert [s.index for s in cfg.snapshots] == [0, 1, 2, 3]
    assert [s.pkl_path.name for s in cfg.snapshots] == [
        "snap-020.pkl", "snap-100.pkl", "snap-200.pkl", "snap-500.pkl"
    ]
    first = cfg.snapshots[0]
    assert isinstance(first, SnapshotInfo)
    assert first.index == 0
    assert first.pkl_path == snapshots_dir / "snap-020.pkl"


def test_canonical_defaults_to_middle_snapshot(tmp_path):
    # N=4 -> middle index = N // 2 = 2
    snapshots_dir = make_snapshots_dir(
        tmp_path, ["a.pkl", "b.pkl", "c.pkl", "d.pkl"]
    )
    cfg = load_run(snapshots_dir)
    assert cfg.canonical_index == 2


def test_canonical_index_override_is_honored(tmp_path):
    snapshots_dir = make_snapshots_dir(
        tmp_path, ["a.pkl", "b.pkl", "c.pkl", "d.pkl"]
    )
    cfg = load_run(snapshots_dir, canonical_index=1)
    assert cfg.canonical_index == 1


def test_canonical_index_override_out_of_range_raises(tmp_path):
    snapshots_dir = make_snapshots_dir(
        tmp_path, ["a.pkl", "b.pkl", "c.pkl", "d.pkl"]
    )
    with pytest.raises(ConfigError, match="out of range"):
        load_run(snapshots_dir, canonical_index=99)


def test_missing_snapshots_dir_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="Snapshots directory not found"):
        load_run(tmp_path / "does-not-exist")


def test_fewer_than_two_snapshots_raises_config_error(tmp_path):
    snapshots_dir = make_snapshots_dir(tmp_path, ["only.pkl"])
    with pytest.raises(ConfigError, match="at least 2"):
        load_run(snapshots_dir)


def test_non_pkl_files_are_ignored(tmp_path):
    snapshots_dir = make_snapshots_dir(
        tmp_path, ["a.pkl", "b.pkl"]
    )
    (snapshots_dir / "fakes000100.png").touch()
    (snapshots_dir / "metric-fid50k_full.jsonl").touch()
    (snapshots_dir / "training_options.json").touch()
    cfg = load_run(snapshots_dir)
    assert len(cfg.snapshots) == 2
    assert [s.index for s in cfg.snapshots] == [0, 1]
