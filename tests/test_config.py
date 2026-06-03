"""Tests for balagan.config: run-folder scan and canonical defaulting."""

from pathlib import Path

import pytest

from balagan.config import (
    ConfigError,
    EngineConfig,
    SnapshotInfo,
    load_run,
)


def make_snapshots_dir(tmp_path: Path, kimgs: list[int]) -> Path:
    """Create a synthetic run folder: empty network-snapshot-*.pkl files."""
    snapshots_dir = tmp_path / "run"
    snapshots_dir.mkdir()
    for kimg in kimgs:
        (snapshots_dir / f"network-snapshot-{kimg:06d}.pkl").touch()
    return snapshots_dir


def test_load_run_indexes_snapshot_files_sorted_by_kimg(tmp_path):
    snapshots_dir = make_snapshots_dir(tmp_path, [500, 20, 200, 100])
    cfg = load_run(snapshots_dir)
    assert isinstance(cfg, EngineConfig)
    assert [s.kimg for s in cfg.snapshots] == [20, 100, 200, 500]
    first = cfg.snapshots[0]
    assert isinstance(first, SnapshotInfo)
    assert first.kimg == 20
    assert first.pkl_path == snapshots_dir / "network-snapshot-000020.pkl"


def test_canonical_defaults_to_middle_snapshot(tmp_path):
    # N=4 -> middle index = N // 2 = 2 -> snapshot kimg 300.
    snapshots_dir = make_snapshots_dir(tmp_path, [100, 200, 300, 400])
    cfg = load_run(snapshots_dir)
    assert cfg.canonical_mapping_kimg == 300


def test_canonical_kimg_override_is_honored(tmp_path):
    snapshots_dir = make_snapshots_dir(tmp_path, [100, 200, 300, 400])
    cfg = load_run(snapshots_dir, canonical_kimg=200)
    assert cfg.canonical_mapping_kimg == 200


def test_canonical_kimg_override_must_match_an_indexed_snapshot(tmp_path):
    snapshots_dir = make_snapshots_dir(tmp_path, [100, 200, 300, 400])
    with pytest.raises(ConfigError, match="no matching snapshot"):
        load_run(snapshots_dir, canonical_kimg=999)


def test_missing_snapshots_dir_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="Snapshots directory not found"):
        load_run(tmp_path / "does-not-exist")


def test_fewer_than_two_snapshots_raises_config_error(tmp_path):
    snapshots_dir = make_snapshots_dir(tmp_path, [100])
    with pytest.raises(ConfigError, match="at least 2"):
        load_run(snapshots_dir)


def test_non_snapshot_files_are_ignored(tmp_path):
    snapshots_dir = make_snapshots_dir(tmp_path, [100, 200])
    (snapshots_dir / "fakes000100.png").touch()
    (snapshots_dir / "metric-fid50k_full.jsonl").touch()
    (snapshots_dir / "training_options.json").touch()
    cfg = load_run(snapshots_dir)
    assert [s.kimg for s in cfg.snapshots] == [100, 200]
