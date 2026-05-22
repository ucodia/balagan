"""Tests for balagan.config: phase config parsing and validation."""

import json
from pathlib import Path

import pytest

from balagan.config import (
    ConfigError,
    EngineConfig,
    Phase,
    PhaseConfig,
    SnapshotInfo,
    load_config,
    parse_phase_config,
)

# A run folder with >=2 snapshots in every phase of the valid config, including
# the canonical snapshot at kimg 2544.
VALID_RUN_KIMGS = [20, 100, 200, 350, 500, 1500, 2540, 2544, 2700, 2880, 5000, 6400]
VALID_FIDS = {kimg: float(100 - i) for i, kimg in enumerate(VALID_RUN_KIMGS)}


def valid_config_dict() -> dict:
    return {
        "kimg_range": {"start": 20, "end": 6400},
        "smoothing_window": 5,
        "floor": 1.0,
        "canonical_mapping_kimg": 2544,
        "phases": [
            {"t_start": 0.0, "t_end": 0.25, "kimg_start": 20, "kimg_end": 200},
            {"t_start": 0.25, "t_end": 0.5, "kimg_start": 200, "kimg_end": 500},
            {"t_start": 0.5, "t_end": 0.75, "kimg_start": 500, "kimg_end": 2540},
            {"t_start": 0.75, "t_end": 0.9, "kimg_start": 2540, "kimg_end": 2880},
            {"t_start": 0.9, "t_end": 1.0, "kimg_start": 2880, "kimg_end": 6400},
        ],
    }


def write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def write_run_dir(
    tmp_path: Path, pkl_kimgs: list[int], fid_by_kimg: dict[int, float]
) -> Path:
    """Create a synthetic run folder: empty snapshot .pkl files plus a
    metric-fid50k_full.jsonl in the StyleGAN2-ADA format."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    for kimg in pkl_kimgs:
        (run_dir / f"network-snapshot-{kimg:06d}.pkl").touch()
    lines = [
        json.dumps(
            {
                "results": {"fid50k_full": fid},
                "metric": "fid50k_full",
                "snapshot_pkl": f"network-snapshot-{kimg:06d}.pkl",
                "timestamp": 0.0,
            }
        )
        for kimg, fid in fid_by_kimg.items()
    ]
    (run_dir / "metric-fid50k_full.jsonl").write_text("\n".join(lines) + "\n")
    return run_dir


# --- parse_phase_config: structural validation -------------------------------


def test_parse_valid_config_returns_phase_config(tmp_path):
    cfg = parse_phase_config(write_config(tmp_path, valid_config_dict()))
    assert isinstance(cfg, PhaseConfig)
    assert cfg.kimg_range == (20, 6400)
    assert cfg.smoothing_window == 5
    assert cfg.floor == 1.0
    assert cfg.canonical_mapping_kimg == 2544
    assert len(cfg.phases) == 5
    assert cfg.phases[0] == Phase(t_start=0.0, t_end=0.25, kimg_start=20, kimg_end=200)
    assert cfg.phases[-1].t_end == 1.0


def test_first_phase_t_start_must_be_zero(tmp_path):
    bad = valid_config_dict()
    bad["phases"][0]["t_start"] = 0.05
    with pytest.raises(ConfigError):
        parse_phase_config(write_config(tmp_path, bad))


def test_last_phase_t_end_must_be_one(tmp_path):
    bad = valid_config_dict()
    bad["phases"][-1]["t_end"] = 0.95
    with pytest.raises(ConfigError):
        parse_phase_config(write_config(tmp_path, bad))


def test_phases_must_be_contiguous_in_t(tmp_path):
    bad = valid_config_dict()
    bad["phases"][2]["t_start"] = 0.55  # previous phase t_end is 0.5
    with pytest.raises(ConfigError):
        parse_phase_config(write_config(tmp_path, bad))


def test_phases_must_be_contiguous_in_kimg(tmp_path):
    bad = valid_config_dict()
    bad["phases"][2]["kimg_start"] = 550  # previous phase kimg_end is 500
    with pytest.raises(ConfigError):
        parse_phase_config(write_config(tmp_path, bad))


def test_kimg_range_start_must_match_first_phase(tmp_path):
    bad = valid_config_dict()
    bad["kimg_range"]["start"] = 0  # first phase kimg_start is 20
    with pytest.raises(ConfigError):
        parse_phase_config(write_config(tmp_path, bad))


def test_kimg_range_end_must_match_last_phase(tmp_path):
    bad = valid_config_dict()
    bad["kimg_range"]["end"] = 9999  # last phase kimg_end is 6400
    with pytest.raises(ConfigError):
        parse_phase_config(write_config(tmp_path, bad))


def test_malformed_json_raises_config_error(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{ not valid json")
    with pytest.raises(ConfigError):
        parse_phase_config(path)


def test_missing_required_key_raises_config_error(tmp_path):
    bad = valid_config_dict()
    del bad["canonical_mapping_kimg"]
    with pytest.raises(ConfigError):
        parse_phase_config(write_config(tmp_path, bad))


def test_missing_config_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        parse_phase_config(tmp_path / "does-not-exist.json")


# --- load_config: run folder indexing + full validation ----------------------


def test_load_config_builds_snapshot_index(tmp_path):
    config_path = write_config(tmp_path, valid_config_dict())
    run_dir = write_run_dir(tmp_path, VALID_RUN_KIMGS, VALID_FIDS)
    cfg = load_config(config_path, run_dir)
    assert isinstance(cfg, EngineConfig)
    assert [s.kimg for s in cfg.snapshots] == sorted(VALID_RUN_KIMGS)
    first = cfg.snapshots[0]
    assert isinstance(first, SnapshotInfo)
    assert first.kimg == 20
    assert first.fid_raw == VALID_FIDS[20]
    assert first.pkl_path == run_dir / "network-snapshot-000020.pkl"


def test_canonical_snapshot_file_must_exist(tmp_path):
    config_path = write_config(tmp_path, valid_config_dict())
    kimgs = [k for k in VALID_RUN_KIMGS if k != 2544]
    fids = {k: v for k, v in VALID_FIDS.items() if k != 2544}
    run_dir = write_run_dir(tmp_path, kimgs, fids)
    with pytest.raises(ConfigError):
        load_config(config_path, run_dir)


def test_phase_with_fewer_than_two_snapshots_raises(tmp_path):
    config_path = write_config(tmp_path, valid_config_dict())
    # Phases 1, 2, 4, 5 keep >=2 snapshots; phase 3 (kimg 500-2540) gets none.
    kimgs = [20, 100, 200, 300, 2544, 2700, 2800, 3000, 6400]
    run_dir = write_run_dir(tmp_path, kimgs, {k: 50.0 for k in kimgs})
    with pytest.raises(ConfigError):
        load_config(config_path, run_dir)


def test_snapshots_without_fid_are_excluded(tmp_path):
    config_path = write_config(tmp_path, valid_config_dict())
    run_dir = write_run_dir(tmp_path, VALID_RUN_KIMGS, VALID_FIDS)
    (run_dir / "network-snapshot-000300.pkl").touch()  # pkl present, no FID entry
    cfg = load_config(config_path, run_dir)
    assert 300 not in [s.kimg for s in cfg.snapshots]


def test_snapshots_outside_kimg_range_are_excluded(tmp_path):
    config_path = write_config(tmp_path, valid_config_dict())
    kimgs = VALID_RUN_KIMGS + [10, 8000]
    fids = {**VALID_FIDS, 10: 200.0, 8000: 5.0}
    run_dir = write_run_dir(tmp_path, kimgs, fids)
    cfg = load_config(config_path, run_dir)
    index_kimgs = [s.kimg for s in cfg.snapshots]
    assert 10 not in index_kimgs
    assert 8000 not in index_kimgs


def test_blank_lines_in_fid_jsonl_are_ignored(tmp_path):
    config_path = write_config(tmp_path, valid_config_dict())
    run_dir = write_run_dir(tmp_path, VALID_RUN_KIMGS, VALID_FIDS)
    jsonl = run_dir / "metric-fid50k_full.jsonl"
    jsonl.write_text(jsonl.read_text() + "\n   \n")
    cfg = load_config(config_path, run_dir)
    assert len(cfg.snapshots) == len(VALID_RUN_KIMGS)


def test_missing_fid_jsonl_raises_config_error(tmp_path):
    config_path = write_config(tmp_path, valid_config_dict())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for kimg in VALID_RUN_KIMGS:
        (run_dir / f"network-snapshot-{kimg:06d}.pkl").touch()
    with pytest.raises(ConfigError):
        load_config(config_path, run_dir)
