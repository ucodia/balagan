"""Phase config loading, run-folder indexing, and validation for the BalaGAN engine."""

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SNAPSHOT_RE = re.compile(r"network-snapshot-(\d+)\.pkl")
_FID_METRIC = "fid50k_full"
_FID_JSONL_NAME = f"metric-{_FID_METRIC}.jsonl"
_T_TOLERANCE = 1e-9


class ConfigError(Exception):
    """Raised when a phase config or training run folder fails validation."""


@dataclass(frozen=True)
class Phase:
    """One contiguous segment of the trajectory, in both t- and kimg-space."""

    t_start: float
    t_end: float
    kimg_start: int
    kimg_end: int


@dataclass(frozen=True)
class PhaseConfig:
    """A parsed and structurally validated phase config JSON."""

    kimg_range: tuple[int, int]
    smoothing_window: int
    floor: float
    canonical_mapping_kimg: int
    phases: tuple[Phase, ...]


@dataclass(frozen=True)
class SnapshotInfo:
    """One indexed training snapshot: its kimg, raw FID, and .pkl location."""

    kimg: int
    fid_raw: float
    pkl_path: Path


@dataclass(frozen=True)
class EngineConfig:
    """A phase config validated against a concrete training run folder."""

    run_dir: Path
    phase_config: PhaseConfig
    snapshots: tuple[SnapshotInfo, ...]


def parse_phase_config(config_path: Path | str) -> PhaseConfig:
    """Load a phase config JSON file and validate its structural rules.

    Checks t-contiguity (phases meet, first starts at 0.0, last ends at 1.0),
    kimg-contiguity, and that kimg_range matches the first/last phase
    boundaries. Raises ConfigError with an actionable message on any violation.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"Phase config file not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Phase config {config_path} is not valid JSON: {exc}"
        ) from exc

    phase_config = _build_phase_config(raw, config_path)
    _validate_phase_structure(phase_config, config_path)
    return phase_config


def load_config(config_path: Path | str, run_dir: Path | str) -> EngineConfig:
    """Load and fully validate a phase config against a training run folder.

    Runs all structural checks from parse_phase_config, then reads the run
    folder to build the snapshot index and validate the run-folder-dependent
    rules: the canonical snapshot file must exist, and every phase must contain
    at least two snapshots with FID data.
    """
    phase_config = parse_phase_config(config_path)
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ConfigError(f"Run directory not found: {run_dir}")

    snapshots = _build_snapshot_index(run_dir, phase_config)
    _validate_run_folder(run_dir, phase_config, snapshots)
    logger.info(
        "Loaded config %s against run %s: %d snapshots indexed",
        config_path,
        run_dir,
        len(snapshots),
    )
    return EngineConfig(run_dir=run_dir, phase_config=phase_config, snapshots=snapshots)


def _require(mapping: Any, key: str, source: Path) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ConfigError(f"Phase config {source} is missing required key: '{key}'")
    return mapping[key]


def _build_phase_config(raw: Any, source: Path) -> PhaseConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"Phase config {source} must be a JSON object")
    try:
        kimg_range_raw = _require(raw, "kimg_range", source)
        kimg_range = (
            int(_require(kimg_range_raw, "start", source)),
            int(_require(kimg_range_raw, "end", source)),
        )
        smoothing_window = int(_require(raw, "smoothing_window", source))
        floor = float(_require(raw, "floor", source))
        canonical_mapping_kimg = int(_require(raw, "canonical_mapping_kimg", source))
        phases = tuple(
            Phase(
                t_start=float(_require(p, "t_start", source)),
                t_end=float(_require(p, "t_end", source)),
                kimg_start=int(_require(p, "kimg_start", source)),
                kimg_end=int(_require(p, "kimg_end", source)),
            )
            for p in _require(raw, "phases", source)
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Phase config {source} has a malformed value: {exc}"
        ) from exc

    if not phases:
        raise ConfigError(f"Phase config {source} defines no phases")
    return PhaseConfig(
        kimg_range, smoothing_window, floor, canonical_mapping_kimg, phases
    )


def _validate_phase_structure(cfg: PhaseConfig, source: Path) -> None:
    phases = cfg.phases
    if not math.isclose(phases[0].t_start, 0.0, abs_tol=_T_TOLERANCE):
        raise ConfigError(
            f"Phase config {source}: first phase t_start must be 0.0, "
            f"got {phases[0].t_start}"
        )
    if not math.isclose(phases[-1].t_end, 1.0, abs_tol=_T_TOLERANCE):
        raise ConfigError(
            f"Phase config {source}: last phase t_end must be 1.0, "
            f"got {phases[-1].t_end}"
        )
    for prev, curr in zip(phases, phases[1:]):
        if not math.isclose(curr.t_start, prev.t_end, abs_tol=_T_TOLERANCE):
            raise ConfigError(
                f"Phase config {source}: phases are not contiguous in t "
                f"(phase ends at {prev.t_end}, next starts at {curr.t_start})"
            )
        if curr.kimg_start != prev.kimg_end:
            raise ConfigError(
                f"Phase config {source}: phases are not contiguous in kimg "
                f"(phase ends at {prev.kimg_end}, next starts at {curr.kimg_start})"
            )
    if cfg.kimg_range[0] != phases[0].kimg_start:
        raise ConfigError(
            f"Phase config {source}: kimg_range.start ({cfg.kimg_range[0]}) must "
            f"equal the first phase kimg_start ({phases[0].kimg_start})"
        )
    if cfg.kimg_range[1] != phases[-1].kimg_end:
        raise ConfigError(
            f"Phase config {source}: kimg_range.end ({cfg.kimg_range[1]}) must "
            f"equal the last phase kimg_end ({phases[-1].kimg_end})"
        )


def _kimg_from_pkl_name(name: str) -> int | None:
    match = _SNAPSHOT_RE.fullmatch(name)
    return int(match.group(1)) if match else None


def _build_snapshot_index(
    run_dir: Path, phase_config: PhaseConfig
) -> tuple[SnapshotInfo, ...]:
    pkl_paths: dict[int, Path] = {}
    for path in run_dir.glob("network-snapshot-*.pkl"):
        kimg = _kimg_from_pkl_name(path.name)
        if kimg is not None:
            pkl_paths[kimg] = path

    fids = _parse_fid_jsonl(run_dir / _FID_JSONL_NAME)
    low, high = phase_config.kimg_range
    return tuple(
        SnapshotInfo(kimg=kimg, fid_raw=fids[kimg], pkl_path=pkl_paths[kimg])
        for kimg in sorted(pkl_paths)
        if kimg in fids and low <= kimg <= high
    )


def _parse_fid_jsonl(jsonl_path: Path) -> dict[int, float]:
    if not jsonl_path.is_file():
        raise ConfigError(f"FID metric file not found: {jsonl_path}")
    fids: dict[int, float] = {}
    for lineno, line in enumerate(jsonl_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            snapshot_pkl = record["snapshot_pkl"]
            fid = float(record["results"][_FID_METRIC])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Skipping malformed FID entry at %s:%d (%s)", jsonl_path, lineno, exc
            )
            continue
        if snapshot_pkl is None:
            continue
        kimg = _kimg_from_pkl_name(Path(snapshot_pkl).name)
        if kimg is not None:
            fids[kimg] = fid
    return fids


def _validate_run_folder(
    run_dir: Path, phase_config: PhaseConfig, snapshots: tuple[SnapshotInfo, ...]
) -> None:
    canonical_kimg = phase_config.canonical_mapping_kimg
    canonical_pkl = run_dir / f"network-snapshot-{canonical_kimg:06d}.pkl"
    if not canonical_pkl.is_file():
        raise ConfigError(
            f"canonical_mapping_kimg {canonical_kimg} has no snapshot file "
            f"(expected {canonical_pkl})"
        )
    for index, phase in enumerate(phase_config.phases):
        count = sum(
            1 for s in snapshots if phase.kimg_start <= s.kimg <= phase.kimg_end
        )
        if count < 2:
            raise ConfigError(
                f"Phase {index} (kimg {phase.kimg_start}-{phase.kimg_end}) has "
                f"{count} snapshot(s) with FID data; at least 2 are required"
            )
