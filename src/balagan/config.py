"""Run-folder scan: index snapshot .pkl files, default the canonical mapping snapshot."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SNAPSHOT_RE = re.compile(r"network-snapshot-(\d+)\.pkl")


class ConfigError(Exception):
    """Raised when a run folder fails validation."""


@dataclass(frozen=True)
class SnapshotInfo:
    """One indexed training snapshot: its kimg and .pkl location."""

    kimg: int
    pkl_path: Path


@dataclass(frozen=True)
class EngineConfig:
    """A run folder with its indexed snapshots and the canonical mapping kimg."""

    snapshots_dir: Path
    canonical_mapping_kimg: int
    snapshots: tuple[SnapshotInfo, ...]


def load_run(
    snapshots_dir: Path | str, canonical_kimg: int | None = None
) -> EngineConfig:
    """Scan a run folder for network-snapshot-*.pkl files.

    Snapshots are sorted by kimg. The canonical mapping snapshot defaults to
    the middle of the sorted list (``snapshots[len(snapshots) // 2]``); pass
    ``canonical_kimg`` to override. The override must correspond to an
    indexed snapshot.

    Raises ``ConfigError`` if the directory is missing, contains fewer than
    two snapshots, or the override doesn't match any snapshot file.
    """
    snapshots_dir = Path(snapshots_dir)
    if not snapshots_dir.is_dir():
        raise ConfigError(f"Snapshots directory not found: {snapshots_dir}")

    snapshots = _build_snapshot_index(snapshots_dir)
    if len(snapshots) < 2:
        raise ConfigError(
            f"Snapshots directory {snapshots_dir} has {len(snapshots)} snapshot file(s); "
            f"at least 2 are required"
        )

    if canonical_kimg is None:
        canonical_kimg = snapshots[len(snapshots) // 2].kimg
    elif not any(s.kimg == canonical_kimg for s in snapshots):
        raise ConfigError(
            f"canonical_kimg {canonical_kimg} has no matching snapshot file "
            f"in {snapshots_dir}"
        )

    logger.info(
        "Loaded run %s: %d snapshots, canonical kimg %d",
        snapshots_dir,
        len(snapshots),
        canonical_kimg,
    )
    return EngineConfig(
        snapshots_dir=snapshots_dir,
        canonical_mapping_kimg=canonical_kimg,
        snapshots=snapshots,
    )


def _build_snapshot_index(snapshots_dir: Path) -> tuple[SnapshotInfo, ...]:
    snapshots: list[SnapshotInfo] = []
    for path in snapshots_dir.glob("network-snapshot-*.pkl"):
        match = _SNAPSHOT_RE.fullmatch(path.name)
        if match:
            snapshots.append(SnapshotInfo(kimg=int(match.group(1)), pkl_path=path))
    return tuple(sorted(snapshots, key=lambda s: s.kimg))
