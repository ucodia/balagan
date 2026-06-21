"""Run-folder scan: index snapshot .pkl files, default the canonical mapping snapshot."""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when a run folder fails validation."""


@dataclass(frozen=True)
class SnapshotInfo:
    """One indexed snapshot: its sort-order index and .pkl location."""

    index: int
    pkl_path: Path


@dataclass(frozen=True)
class EngineConfig:
    """A run folder with its indexed snapshots and the canonical mapping index."""

    snapshots_dir: Path
    canonical_index: int
    snapshots: tuple[SnapshotInfo, ...]


def load_run(
    snapshots_dir: Path | str, canonical_index: int | None = None
) -> EngineConfig:
    """Scan a run folder for .pkl files.

    Snapshots are sorted by filename. The canonical mapping snapshot defaults to
    the middle of the sorted list (``snapshots[len(snapshots) // 2]``); pass
    ``canonical_index`` to override with a 0-based index into the sorted list.

    Raises ``ConfigError`` if the directory is missing, contains fewer than
    two snapshots, or the override index is out of range.
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

    if canonical_index is None:
        canonical_index = len(snapshots) // 2
    elif not (0 <= canonical_index < len(snapshots)):
        raise ConfigError(
            f"canonical_index {canonical_index} is out of range for "
            f"{len(snapshots)} snapshots in {snapshots_dir}"
        )

    logger.info(
        "Loaded run %s: %d snapshots, canonical index %d (%s)",
        snapshots_dir,
        len(snapshots),
        canonical_index,
        snapshots[canonical_index].pkl_path.name,
    )
    return EngineConfig(
        snapshots_dir=snapshots_dir,
        canonical_index=canonical_index,
        snapshots=snapshots,
    )


def _build_snapshot_index(snapshots_dir: Path) -> tuple[SnapshotInfo, ...]:
    paths = sorted(snapshots_dir.glob("*.pkl"), key=lambda p: p.name)
    return tuple(SnapshotInfo(index=i, pkl_path=path) for i, path in enumerate(paths))
