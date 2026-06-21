"""Naive linear interpolation from a [0, 1] position to a snapshot pair."""

from collections.abc import Sequence

from balagan.config import SnapshotInfo


class Interpolator:
    """Maps an audience position t in [0, 1] to a bracketing index pair and a
    [0, 1] blend factor.

    Snapshots are laid out at equal t intervals — snapshot i sits at
    ``t = i / (N - 1)``. Curation is the operator's job: whichever snapshots
    are in the run folder are the stops in t-space.
    """

    def __init__(self, snapshots: Sequence[SnapshotInfo]) -> None:
        if len(snapshots) < 2:
            raise ValueError("Interpolator requires at least two snapshots")
        self._indices = tuple(
            s.index for s in sorted(snapshots, key=lambda s: s.index)
        )

    def __call__(self, t: float) -> tuple[int, int, float]:
        """Return ``(index_a, index_b, blend)`` bracketing position ``t``."""
        t = min(1.0, max(0.0, t))
        position = t * (len(self._indices) - 1)
        bracket = min(int(position), len(self._indices) - 2)
        return self._indices[bracket], self._indices[bracket + 1], position - bracket
