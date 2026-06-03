"""Naive linear interpolation from a [0, 1] position to a snapshot pair."""

from collections.abc import Sequence

from balagan.config import SnapshotInfo


class Interpolator:
    """Maps an audience position t in [0, 1] to a bracketing kimg pair and a
    [0, 1] blend factor.

    Snapshots are sorted by kimg and laid out at equal t intervals -- snapshot
    i sits at ``t = i / (N - 1)``. Curation is the operator's job: whichever
    snapshots are in the run folder are the stops in t-space.
    """

    def __init__(self, snapshots: Sequence[SnapshotInfo]) -> None:
        if len(snapshots) < 2:
            raise ValueError("Interpolator requires at least two snapshots")
        self._kimgs = tuple(
            s.kimg for s in sorted(snapshots, key=lambda s: s.kimg)
        )

    def __call__(self, t: float) -> tuple[int, int, float]:
        """Return ``(kimg_a, kimg_b, blend)`` bracketing position ``t``."""
        t = min(1.0, max(0.0, t))
        position = t * (len(self._kimgs) - 1)
        bracket = min(int(position), len(self._kimgs) - 2)
        return self._kimgs[bracket], self._kimgs[bracket + 1], position - bracket
