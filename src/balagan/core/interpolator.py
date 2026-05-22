"""Phase-aware, FID-weighted interpolation from a [0, 1] position to a snapshot pair."""

import bisect
import logging
from collections.abc import Sequence
from itertools import accumulate

from balagan.config import PhaseConfig, SnapshotInfo

logger = logging.getLogger(__name__)


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Centered moving average; the window shrinks at the sequence edges."""
    if window <= 1:
        return list(values)
    half = window // 2
    n = len(values)
    smoothed = []
    for i in range(n):
        low = max(0, i - half)
        high = min(n, i + half + 1)
        smoothed.append(sum(values[low:high]) / (high - low))
    return smoothed


def _build_t_coords(
    kimgs: list[int], smoothed: list[float], phase_config: PhaseConfig
) -> dict[int, float]:
    """Assign every snapshot a global t coordinate.

    Within a phase, the interval between consecutive snapshots is weighted by
    the absolute smoothed-FID delta, floored so flat-FID stretches still
    advance. Cumulative weights are normalized to span the phase's t range. The
    first and last snapshot of a phase are pinned exactly to the phase bounds,
    which keeps a boundary snapshot (shared by two adjacent phases) consistent.
    """
    floor = phase_config.floor
    t_by_kimg: dict[int, float] = {}
    for phase in phase_config.phases:
        members = [
            i
            for i, kimg in enumerate(kimgs)
            if phase.kimg_start <= kimg <= phase.kimg_end
        ]
        if len(members) < 2:
            raise ValueError(
                f"Phase kimg {phase.kimg_start}-{phase.kimg_end} has "
                f"{len(members)} snapshot(s); at least 2 are required"
            )
        weights = [
            max(abs(smoothed[members[k + 1]] - smoothed[members[k]]), floor)
            for k in range(len(members) - 1)
        ]
        cumulative = [0.0, *accumulate(weights)]
        total = cumulative[-1]
        span = phase.t_end - phase.t_start
        last = len(members) - 1
        for k, index in enumerate(members):
            if k == 0:
                t_coord = phase.t_start
            elif k == last:
                t_coord = phase.t_end
            elif total > 0.0:
                t_coord = phase.t_start + span * (cumulative[k] / total)
            else:
                t_coord = phase.t_start + span * (k / last)
            t_by_kimg[kimgs[index]] = t_coord
    return t_by_kimg


class Interpolator:
    """Maps an audience position t in [0, 1] to a bracketing snapshot pair.

    Construction assigns every indexed snapshot a strictly monotonic t
    coordinate (see _build_t_coords). A call binary-searches the bracket
    containing t and returns the two bounding kimgs plus a linear blend factor.
    """

    def __init__(
        self, snapshots: Sequence[SnapshotInfo], phase_config: PhaseConfig
    ) -> None:
        if len(snapshots) < 2:
            raise ValueError("Interpolator requires at least two snapshots")
        ordered = sorted(snapshots, key=lambda s: s.kimg)
        kimgs = [s.kimg for s in ordered]
        smoothed = _rolling_mean(
            [s.fid_raw for s in ordered], phase_config.smoothing_window
        )
        t_by_kimg = _build_t_coords(kimgs, smoothed, phase_config)

        self._kimgs: list[int] = sorted(t_by_kimg)
        self._t_coords: list[float] = [t_by_kimg[kimg] for kimg in self._kimgs]
        logger.info(
            "Interpolator built: %d snapshots, t spans [%.3f, %.3f]",
            len(self._kimgs),
            self._t_coords[0],
            self._t_coords[-1],
        )

    def __call__(self, t: float) -> tuple[int, int, float]:
        """Return (snap_a_kimg, snap_b_kimg, blend) bracketing position ``t``."""
        t = min(1.0, max(0.0, t))
        max_bracket = len(self._kimgs) - 2
        bracket = bisect.bisect_right(self._t_coords, t) - 1
        bracket = min(max(bracket, 0), max_bracket)
        t_a = self._t_coords[bracket]
        t_b = self._t_coords[bracket + 1]
        span = t_b - t_a
        blend = (t - t_a) / span if span > 0.0 else 0.0
        blend = min(1.0, max(0.0, blend))
        return self._kimgs[bracket], self._kimgs[bracket + 1], blend
