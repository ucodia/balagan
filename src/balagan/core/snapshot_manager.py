"""Rolling window of synthesis networks with a background loader thread."""

import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from balagan.config import SnapshotInfo

logger = logging.getLogger(__name__)


def _compute_window(
    num_snapshots: int,
    canonical_index: int | None,
    index_a: int,
    index_b: int,
    window_size: int,
) -> set[int]:
    """Snapshot indices to keep resident.

    The window always holds the canonical slot (when present) and the active
    pair, then pads symmetrically around the pair -- biased toward later
    snapshots and clamped at the list edges -- until it reaches ``window_size``
    distinct indices (or the list is exhausted). A canonical index that
    coincides with a pair or padding slot is not double-counted.
    """
    window: set[int] = {index_a, index_b}
    if canonical_index is not None:
        window.add(canonical_index)
    left = min(index_a, index_b) - 1
    right = max(index_a, index_b) + 1
    take_right = True
    while len(window) < window_size and (left >= 0 or right < num_snapshots):
        if take_right and right < num_snapshots:
            window.add(right)
            right += 1
        elif left >= 0:
            window.add(left)
            left -= 1
        elif right < num_snapshots:
            window.add(right)
            right += 1
        take_right = not take_right
    return window


class SnapshotManager:
    """Keeps a rolling window of synthesis networks loaded on the inference device.

    The window always includes the canonical snapshot (if its kimg is indexed)
    and the active pair, padded around the pair. Loads run on a background
    thread so the render thread is never blocked; ``prime`` does a synchronous
    initial load to avoid a black-frame startup.
    """

    def __init__(
        self,
        snapshots: Sequence[SnapshotInfo],
        canonical_kimg: int,
        loader: Callable[[Path], torch.nn.Module],
        window_size: int = 8,
    ) -> None:
        self._snapshots = sorted(snapshots, key=lambda snap: snap.kimg)
        self._index_of = {snap.kimg: i for i, snap in enumerate(self._snapshots)}
        self._canonical_kimg = canonical_kimg
        self._loader = loader
        # A non-positive window size means "no limit": keep every snapshot resident.
        self._window_size = window_size if window_size > 0 else len(self._snapshots)

        self._loaded: dict[int, torch.nn.Module] = {}
        self._desired: set[int] = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    def prime(self, kimg_a: int, kimg_b: int) -> None:
        """Synchronously load the window for a pair; blocks until the window is
        resident. Used at startup to avoid a black first frame."""
        with self._lock:
            self._desired = self._window_kimgs(kimg_a, kimg_b)
        while self._reconcile_step():
            pass
        logger.info("Snapshot manager primed window: %s", sorted(self._loaded))

    def start(self) -> None:
        """Start the background loader thread."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loader_loop, name="snapshot-loader", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background loader thread and wait for it to exit."""
        self._running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def set_active_pair(self, kimg_a: int, kimg_b: int) -> None:
        """Recompute the desired window and wake the loader thread. Non-blocking."""
        desired = self._window_kimgs(kimg_a, kimg_b)
        with self._lock:
            if desired == self._desired:
                return
            self._desired = desired
        self._wake.set()

    def get_synthesis(self, kimg: int) -> torch.nn.Module | None:
        """Return the loaded synthesis network for a kimg, or None if not loaded."""
        with self._lock:
            return self._loaded.get(kimg)

    def loaded_networks(self) -> dict[int, torch.nn.Module]:
        """An atomic snapshot of the resident networks, keyed by kimg.

        The returned dict is a private copy taken under the lock: it stays
        stable even if the loader thread evicts snapshots immediately
        afterward, and its references keep those networks alive for as long as
        the caller holds it. A render frame must take this single view rather
        than combining ``loaded_kimgs`` and ``get_synthesis``, which can
        disagree once the loader thread evicts between the two calls.
        """
        with self._lock:
            return dict(self._loaded)

    def is_pair_ready(self, kimg_a: int, kimg_b: int) -> bool:
        """Whether both snapshots of a pair are currently loaded."""
        with self._lock:
            return kimg_a in self._loaded and kimg_b in self._loaded

    def loaded_kimgs(self) -> set[int]:
        """The kimgs of all currently-loaded snapshots."""
        with self._lock:
            return set(self._loaded)

    def pending_count(self) -> int:
        """How many desired snapshots are not yet loaded."""
        with self._lock:
            return len(self._desired - self._loaded.keys())

    def _window_kimgs(self, kimg_a: int, kimg_b: int) -> set[int]:
        indices = _compute_window(
            len(self._snapshots),
            self._index_of.get(self._canonical_kimg),
            self._index_of[kimg_a],
            self._index_of[kimg_b],
            self._window_size,
        )
        return {self._snapshots[index].kimg for index in indices}

    def _loader_loop(self) -> None:
        while self._running:
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            while self._running and self._reconcile_step():
                pass

    def _reconcile_step(self) -> bool:
        """Evict snapshots outside the window and load one missing snapshot.

        The load itself runs outside the lock so the render thread is never
        blocked. Returns True while more snapshots still need loading.
        """
        with self._lock:
            evicted = self._loaded.keys() - self._desired
            for kimg in evicted:
                del self._loaded[kimg]
            pending = sorted(self._desired - self._loaded.keys())
        for kimg in evicted:
            logger.debug("Snapshot manager evicted kimg %d", kimg)
        if not pending:
            return False

        target = pending[0]
        network = self._loader(self._snapshots[self._index_of[target]].pkl_path)
        with self._lock:
            stored = target in self._desired
            if stored:
                self._loaded[target] = network
            more = bool(self._desired - self._loaded.keys())
        if stored:
            logger.info("Snapshot manager loaded kimg %d", target)
        return more
