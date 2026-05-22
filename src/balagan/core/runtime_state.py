"""Thread-safe shared runtime state for the GUI, OSC, and render threads."""

import threading
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class StateSnapshot:
    """An immutable snapshot of the runtime state, consumed once per frame."""

    position: float = 0.0
    latent_x: float = 0.0
    latent_y: float = 0.0
    anim_playing: bool = False
    anim_speed: float = 0.25
    truncation_psi: float = 0.7
    fps_cap: int = 30
    spout_syphon_enabled: bool = False
    debug: bool = False


class RuntimeState:
    """Thread-safe holder for the values mutated by GUI controls and OSC input.

    The current values live in an immutable StateSnapshot; every update swaps in
    a freshly built one under a lock, and snapshot() hands the render thread the
    current immutable instance directly -- no torn reads, no defensive copy.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = StateSnapshot()

    def update(self, **changes: object) -> None:
        """Atomically apply one or more field changes.

        Raises TypeError if a change names a field that does not exist.
        """
        with self._lock:
            self._current = replace(self._current, **changes)

    def snapshot(self) -> StateSnapshot:
        """Return the current immutable state for the render thread to consume."""
        with self._lock:
            return self._current
