"""Standalone synthesis benchmark (temporary; not part of the package).

Builds the real engine from a snapshots dir, renders a fixed number of frames
with no Spout/OSC, and reports end-to-end fps plus the engine's per-phase
breakdown. Used to measure the synthesis bottleneck and validate CUDA-graph
capture. Run:

    uv run python bench_synthesis.py --snapshots-dir <dir> --canonical-kimg <k>
"""

import argparse
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

_STYLEGAN3_DIR = Path(__file__).resolve().parent / "stylegan3"


class UtilSampler(threading.Thread):
    """Polls nvidia-smi GPU utilization on a background thread (during the
    timed loop only, so model-load/JIT idle never pollutes the average)."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self._interval = interval
        self._stop_event = threading.Event()
        self.samples: list[int] = []

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                self.samples.append(int(out.stdout.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001
                pass
            self._stop_event.wait(self._interval)

    def stop(self):
        self._stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots-dir", required=True)
    parser.add_argument("--canonical-kimg", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=0, help="0 = load all")
    parser.add_argument("--position", type=float, default=0.5, help="alpha!=0 by default")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if str(_STYLEGAN3_DIR) not in sys.path:
        sys.path.insert(0, str(_STYLEGAN3_DIR))

    import torch

    from balagan.config import load_run
    from balagan.core.engine import build_engine
    from balagan.core.runtime_state import RuntimeState

    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"
    print(f"device={device} torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    config = load_run(args.snapshots_dir, args.canonical_kimg)
    state = RuntimeState()
    state.update(fps_cap=0, position=args.position)
    engine = build_engine(
        config,
        device,
        window_size=args.window_size,
        runtime_state=state,
        use_cuda_graph=not args.no_cuda_graph,
    )
    engine.prime()
    engine.start()

    # Warmup: triggers custom-op JIT compile, cudnn algo selection, allocator fill.
    for _ in range(args.warmup):
        engine.render_frame()
    torch.cuda.synchronize()

    sampler = UtilSampler()
    sampler.start()
    start = time.perf_counter()
    for _ in range(args.frames):
        engine.render_frame()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    sampler.stop()
    sampler.join(timeout=1.0)

    engine.stop()
    fps = args.frames / elapsed
    util = sampler.samples
    util_avg = sum(util) / len(util) if util else 0
    mode = "eager" if args.no_cuda_graph else "cuda-graph"
    print(f"\n=== [{mode}] {args.frames} frames in {elapsed:.3f}s -> {fps:.1f} fps "
          f"({elapsed / args.frames * 1e3:.2f} ms/frame) ===")
    print(f"=== GPU util during render: avg {util_avg:.0f}%  "
          f"(min {min(util) if util else 0} max {max(util) if util else 0}, "
          f"n={len(util)}) ===")


if __name__ == "__main__":
    main()
