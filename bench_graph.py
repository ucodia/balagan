"""CUDA-graph capturability + speedup probe (temporary).

Isolates the synthesis forward: times eager vs CUDA-graph replay on the real
blended network, and checks output equivalence. Answers whether StyleGAN3's
custom ops are capture-safe and whether graphing removes the launch-bound idle.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

_STYLEGAN3_DIR = Path(__file__).resolve().parent / "stylegan3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots-dir", required=True)
    parser.add_argument("--canonical-kimg", type=int, default=None)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    if str(_STYLEGAN3_DIR) not in sys.path:
        sys.path.insert(0, str(_STYLEGAN3_DIR))

    import torch

    from balagan.config import load_run
    from balagan.core.engine import build_engine
    from balagan.core.runtime_state import RuntimeState

    device = "cuda"
    config = load_run(args.snapshots_dir, args.canonical_kimg)
    state = RuntimeState()
    state.update(fps_cap=0, position=0.5)
    engine = build_engine(config, device, window_size=0, runtime_state=state)
    engine.prime()
    engine.render_frame()  # populate blender cache

    kimg_a, kimg_b, alpha = engine._interpolator(0.5)
    print(f"pair {kimg_a}->{kimg_b} alpha={alpha:.3f}")

    def timed(fn, n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    with torch.no_grad():
        synth = engine._weight_blender(kimg_a, kimg_b, alpha)
        ws = engine._latent_navigator(0.0, 0.0, 1.0).unsqueeze(0).contiguous()

        # --- eager ---
        for _ in range(20):
            out_eager = synth(ws=ws, noise_mode="const")
        eager_ms = timed(lambda: synth(ws=ws, noise_mode="const"), args.iters)
        out_eager = synth(ws=ws, noise_mode="const").clone()

        # --- capture ---
        static_ws = ws.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):
                synth(ws=static_ws, noise_mode="const")
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                static_out = synth(ws=static_ws, noise_mode="const")
        except Exception as exc:  # noqa: BLE001
            print(f"\nCAPTURE FAILED: {type(exc).__name__}: {exc}")
            return

        graph_ms = timed(graph.replay, args.iters)

        # --- correctness (same weights) ---
        static_ws.copy_(ws)
        graph.replay()
        torch.cuda.synchronize()
        max_diff = (static_out - out_eager).abs().max().item()

        # --- crux: re-blend new weights in place, replay must reflect them ---
        engine._weight_blender(kimg_a, kimg_b, 0.2)  # lerp alpha=0.2 into blend_net params
        graph.replay()
        torch.cuda.synchronize()
        out_eager_a2 = synth(ws=ws, noise_mode="const")  # synth IS blend_net (now alpha=0.2)
        diff_vs_new = (static_out - out_eager_a2).abs().max().item()
        diff_vs_old = (static_out - out_eager).abs().max().item()

    print(f"\neager   : {eager_ms:6.2f} ms  ({1000/eager_ms:.1f} fps synth-only)")
    print(f"graph   : {graph_ms:6.2f} ms  ({1000/graph_ms:.1f} fps synth-only)")
    print(f"speedup : {eager_ms / graph_ms:.2f}x")
    print(f"max|diff| same-weights : {max_diff:.6f}  (0 = identical)")
    print(f"after in-place reblend -> replay:")
    print(f"  diff vs new eager : {diff_vs_new:.6f}  (want ~0: replay saw new weights)")
    print(f"  diff vs old output: {diff_vs_old:.6f}  (want >0: weights actually changed)")


if __name__ == "__main__":
    main()
