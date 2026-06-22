#!/usr/bin/env python
"""Benchmark raw render throughput and web-streaming quality across codecs.

Renders frames from a run with the live engine, then pushes the *same* frames
through each codec's encode → decode round trip (decode is deterministic, so the
result equals what a browser client would display) and reports encode cost, data
rate, and PSNR vs the raw frame.

Usage:
    uv run python scripts/benchmark_stream.py --snapshots-dir <run>
    uv run python scripts/benchmark_stream.py --snapshots-dir <run> \
        --frames 120 --bitrate 25000000 --codecs libx264,h264_videotoolbox \
        --samples-dir ./bench-samples

A single-snapshot run (e.g. one FFHQ .pkl) is a good fixed reference.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent


def _setup_stylegan3_path() -> None:
    sg3 = _REPO / "stylegan3"
    if str(sg3) not in sys.path:
        sys.path.insert(0, str(sg3))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_codecs() -> list[str]:
    if sys.platform == "darwin":
        return ["libx264", "h264_videotoolbox", "hevc_videotoolbox"]
    if sys.platform == "win32":
        return ["libx264", "h264_nvenc", "hevc_nvenc"]
    return ["libx264"]


def _decoder_name(codec: str) -> str:
    return "hevc" if ("hevc" in codec or "265" in codec) else "h264"


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 20 * math.log10(255) - 10 * math.log10(mse)


def render_frames(
    snapshots_dir: Path, device: str, count: int, window_size: int
) -> tuple[list[np.ndarray], list[float]]:
    """Render ``count`` frames with a moving latent and return them with per-frame
    render times (the FPS cap is disabled to measure true render throughput)."""
    from balagan.config import load_run
    from balagan.core.engine import build_engine
    from balagan.core.runtime_state import RuntimeState

    state = RuntimeState()
    state.update(fps_cap=0)  # unlimited: measure real render speed
    run = load_run(snapshots_dir)
    engine = build_engine(run, device, window_size=window_size, runtime_state=state)
    engine.prime()
    engine.start()

    frames: list[np.ndarray] = []
    times: list[float] = []
    warmup = 5
    for i in range(count + warmup):
        state.update(
            position=(i % 100) / 100.0,
            latent_x=i * 0.05,
            latent_y=i * 0.03,
            truncation_psi=0.7,
        )
        start = time.perf_counter()
        frame = engine.render_frame()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            frames.append(frame.copy())
            times.append(elapsed)
    engine.stop()
    return frames, times


def bench_codec(
    codec: str, frames: list[np.ndarray], fps: int, bitrate: int
) -> dict:
    """Encode then decode ``frames`` with ``codec``; return timing/size/PSNR."""
    import av

    from balagan.io.video_encoder import VideoEncoder, config_for

    height, width = frames[0].shape[:2]
    encoder = VideoEncoder(width, height, config_for(codec, fps=fps, bitrate=bitrate))

    chunks = []
    encode_seconds = 0.0
    for frame in frames:
        start = time.perf_counter()
        chunks.extend(encoder.encode(frame))
        encode_seconds += time.perf_counter() - start
    start = time.perf_counter()
    chunks.extend(encoder.close())
    encode_seconds += time.perf_counter() - start

    decoder = av.CodecContext.create(_decoder_name(codec), "r")
    decoded: list[np.ndarray] = []
    for chunk in chunks:
        for fr in decoder.decode(av.Packet(chunk.data)):
            decoded.append(fr.to_ndarray(format="rgb24"))
    for fr in decoder.decode(None):
        decoded.append(fr.to_ndarray(format="rgb24"))

    matched = min(len(frames), len(decoded))
    psnrs = [_psnr(frames[i], decoded[i]) for i in range(matched)]
    total_bytes = sum(len(c.data) for c in chunks)
    return {
        "encode_ms": 1000.0 * encode_seconds / len(frames),
        "kb_per_frame": total_bytes / len(frames) / 1024,
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "decoded": decoded[:matched],
    }


def _chroma_floor(frame: np.ndarray) -> float:
    import av

    vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
    roundtrip = vf.reformat(format="yuv420p").reformat(format="rgb24")
    return _psnr(frame, roundtrip.to_ndarray(format="rgb24"))


def _save_samples(samples_dir: Path, codec: str, frames, decoded) -> None:
    from PIL import Image

    out = samples_dir / codec.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    for idx in {0, len(decoded) // 2, len(decoded) - 1}:
        Image.fromarray(frames[idx]).save(out / f"raw_{idx:03d}.png")
        Image.fromarray(decoded[idx]).save(out / f"decoded_{idx:03d}.png")
        diff = np.abs(frames[idx].astype(np.int16) - decoded[idx].astype(np.int16))
        amp = np.clip(diff * 8, 0, 255).astype(np.uint8)
        Image.fromarray(amp).save(out / f"diff_x8_{idx:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--bitrate", type=int, default=25_000_000)
    parser.add_argument(
        "--fps", type=int, default=30, help="encoder rate-control fps (not render fps)"
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument(
        "--codecs", default=None, help="comma-separated; default: libx264 + platform HW"
    )
    parser.add_argument("--samples-dir", type=Path, default=None)
    args = parser.parse_args()

    _setup_stylegan3_path()
    device = _resolve_device(args.device)
    codecs = args.codecs.split(",") if args.codecs else _default_codecs()

    print(f"Rendering {args.frames} frames on device={device} …")
    frames, times = render_frames(
        args.snapshots_dir, device, args.frames, args.window_size
    )
    h, w = frames[0].shape[:2]
    mean_dt = sum(times) / len(times)
    print(f"\nResolution: {w}x{h}")
    print(
        f"Raw render: {1.0 / mean_dt:5.1f} fps "
        f"(mean {1000 * mean_dt:.1f} ms/frame, "
        f"best {1000 * min(times):.1f}, worst {1000 * max(times):.1f})"
    )
    print(f"Chroma 4:2:0 floor (no compression): {_chroma_floor(frames[len(frames) // 2]):.1f} dB")
    print(f"\nEncoder round trip @ {args.bitrate / 1e6:.0f} Mbps target:\n")
    print(f"  {'codec':<20} {'encode ms/frame':>16} {'KB/frame':>10} {'PSNR dB':>9}")
    print(f"  {'-' * 20} {'-' * 16} {'-' * 10} {'-' * 9}")
    for codec in codecs:
        try:
            r = bench_codec(codec, frames, args.fps, args.bitrate)
        except Exception as exc:  # noqa: BLE001 — report unavailable codecs, continue
            print(f"  {codec:<20} {'unavailable: ' + str(exc)[:40]}")
            continue
        print(
            f"  {codec:<20} {r['encode_ms']:>16.1f} "
            f"{r['kb_per_frame']:>10.0f} {r['psnr']:>9.1f}"
        )
        if args.samples_dir is not None:
            _save_samples(args.samples_dir, codec, frames, r["decoded"])
    if args.samples_dir is not None:
        print(f"\nSaved sample raw/decoded/diff images under {args.samples_dir}")


if __name__ == "__main__":
    main()
