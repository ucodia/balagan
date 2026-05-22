"""Render a multi-checkpoint interpolation video (Balagan-style).

Generalizes gen_video.py by adding a second interpolation axis: a "time"
value in [0, 1] that selects which two adjacent StyleGAN2-ADA checkpoints to
blend, plus the blend weight between them. The seed (W-space) cycle and the
time (checkpoint) cycle each complete exactly once over `--duration` seconds
and loop seamlessly.

Two interpolation modes are available via --interpolation-mode:

  image  (default): synthesize from both adjacent checkpoints, alpha-blend
                    the output pixels. Two forward passes per frame.
                    Crossfade aesthetic: ghosting between two distinct outputs.

  weight: linearly interpolate the network weights between adjacent
          checkpoints, then synthesize once from the blended generator.
          One forward pass per frame, but per-frame weight blending.
          Smoother single-output aesthetic; output of an "intermediate
          checkpoint" that lives between the two real ones. Only safe
          because all checkpoints come from the same training run and
          therefore the same loss basin.

Run from inside the StyleGAN2-ADA repo (or with it on PYTHONPATH) so that
`dnnlib`, `legacy`, and `torch_utils` are importable. Checkpoints must come
from the same training run (same z_dim, w_dim, num_ws, img_resolution) and
must be passed in kimg order -- the script does not sort them.

Example:

    python gen_balagan.py \\
        --checkpoints "snaps/*.pkl" \\
        --seeds 12,42,78,38 \\
        --times 0.0,0.5,1.0,0.5 \\
        --duration 30 --fps 60 --trunc 0.7 \\
        --interpolation-mode image \\
        --output balagan_test_image.mp4

    # Same args, different mode, for A/B comparison:
    python gen_balagan.py ... --interpolation-mode weight --output balagan_test_weight.mp4

--checkpoints is repeatable. Each value is a literal path or a quoted glob
("snaps/*.pkl"); globs are expanded and sorted alphabetically inside the
script. Order across --checkpoints flags is preserved (kimg order is the
caller's responsibility).
"""

import copy
import glob
import math
import os
import re
from typing import Dict, List, Sequence, Tuple

import click
import dnnlib
import imageio
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import scipy.interpolate
import torch
from tqdm import tqdm

import legacy

# ---------------------------------------------------------------------------

def parse_int_list(s):
    """Comma-separated ints with inclusive ranges, e.g. "1,2,5-10" -> [1,2,5,6,7,8,9,10]."""
    if isinstance(s, list):
        return s
    range_re = re.compile(r'^(\d+)-(\d+)$')
    out: List[int] = []
    for p in s.split(','):
        p = p.strip()
        if not p:
            continue
        m = range_re.match(p)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            out.extend(range(lo, hi + 1) if lo <= hi else range(lo, hi - 1, -1))
        else:
            out.append(int(p))
    return out


def parse_float_list(s):
    if isinstance(s, list):
        return s
    return [float(x) for x in s.split(',') if x.strip()]


def expand_checkpoint_paths(values: Sequence[str]) -> List[str]:
    """Expand each value as `~`/glob, preserving input order across values.
    Glob matches within a single value are sorted alphabetically."""
    glob_chars = set('*?[')
    out: List[str] = []
    for v in values:
        v = os.path.expanduser(v)
        if any(c in v for c in glob_chars):
            matches = sorted(glob.glob(v))
            if not matches:
                raise click.ClickException(f'--checkpoints pattern "{v}" matched no files.')
            out.extend(matches)
        else:
            out.append(v)
    return out


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_checkpoints(paths: Sequence[str], device: torch.device):
    generators = []
    for path in paths:
        print(f'Loading network from "{path}"...')
        with dnnlib.util.open_url(path) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
        generators.append(G)

    g0 = generators[0]
    for i in range(1, len(generators)):
        g = generators[i]
        for attr in ('z_dim', 'w_dim', 'num_ws', 'img_resolution'):
            if getattr(g, attr) != getattr(g0, attr):
                raise click.ClickException(
                    f'Checkpoint mismatch: {paths[i]} has {attr}={getattr(g, attr)} '
                    f'but {paths[0]} has {attr}={getattr(g0, attr)}. '
                    'All checkpoints must come from the same training run.'
                )
    return generators


def build_seed_interp(ws_keyframes_np: np.ndarray, wraps: int = 2):
    n = ws_keyframes_np.shape[0]
    x = np.arange(-n * wraps, n * (wraps + 1))
    y = np.tile(ws_keyframes_np, (wraps * 2 + 1, 1, 1))
    return scipy.interpolate.interp1d(x, y, kind='cubic', axis=0)


def build_time_interp(times: Sequence[float], wraps: int = 2):
    m = len(times)
    times_arr = np.asarray(times, dtype=np.float32)
    x = np.arange(-m * wraps, m * (wraps + 1))
    y = np.tile(times_arr, wraps * 2 + 1)
    return scipy.interpolate.interp1d(x, y, kind='linear')


def time_to_blend(t: float, n: int) -> Tuple[int, int, float]:
    """Map t in [0, 1] to (lower_idx, upper_idx, alpha) over n checkpoints."""
    if n == 1:
        return 0, 0, 0.0
    pos = t * (n - 1)
    lower_idx = max(0, min(int(math.floor(pos)), n - 1))
    upper_idx = min(lower_idx + 1, n - 1)
    alpha = float(pos - lower_idx)
    return lower_idx, upper_idx, alpha


def synth_pair(generators, w_tensor: torch.Tensor, lower_idx: int, upper_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    img_lower = generators[lower_idx].synthesis(ws=w_tensor.unsqueeze(0), noise_mode='const')[0]
    if upper_idx == lower_idx:
        return img_lower, img_lower
    img_upper = generators[upper_idx].synthesis(ws=w_tensor.unsqueeze(0), noise_mode='const')[0]
    return img_lower, img_upper


def blend_pair(img_lower: torch.Tensor, img_upper: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha == 0.0 or img_lower is img_upper:
        return img_lower
    return (1.0 - alpha) * img_lower + alpha * img_upper


def to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img.permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------
# Weight-space interpolation helpers.
#
# Strategy:
#   1. Cache each generator's full state_dict once at startup, keeping the
#      tensors on-device. State dicts include both parameters and buffers
#      (e.g. running stats), so we use state_dict() rather than parameters().
#   2. Pre-allocate one "blend target" generator (deepcopy of G[0]) whose
#      tensors we will overwrite in-place each frame.
#   3. Per frame: lerp lower/upper state dicts into the target's tensors via
#      `out.copy_(lower * (1-a) + upper * a)`. No allocation, no instance
#      creation, no state_dict(...) round-trip.
# ---------------------------------------------------------------------------

def cache_state_dicts(generators) -> List[Dict[str, torch.Tensor]]:
    """Snapshot each generator's state_dict; tensors stay on their existing device."""
    return [{k: v.detach() for k, v in g.state_dict().items()} for g in generators]


def make_blend_generator(g0):
    """Pre-allocate a generator whose tensors we will overwrite in-place each frame."""
    return copy.deepcopy(g0)


def write_blended_state(
    blend_state: Dict[str, torch.Tensor],
    sd_lower: Dict[str, torch.Tensor],
    sd_upper: Dict[str, torch.Tensor],
    alpha: float,
) -> None:
    """In-place: blend_state[k] <- (1-alpha)*sd_lower[k] + alpha*sd_upper[k].

    Falls back to a straight copy for non-floating-point tensors (e.g. integer
    buffers like num_batches_tracked) where lerp is meaningless.
    """
    for k, dst in blend_state.items():
        a = sd_lower[k]
        b = sd_upper[k]
        if dst.is_floating_point():
            # dst <- a*(1-alpha) + b*alpha, in place, no temporaries.
            torch.lerp(a, b, alpha, out=dst)
        else:
            dst.copy_(a if alpha < 0.5 else b)


def synth_weight_blend(
    G_blend,
    blend_state: Dict[str, torch.Tensor],
    state_dicts: List[Dict[str, torch.Tensor]],
    generators,
    w_tensor: torch.Tensor,
    lower_idx: int,
    upper_idx: int,
    alpha: float,
):
    """Synthesize once from the weight-blended generator.

    Fast paths:
      - alpha == 0 or lower == upper: synthesize directly from generators[lower_idx].
      - alpha == 1: synthesize directly from generators[upper_idx].
    Otherwise: blend weights into G_blend, synthesize once.
    """
    if upper_idx == lower_idx or alpha == 0.0:
        G_active = generators[lower_idx]
    elif alpha == 1.0:
        G_active = generators[upper_idx]
    else:
        write_blended_state(blend_state, state_dicts[lower_idx], state_dicts[upper_idx], alpha)
        G_active = G_blend
    return G_active.synthesis(ws=w_tensor.unsqueeze(0), noise_mode='const')[0]


# ---------------------------------------------------------------------------

def load_overlay_font(px: int) -> PIL.ImageFont.ImageFont:
    candidates = [
        '/System/Library/Fonts/Menlo.ttc',
        '/System/Library/Fonts/Monaco.ttf',
        '/Library/Fonts/Andale Mono.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return PIL.ImageFont.truetype(path, size=px)
            except Exception:
                continue
    try:
        return PIL.ImageFont.load_default(size=px)
    except TypeError:
        return PIL.ImageFont.load_default()


def draw_overlay(frame: np.ndarray, lines: Sequence[str], font: PIL.ImageFont.ImageFont) -> np.ndarray:
    base = PIL.Image.fromarray(frame, 'RGB').convert('RGBA')
    overlay = PIL.Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(overlay)

    pad = 10
    spacing = 4
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_h = max(b[3] - b[1] for b in bboxes)
    text_w = max(b[2] - b[0] for b in bboxes)
    box_w = text_w + 2 * pad
    box_h = line_h * len(lines) + spacing * (len(lines) - 1) + 2 * pad

    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 170))
    y = pad
    for line in lines:
        draw.text((pad, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h + spacing

    return np.asarray(PIL.Image.alpha_composite(base, overlay).convert('RGB'))


# ---------------------------------------------------------------------------

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--checkpoints', 'checkpoint_paths', multiple=True, required=True, type=str,
              help='Path or QUOTED glob to .pkl checkpoint(s). Repeatable. '
                   'Quote globs (e.g. --checkpoints "snaps/*.pkl") so the script does the expansion; '
                   'unquoted shell globs will not work because click options accept only one value per flag.')
@click.option('--seeds', type=parse_int_list, required=True,
              help='Comma-separated seed keyframes with optional inclusive ranges (e.g. "12,42,78" or "0-29" or "1,5-10,42"). Loops back to the first.')
@click.option('--times', type=parse_float_list, required=True,
              help='Comma-separated time keyframes in [0,1] (e.g. "0.0,0.5,1.0,0.5"). Loops back to the first.')
@click.option('--duration', type=float, default=30.0, show_default=True,
              help='Total video length in seconds. Both seed and time cycles complete exactly once over this duration.')
@click.option('--fps', type=int, default=60, show_default=True, help='Output framerate.')
@click.option('--trunc', 'truncation_psi', type=float, default=1.0, show_default=True,
              help='Truncation psi, in (0, 1].')
@click.option('--interpolation-mode', 'interpolation_mode',
              type=click.Choice(['image', 'weight'], case_sensitive=False),
              default='image', show_default=True,
              help='How to interpolate between adjacent checkpoints. '
                   '"image" blends the two synthesized images (2 forward passes/frame). '
                   '"weight" lerps the network weights between adjacent checkpoints and synthesizes once '
                   '(1 forward pass/frame, plus per-frame weight lerp; only valid for checkpoints from '
                   'the same training run).')
@click.option('--output', type=click.Path(dir_okay=False), required=True,
              help='Output .mp4 path.')
@click.option('--bitrate', type=str, default='12M', show_default=True, help='ffmpeg bitrate.')
@click.option('--save-frames', is_flag=True,
              help='Also save PNG frames into a sibling folder named after the output (without extension). '
                   'In image mode: three PNGs per frame (frameNNNNNN.png blend, -a.png lower, -b.png upper). '
                   'In weight mode: one PNG per frame (frameNNNNNN.png), since only one synthesis happens. '
                   'PNGs are always clean (no overlay).')
@click.option('--debug-overlay', is_flag=True,
              help='Burn current frame index, seed/time phases, interpolated t, active checkpoint blend, '
                   'and interpolation mode into each video frame.')
def main(
    checkpoint_paths: Tuple[str, ...],
    seeds: List[int],
    times: List[float],
    duration: float,
    fps: int,
    truncation_psi: float,
    interpolation_mode: str,
    output: str,
    bitrate: str,
    save_frames: bool,
    debug_overlay: bool,
):
    if not (0.0 < truncation_psi <= 1.0):
        raise click.ClickException(f'--trunc must be in (0, 1], got {truncation_psi}.')
    if duration <= 0:
        raise click.ClickException(f'--duration must be > 0, got {duration}.')
    if fps <= 0:
        raise click.ClickException(f'--fps must be > 0, got {fps}.')
    for t in times:
        if not (0.0 <= t <= 1.0):
            raise click.ClickException(f'--times values must be in [0, 1], got {t}.')

    interpolation_mode = interpolation_mode.lower()

    device = select_device()
    print(f'Using device: {device}')

    paths = expand_checkpoint_paths(checkpoint_paths)
    print(f'Resolved {len(paths)} checkpoint(s) from {len(checkpoint_paths)} --checkpoints arg(s).')
    generators = load_checkpoints(paths, device)
    G0 = generators[0]
    print(f'{len(generators)} checkpoint(s); '
          f'z_dim={G0.z_dim} w_dim={G0.w_dim} num_ws={G0.num_ws} res={G0.img_resolution}')
    print(f'Interpolation mode: {interpolation_mode}')

    # Sample z keyframes from seeds. Mirror gen_video.py's MPS dtype quirk.
    if device.type == 'mps':
        z_np = np.stack([np.random.RandomState(s).randn(G0.z_dim).astype(np.float32) for s in seeds])
    else:
        z_np = np.stack([np.random.RandomState(s).randn(G0.z_dim) for s in seeds])
    zs = torch.from_numpy(z_np).to(device)

    # Map z -> w using the FIRST checkpoint. The mapping network drifts during
    # training, so we have to pick one; the first is arbitrary but consistent.
    ws_keyframes = G0.mapping(z=zs, c=None, truncation_psi=truncation_psi)
    _ = G0.synthesis(ws_keyframes[:1])  # warmup

    seed_interp = build_seed_interp(ws_keyframes.cpu().numpy())
    time_interp = build_time_interp(times)

    # Weight-mode setup: cache state dicts and pre-allocate the blend target.
    state_dicts: List[Dict[str, torch.Tensor]] = []
    G_blend = None
    blend_state: Dict[str, torch.Tensor] = {}
    if interpolation_mode == 'weight' and len(generators) > 1:
        print('Caching state dicts for weight-mode interpolation...')
        state_dicts = cache_state_dicts(generators)
        G_blend = make_blend_generator(G0)
        # Live reference to G_blend's tensors so we can write into them in place.
        blend_state = dict(G_blend.state_dict())

    n_seeds = len(seeds)
    n_times = len(times)
    n_ckpt = len(generators)
    total_frames = max(1, int(round(duration * fps)))
    seconds_per_time = duration / n_times
    seconds_per_seed = duration / n_seeds
    print(f'Seed cycle: {n_seeds} kf @ {seconds_per_seed:.3f}s/kf | '
          f'Time cycle: {n_times} kf @ {seconds_per_time:.3f}s/kf')
    print(f'Total frames: {total_frames} (~{total_frames / fps:.2f}s @ {fps} fps)')

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    writer = imageio.get_writer(output, mode='I', fps=fps, codec='libx264', bitrate=bitrate)
    frames_dir = None
    if save_frames:
        frames_dir = os.path.splitext(output)[0]
        os.makedirs(frames_dir, exist_ok=True)
    overlay_font = load_overlay_font(max(14, G0.img_resolution // 50)) if debug_overlay else None
    try:
        for f in tqdm(range(total_frames)):
            phase = f / total_frames  # in [0, 1)
            seed_pos = phase * n_seeds
            time_pos = phase * n_times
            w = seed_interp(seed_pos)
            t_raw = float(time_interp(time_pos))
            t = float(np.clip(t_raw, 0.0, 1.0))
            lower_idx, upper_idx, alpha = time_to_blend(t, n_ckpt)

            if device.type == 'mps':
                w = w.astype(np.float32)
            w_tensor = torch.from_numpy(w).to(device)

            if interpolation_mode == 'image':
                img_lower, img_upper = synth_pair(generators, w_tensor, lower_idx, upper_idx)
                img = blend_pair(img_lower, img_upper, alpha)
            else:  # weight
                img = synth_weight_blend(
                    G_blend, blend_state, state_dicts, generators,
                    w_tensor, lower_idx, upper_idx, alpha,
                )
                # For save-frames in weight mode, no separate -a/-b images: only one synthesis happened.
                img_lower = None
                img_upper = None

            frame_clean = to_uint8_hwc(img)

            if debug_overlay:
                seed_lo = int(math.floor(seed_pos)) % n_seeds
                seed_hi = (seed_lo + 1) % n_seeds
                seed_frac = seed_pos - math.floor(seed_pos)
                lines = [
                    f'frame {f:06d}/{total_frames}  ({f / fps:6.2f}s)  mode={interpolation_mode}',
                    f'seed phase {seed_pos:7.4f}  {seeds[seed_lo]}->{seeds[seed_hi]} @ {seed_frac:.3f}',
                    f'time phase {time_pos:7.4f}  t={t:.4f} (raw {t_raw:+.4f})',
                    f'ckpt {lower_idx:2d}->{upper_idx:2d} @ alpha={alpha:.4f}',
                ]
                frame_video = draw_overlay(frame_clean, lines, overlay_font)
            else:
                frame_video = frame_clean

            writer.append_data(frame_video)
            if frames_dir is not None:
                PIL.Image.fromarray(frame_clean, 'RGB').save(f'{frames_dir}/frame{f:06d}.png')
                if interpolation_mode == 'image':
                    PIL.Image.fromarray(to_uint8_hwc(img_lower), 'RGB').save(f'{frames_dir}/frame{f:06d}-a.png')
                    PIL.Image.fromarray(to_uint8_hwc(img_upper), 'RGB').save(f'{frames_dir}/frame{f:06d}-b.png')
    finally:
        writer.close()
    print(f'Done. Wrote {output}' + (f' and {frames_dir}/' if frames_dir else '') + '.')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
