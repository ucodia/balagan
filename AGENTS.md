# AGENTS.md

Guidance for AI coding agents working in this repository.

## What BalaGAN is

A real-time weight-interpolation engine that blends StyleGAN snapshots
from a single training run, driven by an audience-position scalar in `[0, 1]`
that maps through a phase-aware, FID-weighted trajectory. It was built for an
interactive installation where audience proximity to a wall drives the model's
"nightmare level" through training — that context informs design choices like
the normalized 0→1 position, the headless mode for venue use, and the
conservative default FPS cap.

## Repo map

```
balagan/
├── src/balagan/
│   ├── cli.py                  # `balagan` entry point (--headless, --debug)
│   ├── config.py               # phase-config JSON + run-folder validation
│   ├── logging_config.py       # console + daily-rotating file
│   ├── core/                   # the pipeline; imports neither gui nor io
│   │   ├── canonical_mapping.py
│   │   ├── engine.py           # per-frame orchestration
│   │   ├── interpolator.py     # phase-aware FID-weighted t-coord mapping
│   │   ├── latent_navigator.py # seed-grid bilinear z→w
│   │   ├── runtime_state.py    # thread-safe shared state
│   │   ├── snapshot_manager.py # rolling window + background loader
│   │   └── weight_blender.py   # in-place state_dict lerp
│   ├── io/
│   │   ├── osc_server.py       # python-osc on its own thread
│   │   ├── frame_output.py     # platform dispatch
│   │   ├── output_macos.py     # Syphon
│   │   └── output_windows.py   # Spout (marked UNVERIFIED — Windows-only deps)
│   └── gui/
│       ├── main_window.py      # PySide6 QMainWindow
│       ├── viewport.py         # rendered frame + mouse drag
│       ├── control_panel.py    # widgets, two-way bound to RuntimeState
│       └── render_worker.py    # QThread driving the engine loop
├── tests/                      # pytest; core fully unit-tested, others integration only
├── stylegan3/                  # git submodule — DO NOT MODIFY
├── prototypes/                 # exploratory; reference for patterns, DO NOT MODIFY
├── pyproject.toml
└── README.md
```

## Setup and common commands

```bash
git submodule update --init --recursive   # required before first uv sync
uv sync                                    # install deps + build the project
uv run pytest                              # full suite (~98 tests)
uv run pytest tests/test_<module>.py -v    # focused
uv run balagan --help                      # CLI surface
uv run ruff check src/ tests/              # lint (if ruff is configured)
uv run ruff format src/ tests/             # format
```

## Platforms

- **macOS** (primary dev target). MPS torch (default index), Syphon output
  via `syphon-python`.
- **Windows.** CUDA torch from the `pytorch-cu128` uv index declared in
  `pyproject.toml`, Spout output via `SpoutGL`, plus Windows-only `pyopengl`
  and `ninja`. The `[tool.uv.sources]` block routes the `torch` dep to that
  index for `sys_platform == 'win32' or 'linux'`.
- **Linux.** Same CUDA index as Windows; not regularly verified.
- **Timer resolution.** Always use `time.perf_counter()` for per-frame
  timing. `time.monotonic()` is ~15 ms granularity on Windows and quantizes
  the frame limiter.
- **Default `--window-size` is 32** (was 8 in early development; raised to
  use typical VRAM / unified-memory budgets).

## Coding conventions

**Module layering.** `gui` may import from `core` and `io`; `core` may
import from neither. Violations indicate a design problem — raise it in
your status report rather than working around it.

**Threading.** Render thread, OSC server thread, snapshot loader thread,
Qt main thread. Shared state goes through `RuntimeState` (lock-guarded
immutable `StateSnapshot` swap). Never call Qt APIs from a non-main thread
— use signals.

**Tensor handling.** Snapshots load to the inference device once and stay
there. Never move tensors between devices on the render hot path. The CPU
copy for the GUI viewport happens once at frame end.

**Logging.** `logging.getLogger(__name__)` at module scope. Never `print()`
outside CLI top-level user-facing output. INFO for state changes, WARNING
for recoverable issues, ERROR for exceptions.

**Errors.** Validation errors (bad config, missing snapshots, malformed
OSC) produce clear actionable messages. Engine-internal errors bubble up
with tracebacks. Never silently swallow exceptions.

**Type hints.** Modern (`list[int]`, not `List[int]`); the project is pinned
to Python 3.12. Hints on public function signatures; inline locals don't
need them unless inference is unclear.

**Comments.** Only when the *why* is non-obvious — see the global CLAUDE.md
rules.

## Testing

- Every module in `src/balagan/core/` has a corresponding
  `tests/test_<module>.py`.
- CLI, GUI and IO are tested at integration level only — no committed Qt
  unit tests; offscreen smoke checks are throwaway.
- Tests use a stub `nn.Module` (`z_dim=4`, `w_dim=4`, `num_ws=2`,
  `img_resolution=64`) for snapshot loading; never real `.pkl` files.
- Run `uv run pytest` before declaring any task complete. If a test fails,
  fix it or explain why it can't be fixed in this scope — don't skip or
  comment out failing tests.

## Things that will trip you up

**The interpolator's t-coordinate construction.** Strictly monotonic,
perceptually weighted within phases (rolling-mean smoothed |ΔFID|, floored
to avoid stalls in flat-FID regions), phase boundaries pinned exactly. If
your implementation passes `tests/test_interpolator.py`, you've got it
right.

**The rolling window with a fixed canonical slot.** The canonical snapshot
always occupies one slot of `window_size`; the remaining slots distribute
around the current pair with symmetric padding, clamped at list edges. The
canonical is never double-counted. See
`src/balagan/core/snapshot_manager.py::_compute_window` and the
`test_window_*` cases in `tests/test_snapshot_manager.py`.

**The seed-grid bilinear is not invented here.** It mirrors NVIDIA's
StyleGAN3 visualizer (`stylegan3/viz/latent_widget.py` ~lines 67-77 and
`stylegan3/viz/renderer.py` ~lines 264-282). The Autolume mouse-drag scale
comes from Metacreation Lab's autolume
(`autolume/widgets/latent_widget.py` ~lines 82-85: `delta / font_size *
4e-2`). Re-read those sources before changing
`src/balagan/core/latent_navigator.py` — don't reconstruct from memory.

**Weight-space blending requires the prototype's specific approach.**
Cache each snapshot's `state_dict()` at load time, pre-allocate one
blend-target generator, lerp in-place into its tensors per call. Never
call `state_dict()` on the per-frame path; never construct new generators
per frame. See `src/balagan/core/weight_blender.py`.

**Z→W mapping uses ONLY the canonical model's mapping network.** Not the
active pair's. Not "whichever is closest." Always the canonical. This is
what gives BalaGAN its chimera behavior: latent geometry is fixed, only
synthesis varies.

**`render_frame` derives the whole frame from one atomic snapshot view.**
`SnapshotManager.loaded_networks()` returns a lock-guarded copy of the
resident networks; `render_frame` uses that single view for the blender
cache, the pair choice, AND the blend. Re-reading the manager mid-frame
races the background loader and has previously produced `KeyError` crashes
that killed the render thread.

## Things not to do

- Do not introduce frame-space blending as a fallback. Weight-space only.
- Do not implement zero-copy CUDA/OpenGL interop. CPU roundtrip is fine.
- Do not implement hot-reload of the config. Restart required.
- Do not modify `stylegan3/` (vendored submodule) or `prototypes/`
  (exploratory code).
- Do not generate large amounts of synthetic test data, model files, or
  sample images. Tests use stub modules.
- Do not commit phase-config JSONs. They are training-run-specific and
  live with the training run, not in version control.
- Do not refactor files you weren't asked to touch.
- Do not add features without explicit user authorization. If something
  seems missing, flag it in your status report and let the human decide.

## Working with the human

For non-trivial multi-step tasks, stop after each step and report:

- What was completed
- What tests pass
- What you noticed but didn't change
- What the next step is

Wait for an explicit "continue" before proceeding.

If you find a bug or design flaw in existing code while working on
something else, note it in your status report. Do not fix it in the same
change unless explicitly asked.

If you hit a genuine ambiguity — not "could be done two ways" but "the
request contradicts itself" or "doesn't say" — ask, don't guess. If forced
to proceed, pick the most conservative interpretation.

Commits only when explicitly asked. Match the project's commit style (short
imperative title; optional explanatory body). Never add links to Claude or
claude.ai sessions in commit messages or PRs.
