# AGENTS.md

Guidance for AI coding agents working in this repository.

## Repo orientation

This is a real-time interpolation engine that blends StyleGAN2-ADA snapshots from a single training run, driven by an audience-position scalar that maps to a perceptual trajectory through training. Read `IMPLEMENTATION.md` for the full specification before doing any non-trivial work.

The `stylegan3/` directory is a git submodule of `https://github.com/ucodia/stylegan3` (a personal fork of NVIDIA's StyleGAN3 repo, used here for its StyleGAN2-ADA training code, network classes, and snapshot loading utilities). Treat it as a vendored dependency: do not modify files inside it. Import from it via the path setup in `cli.py`.

The `prototypes/` directory contains exploratory code (notably `gen_balagan.py`) that informs design decisions but is not part of the production engine. Reference it for patterns, do not depend on or modify it.

## Workflow

This project uses uv. Common commands:

- `uv sync` to install dependencies
- `uv run pytest` to run tests
- `uv run pytest tests/test_<module>.py -v` for focused testing
- `uv run balagan --help` to inspect CLI
- `uv run ruff check src/ tests/` for linting (if ruff is configured)
- `uv run ruff format src/ tests/` for formatting

After `git submodule update --init --recursive`, dependencies install cleanly with `uv sync`. If a teammate adds a dependency, re-run `uv sync` before working.

## Coding conventions

**Module structure.** Each `src/balagan/<area>/` package owns a coherent concept (core, io, gui). Cross-package imports flow downward: `gui` may import from `core` and `io`; `core` may import from neither. Violations of this rule indicate a design problem; raise it in your status report rather than working around it.

**Threading.** The engine runs on a render thread. The OSC server runs on its own thread. The GUI runs on Qt's main thread. The snapshot manager has its own background loader thread. Shared state goes through `RuntimeState` with explicit locking. Never call Qt APIs from a non-main thread; use signals.

**Tensor handling.** Snapshots are loaded onto the inference device once and stay there. Frame outputs go to CPU for display/transport. Never move tensors between devices on the render hot path (per-frame). When you need a CPU copy for the GUI viewport, do it once at the end of the frame.

**Logging.** Use `logging.getLogger(__name__)` at module scope. Never use `print()` outside of CLI top-level user-facing output. Log levels per the spec: INFO for state changes, WARNING for recoverable issues, ERROR for exceptions.

**Errors.** Validation errors (bad config, missing snapshots, malformed OSC) should produce clear actionable messages. Engine-internal errors should bubble up with stack traces in logs. Never silently swallow exceptions.

**Type hints.** Use modern Python type hints (`list[int]`, not `List[int]`) and require Python 3.12. Add hints to public function signatures; inline locals don't need them unless type inference is unclear.

## Testing requirements

Every module in `src/balagan/core/` must have a corresponding test file. The CLI, GUI, and IO modules are tested at integration level only.

Tests should mock snapshot loading where possible. Real snapshot files are too large to ship; tests that need a real network use a fixture that constructs a tiny stub `nn.Module` with z_dim=4, w_dim=4, num_ws=2, img_resolution=64.

Run `uv run pytest` before declaring any task complete. If a test fails, fix it or explain why it cannot be fixed in this scope — do not skip or comment out failing tests.

## Things that will trip you up

**The interpolator's t-coordinate construction is non-obvious.** The math is described in IMPLEMENTATION.md (interpolator section). The key invariants: strictly monotonic, perceptually weighted within phases, floored to avoid stalls in flat-FID regions. If your implementation passes the tests in `tests/test_interpolator.py`, you've got it right.

**The rolling window with a fixed canonical slot.** The canonical snapshot always occupies one slot of `window_size`. The remaining slots distribute around the current pair with symmetric padding, clamped at snapshot list edges. The example in IMPLEMENTATION.md (20 snapshots, canonical at index 6, current pair at (18, 19), window_size=8) produces `{6, 13, 14, 15, 16, 17, 18, 19}` and is the canonical test case.

**The seed-grid bilinear and the autolume drag pattern are not invented here.** They come from NVIDIA's StyleGAN3 visualizer and Metacreation Lab's Autolume respectively. URLs are in the bootstrap prompt. Re-read those sources rather than reconstructing from memory.

**Weight-space blending requires the prototype's specific approach.** Cache state_dicts at load time (not per-frame), pre-allocate one blend target generator, lerp in-place into its tensors. Do not call `state_dict()` per frame; that allocates and is slow. Do not construct new generators per frame.

**Z→W mapping uses ONLY the canonical model's mapping network.** Not the active pair's. Not "whichever is closest." Always the canonical. This is what gives the work its BalaGAN-faithful chimera behavior: latent geometry is fixed, only the synthesis varies.

## Things not to do

Do not add features beyond what IMPLEMENTATION.md specifies. If you think something is missing, raise it in your status report and let the human decide.

Do not refactor files you weren't asked to touch.

Do not add new dependencies without checking the spec — it lists the allowed set.

Do not introduce frame-space blending as a fallback. Weight-space only.

Do not implement zero-copy CUDA-OpenGL interop. CPU roundtrip in v1.

Do not implement hot-reload of config. Restart required.

Do not modify the `stylegan3/` submodule or the `prototypes/` directory.

Do not generate large amounts of synthetic test data, model files, or sample images. Tests use stub modules, not real models.

## Working with the human

Work through `IMPLEMENTATION.md`'s "Implementation order" one step at a time. Stop after each step and report status:

- What was completed
- What tests pass
- What you noticed but didn't change
- What the next step is

Wait for explicit "continue" before proceeding. The human is using this checkpoint pattern to catch design issues early.

If you encounter a genuine ambiguity in the spec — not a "could be done two ways" but a "the spec contradicts itself" or "the spec doesn't say" — ask, don't guess. Pick the most conservative interpretation if forced to proceed.

If you find a bug or design flaw in existing code while working on something else, note it in your status report. Do not fix it in the same change unless explicitly asked.
