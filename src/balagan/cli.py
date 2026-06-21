"""Command-line entry point for the BalaGAN engine."""

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

_STYLEGAN3_DIR = Path(__file__).resolve().parent.parent.parent / "stylegan3"


def _resolve_device(device: str) -> str:
    """Resolve '--device auto' to cuda, mps, or cpu; pass any explicit device through."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_stylegan3_on_path() -> None:
    """Put the vendored stylegan3 submodule on sys.path for legacy/dnnlib imports."""
    path = str(_STYLEGAN3_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _run_headless(engine, osc_server, output_name) -> None:
    """Start the OSC server and run the render loop, publishing each frame."""
    from balagan.io.frame_output import FrameOutput

    osc_server.start()
    engine.prime()
    engine.start()
    first_frame = engine.render_frame()
    height, width = first_frame.shape[:2]
    output = FrameOutput(output_name, width, height)
    output.send(first_frame)
    logger.info("Headless rendering started; press Ctrl+C to stop")
    try:
        while True:
            output.send(engine.render_frame())
    except KeyboardInterrupt:
        logger.info("Interrupt received; shutting down")
    finally:
        output.close()
        engine.stop()
        osc_server.stop()


def _run_gui(
    initial_config, runtime_state, osc_server, device, window_size, output_name
) -> None:
    """Start the OSC server and run the PySide6 GUI until the window closes.

    The engine is built lazily by the render worker once a folder is selected
    (or from ``initial_config`` when one was passed on the command line), so the
    Qt main thread never blocks on the load.
    """
    import signal

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from balagan.gui.main_window import MainWindow

    osc_server.start()
    app = QApplication([])
    window = MainWindow(
        runtime_state,
        device,
        window_size,
        output_name,
        osc_server,
        initial_config=initial_config,
    )
    window.show()

    # Qt's C++ event loop blocks Python's SIGINT delivery, so Ctrl+C is ignored
    # while it runs. Close the window on SIGINT (triggering the worker teardown),
    # and run a periodic no-op timer to give the interpreter a slice in which to
    # actually deliver the signal.
    signal.signal(signal.SIGINT, lambda *_: window.close())
    keepalive = QTimer()
    keepalive.start(200)
    keepalive.timeout.connect(lambda: None)

    logger.info("GUI window opened")
    try:
        app.exec()
    finally:
        osc_server.stop()


@click.command()
@click.option(
    "--snapshots-dir",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the training run folder. Optional in GUI mode (pick a folder "
    "in the window); required with --headless.",
)
@click.option(
    "--canonical-index",
    type=int,
    default=None,
    help="Override the canonical mapping snapshot's 0-based index (default: middle of the sorted snapshots).",
)
@click.option(
    "--headless",
    is_flag=True,
    help="Run without the GUI window; engine and OSC server only.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Overlay engine status data on the output.",
)
@click.option(
    "--osc-port", type=int, default=7700, show_default=True, help="OSC listening port."
)
@click.option(
    "--output-name",
    default="BalaGAN",
    show_default=True,
    help="Spout/Syphon output name.",
)
@click.option("--device", default="auto", show_default=True, help="Inference device.")
@click.option(
    "--window-size",
    type=int,
    default=32,
    show_default=True,
    help="Snapshot manager window size (0 = load all snapshots).",
)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default="logs",
    show_default=True,
    help="Log file directory.",
)
def main(snapshots_dir, canonical_index, headless, debug, osc_port, output_name, device, window_size, log_dir):
    """Real-time interpolation engine blending StyleGAN training snapshots."""
    from balagan.logging_config import setup_logging

    setup_logging(log_dir)

    resolved_device = _resolve_device(device)
    logger.info(
        "Starting BalaGAN | mode=%s device=%s osc-port=%d output=%s window-size=%d",
        "headless" if headless else "gui",
        resolved_device,
        osc_port,
        output_name,
        window_size,
    )

    _ensure_stylegan3_on_path()

    from balagan.config import ConfigError, load_run
    from balagan.core.runtime_state import RuntimeState
    from balagan.io.osc_server import OSCServer

    runtime_state = RuntimeState()
    runtime_state.update(debug=debug)
    osc_server = OSCServer(runtime_state, port=osc_port)

    if headless:
        if snapshots_dir is None:
            raise click.ClickException("--snapshots-dir is required in headless mode")
        try:
            config = load_run(snapshots_dir, canonical_index)
        except ConfigError as exc:
            raise click.ClickException(str(exc)) from exc
        from balagan.core.engine import build_engine

        engine = build_engine(
            config, resolved_device, window_size=window_size, runtime_state=runtime_state
        )
        _run_headless(engine, osc_server, output_name)
    else:
        initial_config = None
        if snapshots_dir is not None:
            try:
                initial_config = load_run(snapshots_dir, canonical_index)
            except ConfigError as exc:
                raise click.ClickException(str(exc)) from exc
        _run_gui(
            initial_config,
            runtime_state,
            osc_server,
            resolved_device,
            window_size,
            output_name,
        )


if __name__ == "__main__":
    main()
