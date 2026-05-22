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


def _run_gui(engine, osc_server, run_dir, output_name) -> None:
    """Start the OSC server and run the PySide6 GUI until the window closes."""
    from PySide6.QtWidgets import QApplication

    from balagan.gui.main_window import MainWindow

    osc_server.start()
    engine.prime()  # eager initial load before the window appears
    app = QApplication([])
    window = MainWindow(engine, run_dir, output_name)
    window.show()
    logger.info("GUI window opened")
    try:
        app.exec()
    finally:
        osc_server.stop()


@click.command()
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the training run folder.",
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the phase config JSON.",
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
    default=8,
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
def main(run_dir, config_path, headless, debug, osc_port, output_name, device, window_size, log_dir):
    """Real-time interpolation engine blending StyleGAN2-ADA training snapshots."""
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

    from balagan.config import ConfigError, load_config

    try:
        config = load_config(config_path, run_dir)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    from balagan.core.engine import build_engine
    from balagan.io.osc_server import OSCServer

    engine = build_engine(config, resolved_device, window_size=window_size)
    engine.runtime_state.update(debug=debug)
    osc_server = OSCServer(engine.runtime_state, port=osc_port)
    if headless:
        _run_headless(engine, osc_server, output_name)
    else:
        _run_gui(engine, osc_server, run_dir, output_name)


if __name__ == "__main__":
    main()
