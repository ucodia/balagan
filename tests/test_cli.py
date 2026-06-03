"""Tests for balagan.cli."""

from click.testing import CliRunner

from balagan.cli import _resolve_device, main


def test_resolve_device_passes_through_explicit_devices():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda:0") == "cuda:0"


def test_resolve_device_auto_picks_an_available_backend():
    assert _resolve_device("auto") in {"cuda", "mps", "cpu"}


def test_cli_help_lists_options():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--snapshots-dir" in result.output
    assert "--headless" in result.output
    assert "--debug" in result.output
    assert "--canonical-kimg" in result.output


def test_cli_headless_requires_a_snapshots_dir():
    result = CliRunner().invoke(main, ["--headless"])
    assert result.exit_code != 0
    assert "--snapshots-dir is required in headless mode" in result.output
