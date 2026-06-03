# BalaGAN

BalaGAN is a real-time weight interpolation engine which allows to navigate latent space across StyleGAN model snapshots.

## Requirements

- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Python 3.12 — the only tested version (uv installs it automatically)

## Installation

Clone the repository together with its `stylegan3` submodule:

```bash
git clone --recursive git@github.com:ucodia/balagan.git
cd balagan
```

If you already cloned without `--recursive`, fetch the submodule:

```bash
git submodule update --init --recursive
```

Install dependencies into a managed virtual environment:

```bash
uv sync
```

## Usage

```bash
uv run balagan --snapshots-dir <training-run-folder>
```

Pass `--headless` to run without the GUI. List all options:

```bash
uv run balagan --help
```

## Development

Run the test suite:

```bash
uv run pytest
```

## License

Apache License 2.0. See LICENSE.

The StyleGAN3 submodule is governed by its own license terms (NVIDIA Source Code License), which apply when you use code from that submodule. This is separate from the license of this repository.
