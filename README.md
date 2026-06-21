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

## Web streaming output (experimental)

`--output web` streams rendered frames to a browser over WebTransport
(HTTP/3 / QUIC), decoded with WebCodecs — no native client, no viewer-side GPU.
Frames are hardware-encoded (VideoToolbox on macOS, NVENC on Windows, libx264
fallback elsewhere). The default Syphon/Spout output is unchanged and remains the
default (`--output auto`).

WebTransport requires TLS. For local development, generate a short-lived
self-signed certificate that the browser trusts via `serverCertificateHashes`:

```bash
uv run python web/generate_cert.py   # writes web/certs/, prints a SHA-256 hash
```

Paste the printed SHA-256 into `web/main.js` (`CERT_HASH`), then start the engine
with web output:

```bash
uv run balagan --headless --snapshots-dir <run> --output web
```

Serve the `web/` directory over plain HTTP (no build step) and open it in
Chrome/Edge:

```bash
python -m http.server -d web 8000   # then open http://localhost:8000
```

The certificate is valid for under 14 days (Chrome's limit for
`serverCertificateHashes`); re-run the generator when it expires. Certificates
live under `web/certs/` and are gitignored — never commit them.

## Development

Run the test suite:

```bash
uv run pytest
```

## License

Apache License 2.0. See LICENSE.

The StyleGAN3 submodule is governed by its own license terms (NVIDIA Source Code License), which apply when you use code from that submodule. This is separate from the license of this repository.
