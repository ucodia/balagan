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
The default Syphon/Spout output is unchanged and remains the default
(`--output auto`).

The web path encodes with **libx264** (`tune=zerolatency`) by default on every
platform: the hardware H.264 encoders (VideoToolbox, NVENC) emit bitstreams that
browser decoders buffer and stall on at each keyframe, adding latency and a
periodic freeze. x264's low-latency signaling decodes immediately, and the
`superfast` preset keeps encode well under a frame at ~1024px. To experiment with
a hardware encoder anyway, pass e.g. `--web-codec h264_videotoolbox`.

WebTransport requires TLS. For local development, generate a short-lived
self-signed certificate that the browser trusts via `serverCertificateHashes`:

```bash
uv run python web/generate_cert.py   # writes web/certs/
```

Then start the engine with web output:

```bash
uv run balagan --headless --snapshots-dir <run> --output web
```

This also hosts the browser client itself over plain HTTP, so just open it in
Chrome/Edge — no separate static server needed:

```
http://127.0.0.1:8000
```

The client fetches the certificate hash and WebTransport port from the engine at
`/config.json`, so nothing machine-specific is hardcoded. Use `127.0.0.1` rather
than `localhost`, which browsers often resolve to IPv6 first while the server
binds IPv4. Change the host port with `--web-ui-port`.

To reach the client from another machine on the LAN, bind it to all interfaces:

```bash
uv run balagan --headless --snapshots-dir <run> --output web --web-host 0.0.0.0
```

The startup log prints the LAN URL to open (e.g. `https://192.168.1.20:8000`).
Non-loopback hosts are served over **HTTPS** — WebTransport requires a secure
context, which a bare `http://<LAN-IP>` is not. Because the cert is self-signed
and issued for `localhost`, the browser shows a one-time warning on the LAN
machine; click through ("Proceed") and the stream connects.

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
