#!/usr/bin/env python
"""Generate a self-signed dev certificate for the WebTransport server.

Usage:
    uv run python web/generate_cert.py [--cert PATH] [--key PATH]

Writes an ECDSA cert/key pair (default: web/certs/) and prints the SHA-256 hash
to copy into the browser client's `serverCertificateHashes`. The certs directory
is gitignored; never commit certificates.
"""

import argparse
from pathlib import Path

from balagan.io.dev_cert import generate_self_signed_cert

_DEFAULT_DIR = Path(__file__).resolve().parent / "certs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert", type=Path, default=_DEFAULT_DIR / "cert.pem")
    parser.add_argument("--key", type=Path, default=_DEFAULT_DIR / "key.pem")
    args = parser.parse_args()

    digest = generate_self_signed_cert(args.cert, args.key)
    print(f"Certificate: {args.cert}")
    print(f"Private key: {args.key}")
    print(f"SHA-256:     {digest}")
    print("\nPaste the SHA-256 into web/main.js (CERT_HASH) before connecting.")


if __name__ == "__main__":
    main()
