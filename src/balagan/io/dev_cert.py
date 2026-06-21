"""Self-signed certificate generation for local WebTransport development.

WebTransport over HTTP/3 requires TLS. Rather than involving a CA, Chrome lets a
client trust a specific certificate via ``serverCertificateHashes`` as long as
the certificate uses an ECDSA key and is valid for at most 14 days. This module
emits exactly such a certificate and returns the SHA-256 hash of its DER
encoding, which the browser client must be configured with.
"""

import datetime
import hashlib
from pathlib import Path


def generate_self_signed_cert(cert_path: Path, key_path: Path) -> str:
    """Write an ECDSA cert/key pair for ``localhost`` and return its SHA-256 hash.

    The certificate is valid for 13 days (under Chrome's 14-day
    ``serverCertificateHashes`` limit). The returned value is the hex SHA-256 of
    the DER-encoded certificate.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert_path = Path(cert_path)
    key_path = Path(key_path)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=13))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False
        )
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    der = certificate.public_bytes(serialization.Encoding.DER)
    return hashlib.sha256(der).hexdigest()


def cert_sha256(cert_path: Path) -> str:
    """Return the hex SHA-256 of an existing certificate's DER encoding.

    This is the value the browser needs in ``serverCertificateHashes``; the web
    UI server serves it to the client so the hash is never hardcoded.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    certificate = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    der = certificate.public_bytes(serialization.Encoding.DER)
    return hashlib.sha256(der).hexdigest()
