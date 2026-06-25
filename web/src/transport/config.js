// The Python UI server synthesizes /config.json with the QUIC port, connection
// path, and (for self-signed dev certs) the certificate's SHA-256 hash.
export async function fetchConfig() {
  const res = await fetch("/config.json", { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`config fetch failed: ${res.status}`);
  }
  return res.json();
}
