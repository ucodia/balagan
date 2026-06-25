// Wire protocol shared with the Python server (src/balagan/io/web_stream.py).
// Every payload the server sends over a unidirectional WebTransport stream is
// prefixed with a 13-byte big-endian header (struct ">BIQ"):
//   uint8  flags        bit 0 = keyframe, bit 1 = state message
//   uint32 sequence
//   uint64 timestamp_ms
// The body is either an H.264 access unit (frame) or UTF-8 JSON (state).

export const KEYFRAME_FLAG = 0x01;
export const STATE_FLAG = 0x02;
export const HEADER_BYTES = 13;

export function parsePayload(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const flags = view.getUint8(0);
  return {
    flags,
    sequence: view.getUint32(1),
    timestampMs: Number(view.getBigUint64(5)),
    isKeyframe: (flags & KEYFRAME_FLAG) !== 0,
    isState: (flags & STATE_FLAG) !== 0,
    body: bytes.subarray(HEADER_BYTES),
  };
}

// Convert the hex SHA-256 string from /config.json into the byte array
// WebTransport's serverCertificateHashes option expects.
export function hexToBytes(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return out;
}

// Concatenate the chunks a single WebTransport stream delivers into one payload.
export function concatChunks(chunks, totalLength) {
  const out = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}
