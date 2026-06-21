// BalaGAN WebTransport + WebCodecs client.
//
// Connects to the BalaGAN WebTransport server, reads one encoded H.264 access
// unit per incoming unidirectional stream, and decodes it to a <canvas> with
// WebCodecs. Each stream carries a 13-byte header (flags | seq | timestamp_ms)
// followed by an Annex B frame, so the decoder is configured WITHOUT a
// `description`.
//
// Before connecting, set CERT_HASH to the SHA-256 printed by
// `uv run python web/generate_cert.py`, and SERVER_URL to your host:port.

const CERT_HASH = "PASTE_SHA256_FROM_generate_cert.py";
const SERVER_URL = "https://localhost:4433/balagan";
const CODEC = "avc1.640028"; // H.264 High@4.0; adjust if your stream differs

const HEADER_BYTES = 13;
const KEYFRAME_FLAG = 0x01;

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");

function setStatus(text) {
  statusEl.textContent = text;
}

function hexToBytes(hex) {
  const clean = hex.trim();
  const bytes = new Uint8Array(clean.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(clean.substr(i * 2, 2), 16);
  }
  return bytes;
}

function makeDecoder() {
  return new VideoDecoder({
    output: (frame) => {
      if (canvas.width !== frame.displayWidth) canvas.width = frame.displayWidth;
      if (canvas.height !== frame.displayHeight)
        canvas.height = frame.displayHeight;
      ctx.drawImage(frame, 0, 0);
      frame.close();
    },
    error: (e) => setStatus(`decoder error: ${e.message}`),
  });
}

async function readFullStream(readable) {
  const reader = readable.getReader();
  const parts = [];
  let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parts.push(value);
    total += value.byteLength;
  }
  const buffer = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    buffer.set(part, offset);
    offset += part.byteLength;
  }
  return buffer;
}

async function run() {
  if (!("WebTransport" in window)) {
    setStatus("WebTransport unavailable — use Chrome/Edge.");
    return;
  }

  const transport = new WebTransport(SERVER_URL, {
    serverCertificateHashes: [{ algorithm: "sha-256", value: hexToBytes(CERT_HASH) }],
  });

  try {
    await transport.ready;
  } catch (e) {
    setStatus(`connection failed: ${e.message}`);
    return;
  }
  setStatus("connected — waiting for frames…");

  const decoder = makeDecoder();
  let configured = false;
  let frames = 0;

  const reader = transport.incomingUnidirectionalStreams.getReader();
  for (;;) {
    const { value: stream, done } = await reader.read();
    if (done) break;

    const buffer = await readFullStream(stream);
    if (buffer.byteLength <= HEADER_BYTES) continue;

    const view = new DataView(buffer.buffer, 0, HEADER_BYTES);
    const flags = view.getUint8(0);
    const isKeyframe = (flags & KEYFRAME_FLAG) !== 0;
    const timestampMs = Number(view.getBigUint64(5));
    const data = buffer.subarray(HEADER_BYTES);

    if (!configured) {
      if (!isKeyframe) continue; // can't start mid-GOP
      decoder.configure({ codec: CODEC, optimizeForLatency: true });
      configured = true;
    }

    decoder.decode(
      new EncodedVideoChunk({
        type: isKeyframe ? "key" : "delta",
        timestamp: timestampMs * 1000, // microseconds
        data,
      })
    );

    frames++;
    if (frames % 30 === 0) setStatus(`streaming — ${frames} frames`);
  }
}

run();
