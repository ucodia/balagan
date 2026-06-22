// BalaGAN WebTransport + WebCodecs client.
//
// Connects to the BalaGAN WebTransport server, reads one encoded H.264 access
// unit per incoming unidirectional stream, and decodes it to a <canvas> with
// WebCodecs. Each stream carries a 13-byte header (flags | seq | timestamp_ms)
// followed by an Annex B frame, so the decoder is configured WITHOUT a
// `description`.
//
// The cert hash and WebTransport port are fetched from the hosting engine at
// /config.json, so nothing machine-specific is hardcoded here. The connection
// uses the same hostname the page was served from (use 127.0.0.1 rather than
// localhost, which browsers often resolve to IPv6 while the server binds IPv4).

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

const frameTimes = [];
let lastStatusAt = 0;

function makeDecoder() {
  return new VideoDecoder({
    output: (frame) => {
      if (canvas.width !== frame.displayWidth) canvas.width = frame.displayWidth;
      if (canvas.height !== frame.displayHeight)
        canvas.height = frame.displayHeight;
      ctx.drawImage(frame, 0, 0);
      frame.close();
      // Actual frames painted in the last second = rendered FPS.
      const now = performance.now();
      frameTimes.push(now);
      while (frameTimes.length && frameTimes[0] <= now - 1000) frameTimes.shift();
      if (now - lastStatusAt > 250) {
        setStatus(`streaming — ${frameTimes.length} fps`);
        lastStatusAt = now;
      }
    },
    error: (e) => {
      console.error("decoder error", e);
      setStatus(`decoder error: ${e.message}`);
    },
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

// Mirrors the Qt viewport's Autolume drag: latent delta = pixel delta * scale,
// with scale = 4e-2 / font size (~13pt), and the seed is absolute on the engine.
const SEED_DRAG_SCALE = 4e-2 / 13;

async function setupControls(transport) {
  const stream = await transport.createBidirectionalStream();
  const writer = stream.writable.getWriter();
  const encoder = new TextEncoder();

  const send = (addr, value) =>
    writer.write(encoder.encode(JSON.stringify({ addr, value }) + "\n"));

  const bind = (id, addr) => {
    const el = document.getElementById(id);
    el.addEventListener("input", () => send(addr, parseFloat(el.value)));
  };
  bind("position", "/position");
  bind("truncation", "/truncation");
  bind("speedX", "/seedSpeedX");
  bind("speedY", "/seedSpeedY");

  const anim = document.getElementById("anim");
  anim.addEventListener("change", () => send("/seedAnim", anim.checked ? 1 : 0));

  // Drag the video canvas to move the seed, like the Qt viewport.
  let seedX = 0;
  let seedY = 0;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.style.cursor = "grab";
  canvas.addEventListener("pointerdown", (ev) => {
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    seedX += (ev.clientX - lastX) * SEED_DRAG_SCALE;
    seedY += (ev.clientY - lastY) * SEED_DRAG_SCALE;
    lastX = ev.clientX;
    lastY = ev.clientY;
    send("/seedX", seedX);
    send("/seedY", seedY);
  });
  const endDrag = () => (dragging = false);
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
}

async function run() {
  if (!("WebTransport" in window)) {
    setStatus("WebTransport unavailable — use Chrome/Edge.");
    return;
  }

  let config;
  try {
    config = await (await fetch("/config.json", { cache: "no-store" })).json();
  } catch (e) {
    setStatus(`could not load config: ${e.message}`);
    return;
  }
  if (!config.certHash) {
    setStatus("server certificate hash unavailable — generate a cert first.");
    return;
  }

  const url = `https://${location.hostname}:${config.webtransportPort}${config.path}`;
  const transport = new WebTransport(url, {
    serverCertificateHashes: [
      { algorithm: "sha-256", value: hexToBytes(config.certHash) },
    ],
  });

  try {
    await transport.ready;
  } catch (e) {
    setStatus(`connection failed: ${e.message}`);
    return;
  }
  setStatus("connected — waiting for frames…");
  setupControls(transport).catch((e) => setStatus(`control error: ${e.message}`));

  const decoder = makeDecoder();
  let configured = false;

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
  }
}

run();
