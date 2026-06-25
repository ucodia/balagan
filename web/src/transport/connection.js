import { parsePayload, hexToBytes, concatChunks } from "./protocol.js";

// Open the WebTransport session described by /config.json. Incoming
// unidirectional streams carry framed payloads (video or state, see
// protocol.js); a single bidirectional stream carries newline-delimited JSON
// control messages back to the server.
export async function openConnection(config, { onFrame, onState, onClose }) {
  const url = `https://${location.hostname}:${config.webtransportPort}${config.path}`;
  const options = {};
  if (config.certHash) {
    options.serverCertificateHashes = [
      { algorithm: "sha-256", value: hexToBytes(config.certHash) },
    ];
  }

  const transport = new WebTransport(url, options);
  await transport.ready;

  const controlStream = await transport.createBidirectionalStream();
  const controlWriter = controlStream.writable.getWriter();
  const encoder = new TextEncoder();

  const send = (addr, value) => {
    controlWriter
      .write(encoder.encode(JSON.stringify({ addr, value }) + "\n"))
      .catch(() => {});
  };

  readIncoming(transport, onFrame, onState);
  transport.closed.then(() => onClose && onClose()).catch(() => onClose && onClose());

  return {
    send,
    close() {
      try {
        transport.close();
      } catch {
        /* already closing */
      }
    },
  };
}

async function readIncoming(transport, onFrame, onState) {
  const decoder = new TextDecoder();
  const reader = transport.incomingUnidirectionalStreams.getReader();
  while (true) {
    let result;
    try {
      result = await reader.read();
    } catch {
      break; // transport closed
    }
    if (result.done) break;
    // Await each stream in turn to preserve the server's frame order.
    await readStream(result.value, decoder, onFrame, onState);
  }
}

async function readStream(stream, decoder, onFrame, onState) {
  const r = stream.getReader();
  const chunks = [];
  let total = 0;
  while (true) {
    let chunk;
    try {
      chunk = await r.read();
    } catch {
      return;
    }
    if (chunk.done) break;
    chunks.push(chunk.value);
    total += chunk.value.length;
  }
  const parsed = parsePayload(concatChunks(chunks, total));
  if (parsed.isState) {
    try {
      onState(JSON.parse(decoder.decode(parsed.body)));
    } catch {
      /* drop malformed state */
    }
  } else {
    onFrame(parsed);
  }
}
