import { describe, it, expect } from "vitest";
import {
  parsePayload,
  hexToBytes,
  concatChunks,
  KEYFRAME_FLAG,
  STATE_FLAG,
  HEADER_BYTES,
} from "../src/transport/protocol.js";

function makePayload(flags, seq, ts, body) {
  const buf = new Uint8Array(HEADER_BYTES + body.length);
  const view = new DataView(buf.buffer);
  view.setUint8(0, flags);
  view.setUint32(1, seq);
  view.setBigUint64(5, BigInt(ts));
  buf.set(body, HEADER_BYTES);
  return buf;
}

describe("parsePayload", () => {
  it("parses a keyframe video payload", () => {
    const p = parsePayload(makePayload(KEYFRAME_FLAG, 42, 1234, new Uint8Array([1, 2, 3, 4])));
    expect(p.isKeyframe).toBe(true);
    expect(p.isState).toBe(false);
    expect(p.sequence).toBe(42);
    expect(p.timestampMs).toBe(1234);
    expect(Array.from(p.body)).toEqual([1, 2, 3, 4]);
  });

  it("detects a state payload via the state flag", () => {
    const json = new TextEncoder().encode('{"type":"state"}');
    const p = parsePayload(makePayload(STATE_FLAG, 0, 0, json));
    expect(p.isState).toBe(true);
    expect(p.isKeyframe).toBe(false);
    expect(new TextDecoder().decode(p.body)).toBe('{"type":"state"}');
  });

  it("reads multi-byte fields big-endian (matching struct '>BIQ')", () => {
    const p = parsePayload(makePayload(0, 0x01020304, 0, new Uint8Array()));
    expect(p.sequence).toBe(0x01020304);
  });
});

describe("hexToBytes", () => {
  it("converts a hex string to bytes", () => {
    expect(Array.from(hexToBytes("00ff10"))).toEqual([0, 255, 16]);
  });
});

describe("concatChunks", () => {
  it("joins stream chunks in order", () => {
    const out = concatChunks([new Uint8Array([1, 2]), new Uint8Array([3])], 3);
    expect(Array.from(out)).toEqual([1, 2, 3]);
  });
});
