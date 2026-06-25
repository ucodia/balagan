import { useCallback, useEffect, useRef, useState } from "react";
import { fetchConfig } from "../transport/config.js";
import { openConnection } from "../transport/connection.js";
import { createDecoder } from "../transport/decoder.js";
import { reconcile } from "./controlSync.js";

// Field names mirror both the StateSnapshot keys the server pushes and the OSC
// addresses ("/" + field), so one map drives the whole UI ↔ server round trip.
const INITIAL_CONTROLS = {
  position: 0,
  truncation: 0.7,
  seedSpeedX: 0.25,
  seedSpeedY: 0,
  seedAnim: false,
  seedX: 0,
  seedY: 0,
  fpsCap: 30,
  debug: false,
  oscPort: 7700,
};

function toWire(field, value) {
  return typeof value === "boolean" ? (value ? 1 : 0) : value;
}

// Orchestrates the live session: fetch config, open the WebTransport
// connection, feed frames to a WebCodecs decoder painting `canvasRef`, and keep
// a reconciled view of the controls. Returns the status line, fps, the current
// control values, and a `send(field, value)` setter that updates locally and
// forwards to the server.
export function useStream(canvasRef) {
  const [status, setStatus] = useState("connecting…");
  const [fps, setFps] = useState(0);
  const [engineStatus, setEngineStatus] = useState("");
  const [controls, setControls] = useState(INITIAL_CONTROLS);

  const controlsRef = useRef(INITIAL_CONTROLS);
  const touchedAtRef = useRef({});
  const connRef = useRef(null);
  const paintsRef = useRef([]);
  const streamingRef = useRef(false);

  controlsRef.current = controls;

  const send = useCallback((field, value) => {
    touchedAtRef.current[field] = performance.now();
    setControls((prev) => ({ ...prev, [field]: value }));
    if (connRef.current) connRef.current.send("/" + field, toWire(field, value));
  }, []);

  useEffect(() => {
    let disposed = false;
    let decoder = null;

    const onPaint = () => {
      paintsRef.current.push(performance.now());
      streamingRef.current = true;
    };
    const onFrame = (parsed) => decoder && decoder.decode(parsed);
    const onState = (incoming) => {
      // `type` is the message tag and `status` is the read-only engine line; keep
      // both out of the reconciled controls so they never become editable fields.
      const { type, status: engine, ...rest } = incoming;
      void type;
      if (engine !== undefined) setEngineStatus(engine);
      const { merged, applied } = reconcile(
        controlsRef.current,
        rest,
        touchedAtRef.current,
        performance.now(),
      );
      if (applied.length) setControls(merged);
    };
    const onClose = () => {
      if (!disposed) setStatus("disconnected");
    };

    (async () => {
      if (!("WebTransport" in window)) {
        setStatus("WebTransport unavailable — use a recent Chromium browser");
        return;
      }
      try {
        const config = await fetchConfig();
        if (disposed) return;
        decoder = createDecoder(canvasRef.current, onPaint);
        connRef.current = await openConnection(config, { onFrame, onState, onClose });
        if (!disposed) setStatus("connected — waiting for frames…");
      } catch (err) {
        if (!disposed) setStatus(`connection failed: ${err.message}`);
      }
    })();

    const fpsTimer = setInterval(() => {
      const now = performance.now();
      paintsRef.current = paintsRef.current.filter((t) => now - t < 1000);
      const count = paintsRef.current.length;
      setFps(count);
      if (streamingRef.current) setStatus(`streaming — ${count} fps`);
    }, 250);

    return () => {
      disposed = true;
      clearInterval(fpsTimer);
      if (connRef.current) connRef.current.close();
      if (decoder) decoder.close();
    };
  }, [canvasRef]);

  return { status, fps, engineStatus, controls, send };
}
