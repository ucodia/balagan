import { SliderControl } from "./SliderControl.jsx";
import { NumberInput } from "./NumberInput.jsx";
import { Toggle } from "./Toggle.jsx";

// Exposes exactly the control surface the server understands over the web
// channel (see io/control_mapping.py, plus the /oscPort handler in web_stream.py):
// position, truncation, seed position/speed/animation, FPS cap, debug overlay, and
// the OSC listening port. Filesystem-bound controls (snapshot folder, Spout/Syphon,
// recording) stay out — the browser client can't drive them. Drag on the viewport
// canvas also writes seedX/seedY, reading and writing the same controls state.
export function ControlPanel({ controls, send, status, engineStatus }) {
  const engineLines = engineStatus ? engineStatus.split(" | ") : [];
  return (
    <div className="panel">
      <section className="group">
        <h2 className="group__title">Navigation</h2>
        <SliderControl
          label="position"
          min={0}
          max={1}
          step={0.001}
          value={controls.position}
          onChange={(v) => send("position", v)}
        />
        <SliderControl
          label="truncation"
          min={0}
          max={1}
          step={0.001}
          value={controls.truncation}
          onChange={(v) => send("truncation", v)}
        />
        <div className="control-grid">
          <NumberInput
            label="seed X"
            step={1}
            value={controls.seedX}
            onChange={(v) => send("seedX", v)}
          />
          <NumberInput
            label="seed Y"
            step={1}
            value={controls.seedY}
            onChange={(v) => send("seedY", v)}
          />
        </div>
        <SliderControl
          label="seed speed X"
          min={-2}
          max={2}
          step={0.01}
          value={controls.seedSpeedX}
          onChange={(v) => send("seedSpeedX", v)}
        />
        <SliderControl
          label="seed speed Y"
          min={-2}
          max={2}
          step={0.01}
          value={controls.seedSpeedY}
          onChange={(v) => send("seedSpeedY", v)}
        />
        <button
          type="button"
          className="button button--play"
          onClick={() => send("seedAnim", !controls.seedAnim)}
        >
          {controls.seedAnim ? "Pause" : "Play"}
        </button>
      </section>

      <section className="group">
        <h2 className="group__title">Output</h2>
        <NumberInput
          label="FPS limit (0 = unlimited)"
          step={1}
          min={0}
          max={120}
          value={controls.fpsCap}
          onChange={(v) => send("fpsCap", v)}
        />
        <Toggle
          id="debug"
          label="debug overlay"
          checked={controls.debug}
          onChange={(v) => send("debug", v)}
        />
        <NumberInput
          label="OSC port"
          step={1}
          min={1024}
          max={65535}
          value={controls.oscPort}
          onChange={(v) => send("oscPort", v)}
        />
      </section>

      <section className="group">
        <h2 className="group__title">Status</h2>
        <div className="status">
          <span>{status}</span>
          {engineLines.map((line, i) => (
            <span key={i}>{line}</span>
          ))}
          <span>OSC listening on port {controls.oscPort}</span>
        </div>
      </section>
    </div>
  );
}
