import { SliderControl } from "./SliderControl.jsx";
import { Toggle } from "./Toggle.jsx";

// Exposes exactly the control surface the server understands (see
// io/control_mapping.py). Drag-driven seedX/seedY live on the viewport canvas.
export function ControlPanel({ controls, send }) {
  return (
    <div className="panel">
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
      <Toggle
        id="seedAnim"
        label="animate seed"
        checked={controls.seedAnim}
        onChange={(v) => send("seedAnim", v)}
      />
    </div>
  );
}
