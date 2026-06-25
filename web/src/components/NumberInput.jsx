import { useEffect, useRef, useState } from "react";
import * as Label from "@radix-ui/react-label";

function parseNumberInput(raw) {
  const n = parseFloat(raw);
  return Number.isFinite(n) ? n : null;
}

function format(value) {
  return String(Math.round(value * 1000) / 1000);
}

export function NumberInput({ label, value, step = 1, min, max, onChange }) {
  const [text, setText] = useState(() => format(value));
  const editing = useRef(false);

  useEffect(() => {
    if (!editing.current) setText(format(value));
  }, [value]);

  const onInput = (e) => {
    setText(e.target.value);
    const parsed = parseNumberInput(e.target.value);
    if (parsed !== null) onChange(parsed);
  };

  return (
    <div className="control">
      <Label.Root className="control__label">{label}</Label.Root>
      <input
        className="number-input"
        type="number"
        step={step}
        min={min}
        max={max}
        value={text}
        onFocus={() => {
          editing.current = true;
        }}
        onBlur={() => {
          editing.current = false;
          setText(format(value));
        }}
        onChange={onInput}
      />
    </div>
  );
}
