import * as Slider from "@radix-ui/react-slider";
import * as Label from "@radix-ui/react-label";

export function SliderControl({ label, min, max, step, value, onChange }) {
  const decimals = step < 0.01 ? 3 : 2;
  return (
    <div className="control">
      <div className="control__row">
        <Label.Root className="control__label">{label}</Label.Root>
        <span className="control__value">{value.toFixed(decimals)}</span>
      </div>
      <Slider.Root
        className="slider"
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
      >
        <Slider.Track className="slider__track">
          <Slider.Range className="slider__range" />
        </Slider.Track>
        <Slider.Thumb className="slider__thumb" aria-label={label} />
      </Slider.Root>
    </div>
  );
}
