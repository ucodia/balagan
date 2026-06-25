import * as Switch from "@radix-ui/react-switch";
import * as Label from "@radix-ui/react-label";

export function Toggle({ id, label, checked, onChange }) {
  return (
    <div className="control control--inline">
      <Label.Root className="control__label" htmlFor={id}>
        {label}
      </Label.Root>
      <Switch.Root
        id={id}
        className="switch"
        checked={checked}
        onCheckedChange={onChange}
      >
        <Switch.Thumb className="switch__thumb" />
      </Switch.Root>
    </div>
  );
}
