import { describe, it, expect } from "vitest";
import { reconcile, TOUCH_GRACE_MS } from "../src/hooks/controlSync.js";

describe("reconcile", () => {
  it("applies every incoming field when nothing was touched", () => {
    const { merged, applied } = reconcile(
      { position: 0, truncation: 0.7 },
      { position: 0.5, truncation: 0.2 },
      {},
      1000,
    );
    expect(merged).toEqual({ position: 0.5, truncation: 0.2 });
    expect(applied.sort()).toEqual(["position", "truncation"]);
  });

  it("holds a field edited inside the grace window", () => {
    const { merged, applied } = reconcile(
      { position: 0.9 },
      { position: 0.1 },
      { position: 1000 },
      1000 + TOUCH_GRACE_MS - 1,
    );
    expect(merged.position).toBe(0.9);
    expect(applied).toEqual([]);
  });

  it("accepts a field once the grace window has elapsed", () => {
    const { merged, applied } = reconcile(
      { position: 0.9 },
      { position: 0.1 },
      { position: 1000 },
      1000 + TOUCH_GRACE_MS + 1,
    );
    expect(merged.position).toBe(0.1);
    expect(applied).toEqual(["position"]);
  });

  it("holds only the touched field and applies the rest", () => {
    const { merged } = reconcile(
      { position: 0.9, truncation: 0.7 },
      { position: 0.1, truncation: 0.2 },
      { position: 1000 },
      1000,
    );
    expect(merged).toEqual({ position: 0.9, truncation: 0.2 });
  });
});
