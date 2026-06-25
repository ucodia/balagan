// Reconcile server state pushes (every 100ms) against locally-edited controls.
// While the user is editing a field — and for a short grace window after their
// last edit — incoming server values for that field are held back so the widget
// does not snap away under the user's hand. Untouched fields track the server.
// Pure and side-effect free so the policy can be unit-tested in isolation.

export const TOUCH_GRACE_MS = 600;

export function reconcile(local, incoming, touchedAt, now, grace = TOUCH_GRACE_MS) {
  const merged = { ...local };
  const applied = [];
  for (const key of Object.keys(incoming)) {
    const last = touchedAt[key];
    const held = last !== undefined && now - last < grace;
    if (!held) {
      merged[key] = incoming[key];
      applied.push(key);
    }
  }
  return { merged, applied };
}
