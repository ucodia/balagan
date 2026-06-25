import { useRef } from "react";

// Pixels-to-latent scale for canvas seed dragging, mirroring the StyleGAN3 /
// Autolume convention (delta / font_size * 4e-2, font_size ≈ 13).
const SEED_DRAG_SCALE = 4e-2 / 13;

export function Viewport({ canvasRef, status, controls, send }) {
  const drag = useRef(null);

  const onPointerDown = (e) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = {
      x: e.clientX,
      y: e.clientY,
      seedX: controls.seedX,
      seedY: controls.seedY,
    };
  };

  const onPointerMove = (e) => {
    if (!drag.current) return;
    const dx = (e.clientX - drag.current.x) * SEED_DRAG_SCALE;
    const dy = (e.clientY - drag.current.y) * SEED_DRAG_SCALE;
    send("seedX", drag.current.seedX + dx);
    send("seedY", drag.current.seedY + dy);
  };

  const endDrag = (e) => {
    if (e.currentTarget.hasPointerCapture?.(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    drag.current = null;
  };

  return (
    <div className="viewport">
      <canvas
        ref={canvasRef}
        className="viewport__canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      />
      <div className="viewport__status">{status}</div>
    </div>
  );
}
