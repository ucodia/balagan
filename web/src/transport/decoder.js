// H.264 decode via WebCodecs. The server emits High@4.0; configuration is
// deferred until the first keyframe so a mid-GOP join starts cleanly.
export const CODEC = "avc1.640028";

export function createDecoder(canvas, onPaint) {
  const ctx = canvas.getContext("2d");
  let configured = false;

  const decoder = new VideoDecoder({
    output: (frame) => {
      if (canvas.width !== frame.displayWidth || canvas.height !== frame.displayHeight) {
        canvas.width = frame.displayWidth;
        canvas.height = frame.displayHeight;
      }
      ctx.drawImage(frame, 0, 0);
      frame.close();
      onPaint && onPaint();
    },
    error: (err) => console.error("VideoDecoder error:", err),
  });

  return {
    decode({ isKeyframe, timestampMs, body }) {
      if (!configured) {
        if (!isKeyframe) return;
        decoder.configure({ codec: CODEC, optimizeForLatency: true });
        configured = true;
      }
      decoder.decode(
        new EncodedVideoChunk({
          type: isKeyframe ? "key" : "delta",
          timestamp: timestampMs * 1000,
          data: body,
        }),
      );
    },
    close() {
      if (decoder.state !== "closed") decoder.close();
    },
  };
}
