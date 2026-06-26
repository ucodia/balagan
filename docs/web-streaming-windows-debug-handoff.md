# Web Streaming (QUIC) — Windows Choppiness: Debugging Handoff

> **Purpose.** Hand this file to a fresh session to continue debugging the
> `--output web` (WebTransport/QUIC) streaming choppiness on Windows **without
> re-deriving anything**. It records the symptom, everything proven, everything
> ruled out (with evidence), the code changes already applied (uncommitted on
> branch `feat-quic-output`), the diagnostic tooling (embedded in full), and the
> prioritized next steps. Read `docs/web-streaming-architecture.md` first for the
> design; this doc assumes it.

> **Process note for the next session.** Earlier in this investigation we burned
> a lot of effort on wrong theories asserted without isolation (a synthetic
> Python loopback client that *cannot reproduce Chrome's flow-control behaviour*,
> see below). The turning point was building a controlled repro that drives a
> **real headless Chrome** against a **real `WebStreamOutput`** with **synthetic
> frames** (no GPU), so every variable is controlled. **Use that harness. Measure
> before claiming. The remaining open question is best answered by measuring the
> browser's paint intervals directly — which has NOT been done yet.**

---

## 1. Symptom

- `--output web` streams to a browser over WebTransport/HTTP3 (aioquic) + WebCodecs.
- **macOS: flawless. Windows (RTX 4090): broken.**
- Evolution across this session's fixes:
  1. Originally: connects, runs ~1 s, then decays to ~0 fps while the engine
     stays at its FPS cap.
  2. After load-shedding + 8 Mbps default + encode-rate cap: streams at ~30 fps
     but **bursty** — "renders at 30 fps then blocks for a moment then continues,"
     and "unlimited FPS → very slow."
  3. After longer-GOP + force-keyframe-on-connect + token-bucket cap:
     **still bursty / "still broken"** (last user report — root cause of the
     residual burst NOT yet pinned).

The platform split is the recurring root of issues on this project: **dev on
macOS, runs on Windows RTX 4090** (see the user's memory notes). A key historical
fact: on macOS the StyleGAN inference caps render at **~9.5 fps**; on the 4090 it
hits the full **30 fps** cap (or more if uncapped). Much of the Windows-only
breakage traces to the higher render rate exercising paths macOS never hit.

---

## 2. How to reproduce (real app)

```
uv run balagan --snapshots-dir C:\Users\ucodia\Desktop\american-nightmare\best\ --output web --headless
```
Then open `http://127.0.0.1:8000` in Chrome. Default `--web-bitrate` is now 8 Mbps
(was 25 Mbps). The server prints `WEBDIAG` lines once per second (temporary
instrumentation, see §6).

**Better: reproduce without the GPU** using the controlled harness in §7. It runs
the real `WebStreamOutput` + the real built web client (`web/dist`) with synthetic
frames and lets you drive headless Chrome and read `WEBDIAG`. This is how all the
controlled results below were obtained.

---

## 3. Architecture facts that matter (why this is fragile)

From `docs/web-streaming-architecture.md` and the code:

- **One unidirectional QUIC stream per frame.** A stream is only reclaimed from
  `quic._streams` once it `is_finished` = `receiver.is_finished AND
  sender.is_finished`. `sender.is_finished` requires the client to **ACK the
  stream data + FIN** (aioquic `quic/stream.py` `on_data_delivered`). The
  `_finish_send_only_stream` hack in `web_stream.py` forces only the *receiver*
  half; the sender half still needs the ACK round-trip.
- **WebTransport flow control is gated on the browser app's reads.** Chrome only
  extends `MAX_DATA` (connection-level flow control) as the JS client **reads**
  the stream bytes. If the JS reader can't keep up, credit isn't released, the
  server's `_remote_max_data_used` pins at `_remote_max_data`, and the server
  can't push more frame data or FINs → streams never finish → unbounded backlog.
- **The browser client reads incoming streams strictly sequentially**
  (`web/src/transport/connection.js` `readIncoming` → `await readStream(...)`
  before accepting the next stream). One stream per frame at 30 fps is a lot of
  per-stream churn.
- **Render thread and QUIC event-loop thread share one GIL.** The render loop
  (`engine.render_frame`) encodes on the calling thread (`WebStreamOutput.send`)
  and hands bytes to the asyncio loop via `call_soon_threadsafe`.
- **Windows asyncio = `ProactorEventLoop`**, and `time.sleep`/loop timer
  granularity is ~15 ms (vs sub-ms on macOS kqueue). This shows up as a measured
  **loopback RTT of ~15 ms** in aioquic — not microseconds.
- Encoder: **libx264** default on all platforms, Annex B in-band SPS/PPS,
  `tune=zerolatency preset=superfast`.

---

## 4. What is PROVEN (with evidence)

All "real Chrome" rows below are from the §7 harness: real `WebStreamOutput`,
real `web/dist` client, headless Chrome, **synthetic 1024² frames, no GPU**, read
off the server-side `WEBDIAG`.

### 4.1 The collapse is a throughput ceiling, NOT stream count, NOT a low-rate defect

| settings (real Chrome) | streams/sec created | result |
|---|---|---|
| 30 fps @ **2 Mbps** | ~40 | ✅ `streams`≈8, healthy |
| 30 fps @ **8 Mbps** | ~40 | ✅ `streams`≈8–12, healthy |
| 5 fps @ 8 Mbps | ~15 | ✅ healthy |
| 30 fps @ **12 Mbps** | ~40 | ⚠️ slow leak (`streams` climbs, Chrome scrambles to extend `maxData`) |
| 30 fps @ **25 Mbps** | ~40 | ❌ collapse: `streams`→325+, `fcLeft`→0 |

The decisive pair: **30 fps @ 2 Mbps (healthy) vs 30 fps @ 25 Mbps (collapse)** —
identical ~40 streams/sec, only the byte rate differs. So the controlling variable
is **aggregate bytes/sec**, and the browser WebTransport path here sustains only
**~8–10 Mbps**. The default 25 Mbps overran it.

Failure signature when over the ceiling (`WEBDIAG`): `sent/s≈40`, `failed/s=0`,
`pending=0/8`, `rtt=2 ms`, but `streams` climbs without bound, `ackfin=0`
(no per-frame stream FIN ever ACKed once behind), `usedData`→`maxData`,
`fcLeft`→0, `blockedUni` climbs. The render thread meanwhile stays at its cap.

**Why macOS is fine:** ~9.5 fps render × ~104 KB/frame ≈ ~8 Mbps — under the
ceiling. The 4090 at 30 fps × the 25 Mbps target ≈ 25 Mbps — over it.

### 4.2 The server had no working load-shedding (a real bug)

The render→loop `deque(maxlen=8)` drop-oldest **never engages** (`pending` stays
0) because `_drain` "successfully" hands each frame to a QUIC stream that then
can't flush (flow-control blocked). So the backlog grows in `quic._streams`
instead of being dropped. Fixed (§5).

### 4.3 Uncapped fps floods and the shed breaks decode

At uncapped render (~41 fps on the 4090, real-app `WEBDIAG`), aggregate exceeds
the ceiling, the load-shedder fires (`failed/s` 5–10), and dropping **delta**
frames breaks the H.264 decode chain → browser mostly frozen between keyframes.
Fix: cap the *encode* rate so we never overdrive (§5).

### 4.4 The encoder bitstream IS low-latency — NOT the §8.2 VideoToolbox issue

`ffmpeg trace_headers` on the actual Windows libx264 output (script in §7.4):
```
max_num_ref_frames        = 1
max_num_reorder_frames    = 0   ← decoder may output each frame immediately
max_dec_frame_buffering   = 1
bitstream_restriction_flag= 1
profile High (100), level 3.1, SPS in-band per keyframe (Annex B)
```
This is the same low-latency signaling that fixed macOS (commit
`7b6cdafbcee358e3c3693cac4e99a8d771cf06d7`). The decoder-buffering/IDR-stall
theory is **ruled out**.

### 4.5 The keyframe spike is real

`measure_kf.py` (§7.3), 8 Mbps, smooth GAN-like content, GOP was 60 (2 s):
```
keyframe mean 92 KB / max 126 KB   vs   delta mean 32 KB   (2.9x)
keyframe ship time @8 Mbps = 94 ms ≈ 2.8 frame-times
```
So every 2 s a keyframe consumes ~3 frames of link time → a ~60 ms hitch on a 2 s
cadence. Matches "blocks for a moment." Mitigated (longer GOP + force-on-connect,
§5) but **may still be visible every ~5 s** — needs client-side confirmation.

---

## 5. What is RULED OUT (with how)

| Hypothesis | Test performed | Result |
|---|---|---|
| OS UDP socket buffers too small | Set `SO_RCVBUF`/`SO_SNDBUF` = 8 MB on both server and client sockets | **No effect** — still collapses at 25 Mbps |
| Windows timer resolution | `ctypes.windll.winmm.timeBeginPeriod(1)` | **No effect** — RTT still ~15 ms, still collapses |
| ProactorEventLoop UDP weakness | Forced `WindowsSelectorEventLoopPolicy` | **No effect** |
| Stream *count* / stream rate | 30 fps @ 2 Mbps = same ~40 streams/sec as failing 25 Mbps | **Healthy** → not stream count |
| Pure render-rate ("Windows too fast") | 30 fps @ 8 Mbps healthy; only byte rate matters | It's bytes/sec, not fps |
| Encoder VUI / decoder buffering (§8.2) | `ffmpeg trace_headers` on real output | Low-latency VUI present → ruled out |
| Synthetic Python loopback client repro | `repro_quic.py` (in-process aioquic client) | **Cannot reproduce** the Chrome failure — aioquic auto-extends flow control regardless of app reads, whereas Chrome gates it on JS reads. **Do not trust this client for flow-control behaviour.** Use real Chrome (§7). |

---

## 6. Code changes already applied (UNCOMMITTED, branch `feat-quic-output`)

All in `src/balagan/`. Run `git diff` to see them. **Includes temporary
instrumentation that must be removed before merge.**

### 6.1 `cli.py`
- `--web-bitrate` default **25_000_000 → 8_000_000** (keeps steady 30 fps under
  the ~8–10 Mbps ceiling). Help text updated.

### 6.2 `io/video_encoder.py`
- `config_for`: `keyframe_interval` **`fps*2` → `fps*5`** (longer GOP → fewer
  keyframe spikes).
- `VideoEncoder.request_keyframe()`: sets `self._force_keyframe`; next `encode()`
  sets `frame.pict_type = av.video.frame.PictureType.I` (verified to force an IDR
  — see §7.5). Thread-safe single-flag.

### 6.3 `io/web_stream.py`
- **Load-shedding** (`_MAX_OUTSTANDING_STREAMS = 24`): in `_Protocol.send_frame`,
  if `len(self._quic._streams) > 24` and the frame is a non-keyframe, non-state
  delta, drop it (don't open a stream). Bounds the backlog; keyframes + state
  always sent. *Proven to bound 25 Mbps to `streams`≈26 instead of 325+.*
- **Token-bucket encode-rate cap** (replaces an earlier jitter-fragile per-frame
  deadline): `WebStreamOutput.send()` refills `self._tokens` at `fps`/sec
  (cap 2.0), drops the frame *before* encoding when `< 1` token. Caps uncapped
  render to the target fps with a continuous bitstream; jitter-robust. *Proven:
  120 fps feed → ~30 fps out, bounded, no shedding.*
- **Force keyframe on connect**: `register()` calls `self._encoder.request_keyframe()`
  so a new viewer starts immediately despite the long GOP.
- **⚠️ TEMP DIAGNOSTICS — REMOVE BEFORE MERGE** (all marked `# TEMP`):
  - `self._diag_sent/_diag_failed/_diag_last_exc` counters.
  - `_diagnostics()` coroutine + `self._diag_task` (logs the `WEBDIAG` line/sec).
  - `send_frame`'s `except Exception` was un-swallowed to record the real
    exception in `_diag_last_exc` and increment `_diag_failed` (drops).
  - These print `WEBDIAG ...` at INFO; remove the task, the counters, the
    `register`/`send` increments, and restore the quiet `except`.

### 6.4 Tests
- `uv run pytest tests/test_web_stream.py tests/test_video_encoder.py` → 14 pass,
  **1 pre-existing failure**: `test_hardware_encoder_is_tuned_and_usable` (NVENC).
  Confirmed failing on committed code too (`git stash` + run). Cause: the test's
  skip-guard wraps `VideoEncoder(...)` construction, but PyAV opens the codec
  lazily on first `encode()` (outside the guard), so an unusable NVENC raises
  instead of skipping. Unrelated to this work; the web path uses libx264.

---

## 7. Diagnostic tooling (embedded — recreate in a scratch dir)

The originals lived in a session scratchpad that does not persist. Recreate these.
They depend only on the project env (`uv run --project <repo> python <script>`).

### 7.1 `WEBDIAG` line format (from the temp `_diagnostics` in `web_stream.py`)

```
WEBDIAG sent/s=N failed/s=N pending=N/8 subs=N
        [streams=N rcvfin=N sndfin=N ackfin=N remoteMaxUni=N blockedUni=N
         maxData=N usedData=N fcLeft=N]
```
- `sent/s` includes ~10 state-push streams/sec + the video frames.
- `failed/s` = frames dropped by load-shedding (or send errors).
- `streams` = `len(quic._streams)` — **the key signal**; should stay single digits.
- `ackfin` = streams whose sender FIN is ACKed. Stays 0 when the client is behind.
- `fcLeft` = `maxData - usedData` = connection flow-control credit left. →0 = stalled.
- `blockedUni` = server is blocked from opening more uni streams (hit Chrome's limit).

### 7.2 `repro_server.py` — real server + real client + synthetic frames

Drive headless Chrome at `http://127.0.0.1:8000` and read its `WEBDIAG`. Env vars:
`REPRO_BITRATE`, `REPRO_FPS` (encoder fps), `REPRO_FEED_FPS` (render rate; set
> FPS to test the encode-rate cap), `REPRO_DURATION`.

```python
import logging, os, sys, threading, time
from pathlib import Path
import numpy as np
from balagan.core.runtime_state import RuntimeState
from balagan.io.dev_cert import generate_self_signed_cert
from balagan.io.web_stream import WebStreamOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
WIDTH = int(os.environ.get("REPRO_W", "1024")); HEIGHT = int(os.environ.get("REPRO_H", "1024"))
BITRATE = int(os.environ.get("REPRO_BITRATE", "25000000"))
FPS = int(os.environ.get("REPRO_FPS", "30")); FEED_FPS = int(os.environ.get("REPRO_FEED_FPS", str(FPS)))
DURATION_S = int(os.environ.get("REPRO_DURATION", "30"))
WEB_DIST = Path(__file__).resolve().parents[0] / "web" / "dist"  # adjust to repo/web/dist

def _frame(i):
    rng = (i * 2654435761) & 0xFFFFFFFF
    b = np.full((HEIGHT, WIDTH, 3), rng & 0xFF, dtype=np.uint8)
    b[:: (i % 7) + 1, :, 0] = (i * 13) & 0xFF; b[:, :: (i % 5) + 1, 1] = (i * 31) & 0xFF
    return b

def _feed(output, stop):
    period = 1.0 / FEED_FPS; i = 0; nxt = time.perf_counter()
    while not stop.is_set():
        output.send(_frame(i)); i += 1; nxt += period
        d = nxt - time.perf_counter()
        if d > 0: time.sleep(d)
        else: nxt = time.perf_counter()

def main():
    tmp = Path("scratch_certs"); tmp.mkdir(exist_ok=True)
    cert, key = tmp / "cert.pem", tmp / "key.pem"; generate_self_signed_cert(cert, key)
    state = RuntimeState(); state.update(fps_cap=FPS)
    output = WebStreamOutput("repro", WIDTH, HEIGHT, cert=cert, key=key, port=4433,
        fps=FPS, bitrate=BITRATE, codec="libx264", runtime_state=state,
        web_dir=WEB_DIST, ui_port=8000, ui_host="127.0.0.1")
    print(f"SERVER UP {WIDTH}x{HEIGHT} {BITRATE}bps feed={FEED_FPS} enc={FPS} -> http://127.0.0.1:8000", flush=True)
    stop = threading.Event(); t = threading.Thread(target=_feed, args=(output, stop), daemon=True); t.start()
    try: time.sleep(DURATION_S)
    finally: stop.set(); t.join(timeout=2); output.close()

if __name__ == "__main__": main()
```

Drive Chrome (Windows, separate process; kill with `taskkill /F /IM chrome.exe /T`):
```
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new `
  --user-data-dir=<fresh-dir> --no-first-run `
  --autoplay-policy=no-user-gesture-required "http://127.0.0.1:8000"
```
WebTransport + WebCodecs work in `--headless=new`. The JS read loop runs even if
decode has issues, so `WEBDIAG` reflects the real Chrome read/flow-control path.

### 7.3 `measure_kf.py` — keyframe vs delta sizes (no network)

```python
import numpy as np
from balagan.io.video_encoder import DEFAULT_WEB_CODEC, VideoEncoder, config_for
cfg = config_for(DEFAULT_WEB_CODEC, fps=30, bitrate=8_000_000); enc = VideoEncoder(1024, 1024, cfg)
yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float32); kf, delta = [], []
for i in range(150):
    p = i * 0.05
    r = (np.sin(xx/90 + p)*.5+.5)*255; g = (np.cos(yy/110 + p*.7)*.5+.5)*255
    bch = (np.sin((xx+yy)/130 + p*1.3)*.5+.5)*255
    f = np.stack([r, g, bch], -1).astype(np.uint8)
    for c in enc.encode(f): (kf if c.is_keyframe else delta).append(len(c.data))
print("gop", cfg.keyframe_interval, "kf KB", np.mean(kf)/1024, "delta KB", np.mean(delta)/1024)
```

### 7.4 Bitstream VUI check — authoritative via ffmpeg `trace_headers`

```python
# dump_stream.py — write a sample Annex B stream
import numpy as np
from balagan.io.video_encoder import DEFAULT_WEB_CODEC, VideoEncoder, config_for
enc = VideoEncoder(512, 512, config_for(DEFAULT_WEB_CODEC, fps=30, bitrate=8_000_000))
with open("sample.h264", "wb") as f:
    for i in range(20):
        for c in enc.encode(np.random.randint(0,255,(512,512,3),dtype=np.uint8)): f.write(c.data)
    for c in enc.close(): f.write(c.data)
```
```
ffmpeg=$(uv run python -c "import imageio_ffmpeg,sys; sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())")
"$ffmpeg" -hide_banner -loglevel trace -i sample.h264 -c copy -bsf:v trace_headers -f null - 2>&1 \
  | grep -iE "max_num_reorder|max_dec_frame|max_num_ref|bitstream_restriction"
```

### 7.5 Forced-keyframe sanity check

```python
import av, numpy as np
ctx = av.CodecContext.create("libx264","w"); ctx.width=ctx.height=256; ctx.pix_fmt="yuv420p"
ctx.framerate=30; ctx.bit_rate=2_000_000; ctx.gop_size=1000
ctx.options={"preset":"superfast","tune":"zerolatency"}
for i in range(12):
    f = av.VideoFrame.from_ndarray(np.full((256,256,3),i*10,dtype=np.uint8),format="rgb24"); f.pts=i
    if i==6: f.pict_type = av.video.frame.PictureType.I
    for p in ctx.encode(f): print(i, p.is_keyframe, p.size)   # frame 6 -> is_keyframe True
```

---

## 8. REMAINING UNSOLVED: still bursty at capped 30 fps / 8 Mbps

After §6's fixes the stream runs ~30 fps but the user still reports it bursty
("still broken"). At steady 30 fps @ 8 Mbps the **server side looks clean** in
`WEBDIAG` (`streams`≈8, `failed/s=0`, `fcLeft` healthy). So the residual burst is
**not** visible from the server-side metrics gathered so far. Candidate causes,
none yet confirmed by client-side measurement:

1. **Residual keyframe spike** (now every ~5 s instead of 2 s). Visible as a
   periodic hitch. Would track the GOP.
2. **Delivery jitter from the Windows asyncio loop** (~15 ms timer quantization).
   aioquic may flush datagrams in ~15 ms batches, so frames arrive at Chrome in
   bursts even though `sent/s` averages 40. Not visible in 1 s `WEBDIAG`
   aggregates.
3. **GIL contention**: at the 30 fps cap the render thread sleeps (frees GIL), but
   under load the encode (libx264, on the render thread) + inference may still
   starve the loop thread in bursts.
4. **Windows power-profile throttle** (SEPARATE, known): the real-app trace showed
   the *engine* dropping to 1–4 fps for ~1 s every ~90 s — matches the user's
   memory note "periodic deep FPS drops = AWCC/OS power switching, not engine
   code." Set the power plan to **Ultimate Performance** / disable core parking to
   remove this confound before judging stream smoothness.
5. **Client-side decode/paint scheduling**: `web/src/transport/decoder.js` draws
   each frame immediately on `VideoDecoder` output with no presentation-time
   pacing, so bursty network arrival → bursty paint.

---

## 9. Recommended next steps (priority order)

1. **MEASURE CLIENT-SIDE PAINT INTERVALS — do this first.** Stop inferring
   smoothness from the server. Add temporary logging to
   `web/src/hooks/useStream.js` `onPaint` (record `performance.now()` deltas; log
   any gap > ~50 ms with whether the painted frame was a keyframe), rebuild
   (`cd web && npm run build`), and capture Chrome's console from headless with
   `--enable-logging=stderr --v=1` (console.log surfaces on stderr). This tells
   you the **exact cadence** of the burst: ~every 5 s → keyframe (lengthen GOP
   more / pace the keyframe); ~33 ms±jitter clusters → delivery/loop jitter;
   ~every 90 s deep → power throttle. **This single measurement disambiguates
   §8.1–8.5.**
2. **Rule out the power throttle** (set Ultimate Performance) so it's not confused
   with the stream.
3. If keyframe: try GOP `fps*10`+ and/or pace the keyframe across a few sends; or
   request a keyframe after a shed (rate-limited to ≤1/s) for fast recovery.
4. If delivery jitter from the loop: investigate aioquic pacing on Proactor; try
   the Selector loop *for latency smoothness* (we only tested it for the
   collapse, where it didn't matter — burst smoothness is a different question);
   consider draining the deque on a tighter cadence.
5. **Consider abandoning one-stream-per-frame** for a **single long-lived
   unidirectional stream with length-prefixed frames** (server writes
   `[len][frame]…` to one stream; client reads it continuously). This removes the
   entire fragile per-frame stream machinery (no reclamation, no per-frame flow
   control, no MAX_STREAMS limit, far less client per-stream churn) at the cost of
   head-of-line blocking *within* the stream — a non-issue on a lossless
   loopback/LAN. This is the highest-leverage architectural change if the burst
   turns out to be per-stream delivery overhead. (`docs/web-streaming-architecture.md`
   §9.3/§12 discuss related framing options.)

---

## 10. Cleanup checklist before this work can merge

- [ ] **Remove all `# TEMP` WEBDIAG instrumentation** from `io/web_stream.py`
      (`_diagnostics`, `_diag_*` counters, `_diag_task`, the `send_frame`
      increments) and **restore the quiet `except Exception`** (but consider
      keeping it at `WARNING` with rate-limiting rather than `debug`, since it hid
      this whole class of failure).
- [ ] Add regression tests: (a) load-shed keeps `streams` bounded under an
      over-ceiling load; (b) the encode-rate cap caps a fast feed to ~fps;
      (c) `request_keyframe()` forces an IDR mid-stream; (d) a new subscriber gets
      a keyframe promptly.
- [ ] Decide whether 8 Mbps is the right default or whether to expose/clamp it
      per-deployment (the ~8–10 Mbps ceiling may differ on a real LAN vs loopback
      — **the ceiling was only measured on this Windows box over loopback**).
- [ ] Note (don't necessarily fix here): the pre-existing
      `test_hardware_encoder_is_tuned_and_usable` NVENC skip-guard bug (§6.4).
- [ ] Re-validate on macOS that none of these changes regress the working path
      (longer GOP, encode-rate cap, load-shed thresholds).

---

## 11. One-paragraph status for the impatient

The Windows web stream had **three stacked real bugs**, all now mitigated: (1) the
server pushed the default 25 Mbps into a browser path that only sustains ~8–10
Mbps and had **no working load-shedding**, so per-frame QUIC streams piled up
unbounded → 0 fps (fixed: load-shed + 8 Mbps default); (2) uncapped fps flooded
and the shed dropped delta frames, breaking decode (fixed: token-bucket encode
cap); (3) a keyframe every 2 s caused a ~60 ms hitch (mitigated: GOP→5 s +
force-keyframe-on-connect). The encoder bitstream was **proven low-latency** (not
the §8.2 VideoToolbox issue). **Still bursty** after all that, and the server-side
metrics are clean at steady state — so the next move is to **measure the browser's
paint intervals directly** (§9.1) to identify the residual cadence, which has not
yet been done. All controlled results were obtained by driving **real headless
Chrome** against a **synthetic-frame server** (§7) — reuse that harness; do not
trust the pure-Python loopback client for flow-control behaviour.
