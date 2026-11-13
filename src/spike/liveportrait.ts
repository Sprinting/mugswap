// SPIKE: does an ML face animator (TPSMM or LivePortrait) look and run well enough in the
// browser to A/B against the warp?
// Run with `npm run dev`, then open http://localhost:5173/spike.html in desktop Chrome.
import "../style.css";
import { FaceLandmarker } from "@mediapipe/tasks-vision";
import { detectImage } from "../face/landmarker";
import { loadClip, trackClip, type Clip } from "../media/clip";
import { selfieFromFile } from "../media/selfie";
import * as lp from "./lp";
import * as tp from "./tpsmm";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const msg = (id: string, text: string, error = false) => {
  $(id).textContent = text;
  $(id).classList.toggle("error", error);
};

// ---------- crops ----------

interface Rect { x: number; y: number; size: number }

// LivePortrait and TPSMM (VoxCeleb) were trained on loose head crops (the face is roughly half the
// crop). We build the same crop from MediaPipe's landmark bounding box.
const CROP_SCALE = 1.9; // crop side / face-box side
const CROP_SHIFT_UP = 0.1; // move the crop up by this fraction of its side, to include hair

function cropRect(points: Float32Array, w: number, h: number): Rect {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < points.length; i += 3) {
    x0 = Math.min(x0, points[i] * w); x1 = Math.max(x1, points[i] * w);
    y0 = Math.min(y0, points[i + 1] * h); y1 = Math.max(y1, points[i + 1] * h);
  }
  const size = Math.max(x1 - x0, y1 - y0) * CROP_SCALE;
  return { x: (x0 + x1) / 2 - size / 2, y: (y0 + y1) / 2 - size / 2 - size * CROP_SHIFT_UP, size };
}

/** Draw `rect` of `src` into a 256x256 canvas. Areas outside the source stay black. */
function cropTo256(src: CanvasImageSource, srcW: number, srcH: number, r: Rect, into: HTMLCanvasElement) {
  const ctx = into.getContext("2d", { willReadFrequently: true })!;
  const s = 256 / r.size;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, 256, 256);
  ctx.drawImage(src, -r.x * s, -r.y * s, srcW * s, srcH * s);
}

/** Moving average over neighbouring frames so the pasted face doesn't jitter with the detector. */
function smooth(rects: (Rect | null)[], radius = 2): (Rect | null)[] {
  return rects.map((r, i) => {
    if (!r) return null;
    let n = 0, x = 0, y = 0, size = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(rects.length - 1, i + radius); j++) {
      const q = rects[j];
      if (q) { n++; x += q.x; y += q.y; size += q.size; }
    }
    return { x: x / n, y: y / n, size: size / n };
  });
}

// The face-oval connections form one closed loop; walk it to get a polygon.
const OVAL = (() => {
  const next = new Map(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL.map((c) => [c.start, c.end]));
  const start = FaceLandmarker.FACE_LANDMARKS_FACE_OVAL[0].start;
  const loop = [start];
  for (let p = next.get(start)!; p !== start && loop.length < 100; p = next.get(p)!) loop.push(p);
  return loop;
})();

// ---------- state ----------

let selfie: HTMLCanvasElement | null = null; // 256x256 selfie crop
let clip: Clip | null = null;
let clipUrl: string | null = null;
let sessions: lp.Sessions | null = null;
let sessionsKey = "";
let tpSessions: tp.Sessions | null = null;
let tpPrecision: tp.Precision = "fp32";

type Model = "tpsmm" | "lp";
const model = () => $<HTMLSelectElement>("model").value as Model;
const syncModelUi = () => document.querySelectorAll<HTMLElement>(".lp-only").forEach((el) => (el.hidden = model() !== "lp"));
$<HTMLSelectElement>("model").onchange = syncModelUi;
syncModelUi();

interface Result { fps: number; frames: (ImageBitmap | null)[]; rects: (Rect | null)[]; oval: (Float32Array | null)[] }
let result: Result | null = null;

const updateRun = () => (($("run") as HTMLButtonElement).disabled = !(selfie && clip));

(async () => {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(o?: object): Promise<{ info?: { vendor?: string; architecture?: string } } | null> } }).gpu;
  if (!gpu) return msg("env", "WebGPU isn't available in this browser, so this spike can't run here. Use desktop Chrome.", true);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  msg("env", adapter ? `WebGPU ready (${adapter.info?.vendor ?? "?"} ${adapter.info?.architecture ?? ""}).` : "WebGPU present but no adapter found.", !adapter);
})();

// ---------- 1. selfie ----------

$<HTMLInputElement>("selfie-file").onchange = async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    const img = await selfieFromFile(file);
    const face = await detectImage(img);
    if (!face) return msg("selfie-msg", "No face found in that photo.", true);
    const view = $<HTMLCanvasElement>("selfie-crop");
    cropTo256(img, img.width, img.height, cropRect(face.points, img.width, img.height), view);
    view.hidden = false;
    selfie = view;
    msg("selfie-msg", "This 256 px crop is what the model sees.");
    updateRun();
  } catch (err) {
    msg("selfie-msg", (err as Error).message, true);
  }
};

// ---------- 2. clip ----------

$<HTMLInputElement>("clip-file").onchange = async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  if (clipUrl) URL.revokeObjectURL(clipUrl);
  clipUrl = URL.createObjectURL(file);
  try {
    clip = await loadClip(clipUrl);
    msg("clip-msg", `Loaded ${clip.video.duration.toFixed(1)} s at ${clip.width}×${clip.height}.`);
    updateRun();
  } catch (err) {
    msg("clip-msg", (err as Error).message, true);
  }
};

// ---------- 3. run ----------

function progress(frac: number, text: string) {
  $("progress").hidden = false;
  $("bar").style.width = `${Math.round(frac * 100)}%`;
  $("progress-text").textContent = text;
}

const seek = (v: HTMLVideoElement, t: number) =>
  new Promise<void>((res) => { v.addEventListener("seeked", () => res(), { once: true }); v.currentTime = t; });

$("run").onclick = async () => {
  if (!selfie || !clip) return;
  const run = $<HTMLButtonElement>("run");
  run.disabled = true;
  const quality = $<HTMLSelectElement>("quality").value as lp.Quality;
  const backend = $<HTMLSelectElement>("backend").value as lp.Backend;
  const fps = Number($<HTMLSelectElement>("fps").value);
  const secs = Math.min(Number($<HTMLSelectElement>("secs").value), clip.video.duration);
  const stats: string[] = [];
  let step = "loading models";
  const say = () => ($("stats").textContent = stats.join("\n"));
  try {
    if (model() === "tpsmm") {
      await runTpsmm(fps, secs, stats, say, (st) => (step = st));
      return;
    }
    // Models
    if (!sessions || sessionsKey !== quality + backend) {
      if (sessions) await lp.release(sessions);
      sessions = null;
      const t0 = performance.now();
      sessions = await lp.loadSessions(quality, backend, (l, t, label) => progress(l / t, `${label}: ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB`));
      sessionsKey = quality + backend;
      stats.push(`Models ready in ${((performance.now() - t0) / 1000).toFixed(1)} s (download or cache + GPU init)`);
      say();
    }
    const s = sessions;

    // Selfie: appearance volume + its own motion, computed once.
    step = "selfie encode";
    let t0 = performance.now();
    const selfieTensor = lp.toTensor(selfie);
    const feature = await lp.appearance(s, selfieTensor);
    const srcMotion = await lp.extractMotion(s, selfieTensor);
    const srcKp = lp.sourceKeypoints(srcMotion);
    stats.push(`Selfie encode: ${(performance.now() - t0).toFixed(0)} ms`);
    say();

    // Clip: face landmarks for the frames we'll use (reuses the M1 tracker).
    step = "clip face tracking";
    progress(0, "tracking clip face…");
    const trackClipInfo = { ...clip, duration: secs };
    const track = await trackClip(trackClipInfo, (d, t) => progress(d / t, `tracking clip face ${d}/${t}`));
    const count = Math.max(1, Math.floor(secs * fps));
    const lmAt = (i: number) => track.landmarks[Math.min(track.frameCount - 1, Math.floor(((i + 0.5) / fps) * track.fps))];
    const rects = smooth(Array.from({ length: count }, (_, i) => { const p = lmAt(i); return p ? cropRect(p, clip!.width, clip!.height) : null; }));

    // Per frame: crop -> motion -> relative keypoints -> stitch -> generate.
    const frames: (ImageBitmap | null)[] = [];
    const drive = $<HTMLCanvasElement>("crop-drive");
    const gen = $<HTMLCanvasElement>("crop-gen");
    gen.width = gen.height = s.size;
    drive.hidden = gen.hidden = false;
    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = clip.width;
    frameCanvas.height = clip.height;
    const fctx = frameCanvas.getContext("2d")!;
    let first: lp.Motion | null = null;
    const ms = { motion: 0, stitch: 0, gen: 0 };
    t0 = performance.now();
    for (let i = 0; i < count; i++) {
      const r = rects[i];
      if (!r) { frames.push(null); continue; }
      await seek(clip.video, (i + 0.5) / fps);
      fctx.drawImage(clip.video, 0, 0, clip.width, clip.height);
      cropTo256(frameCanvas, clip.width, clip.height, r, drive);

      let t = performance.now();
      step = `motion extractor (frame ${i + 1})`;
      const m = await lp.extractMotion(s, lp.toTensor(drive));
      ms.motion += performance.now() - t;
      first ??= m;

      t = performance.now();
      step = `stitching (frame ${i + 1})`;
      const kp = await lp.stitch(s, srcKp, lp.drivingKeypoints(srcMotion, m, first));
      ms.stitch += performance.now() - t;

      t = performance.now();
      step = `generator (frame ${i + 1})`;
      const img = await lp.generate(s, feature, srcKp, kp);
      ms.gen += performance.now() - t;

      gen.getContext("2d")!.putImageData(img, 0, 0);
      frames.push(await createImageBitmap(img));
      const done = i + 1;
      const per = (performance.now() - t0) / done;
      progress(done / count, `frame ${done}/${count} · ${per.toFixed(0)} ms/frame`);
    }
    const n = frames.filter(Boolean).length || 1;
    const total = performance.now() - t0;
    stats.push(
      `${count} frames (${secs} s at ${fps} fps, ${s.size} px) in ${(total / 1000).toFixed(1)} s`,
      `  per frame: motion ${(ms.motion / n).toFixed(0)} ms · stitch ${(ms.stitch / n).toFixed(0)} ms · generator ${(ms.gen / n).toFixed(0)} ms · total ${(total / n).toFixed(0)} ms`,
      `  a 10 s clip at 30 fps would take about ${((total / n) * 300 / 1000).toFixed(0)} s`,
    );
    say();
    feature.dispose();
    result = { fps, frames, rects, oval: Array.from({ length: count }, (_, i) => lmAt(i)) };
    $("out").hidden = false;
    $("player").hidden = false;
    await seek(clip.video, 0);
    drawOutput();
  } catch (err) {
    console.error(err);
    stats.push(`Error during ${step}: ${(err as Error).message}`);
    if (/lost|device/i.test((err as Error).message)) {
      stats.push("The GPU gave up (device lost). Reload the page before retrying; the models stay cached.");
      sessions = null;
      sessionsKey = "";
      tpSessions = null;
    }
    say();
  } finally {
    $("progress").hidden = true;
    run.disabled = false;
  }
};

/** Same flow as LivePortrait below: selfie once, then per frame crop -> keypoints -> generate. */
async function runTpsmm(fps: number, secs: number, stats: string[], say: () => void, setStep: (s: string) => void) {
  if (!selfie || !clip) return;
  if (!tpSessions) {
    setStep("loading models");
    tpPrecision = await tp.pickPrecision();
    const t0 = performance.now();
    tpSessions = await tp.loadSessions(tpPrecision, (l, t, label) => progress(l / t, `${label}: ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB`));
    stats.push(`TPSMM (${tpPrecision}) ready in ${((performance.now() - t0) / 1000).toFixed(1)} s (download + GPU init)`);
    say();
  }
  const s = tpSessions;

  setStep("selfie encode");
  let t0 = performance.now();
  const src = await tp.encodeSource(s, tp.toTensor(selfie));
  stats.push(`Selfie encode: ${(performance.now() - t0).toFixed(0)} ms`);
  say();

  setStep("clip face tracking");
  progress(0, "tracking clip face…");
  const track = await trackClip({ ...clip, duration: secs }, (d, t) => progress(d / t, `tracking clip face ${d}/${t}`));
  const count = Math.max(1, Math.floor(secs * fps));
  const lmAt = (i: number) => track.landmarks[Math.min(track.frameCount - 1, Math.floor(((i + 0.5) / fps) * track.fps))];
  const rects = smooth(Array.from({ length: count }, (_, i) => { const p = lmAt(i); return p ? cropRect(p, clip!.width, clip!.height) : null; }));

  const frames: (ImageBitmap | null)[] = [];
  const drive = $<HTMLCanvasElement>("crop-drive");
  const gen = $<HTMLCanvasElement>("crop-gen");
  gen.width = gen.height = 256;
  drive.hidden = gen.hidden = false;
  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = clip.width;
  frameCanvas.height = clip.height;
  const fctx = frameCanvas.getContext("2d")!;
  let first: Float32Array | null = null;
  const ms = { kp: 0, gen: 0 };
  t0 = performance.now();
  try {
    for (let i = 0; i < count; i++) {
      const r = rects[i];
      if (!r) { frames.push(null); continue; }
      await seek(clip.video, (i + 0.5) / fps);
      fctx.drawImage(clip.video, 0, 0, clip.width, clip.height);
      cropTo256(frameCanvas, clip.width, clip.height, r, drive);

      let t = performance.now();
      setStep(`keypoints (frame ${i + 1})`);
      const kp = await tp.keypoints(s, tp.toTensor(drive));
      ms.kp += performance.now() - t;
      first ??= kp;

      t = performance.now();
      setStep(`generator (frame ${i + 1})`);
      const img = await tp.generate(s, src, tp.relativeKeypoints(src.kp, kp, first));
      ms.gen += performance.now() - t;

      gen.getContext("2d")!.putImageData(img, 0, 0);
      frames.push(await createImageBitmap(img));
      const per = (performance.now() - t0) / (i + 1);
      progress((i + 1) / count, `frame ${i + 1}/${count} · ${per.toFixed(0)} ms/frame`);
    }
  } finally {
    tp.disposeSource(src);
  }
  const n = frames.filter(Boolean).length || 1;
  const total = performance.now() - t0;
  stats.push(
    `${count} frames (${secs} s at ${fps} fps, 256 px, TPSMM ${tpPrecision}) in ${(total / 1000).toFixed(1)} s`,
    `  per frame: keypoints ${(ms.kp / n).toFixed(0)} ms · generator ${(ms.gen / n).toFixed(0)} ms · total incl. seeking ${(total / n).toFixed(0)} ms`,
    `  a 10 s clip at 30 fps would take about ${((total / n) * 300 / 1000).toFixed(0)} s`,
  );
  say();
  result = { fps, frames, rects, oval: Array.from({ length: count }, (_, i) => lmAt(i)) };
  $("out").hidden = false;
  $("player").hidden = false;
  await seek(clip.video, 0);
  drawOutput();
}

// ---------- playback ----------

const out = $<HTMLCanvasElement>("out");
const layer = document.createElement("canvas");
const mask = document.createElement("canvas");

/** Clip frame + the animated selfie pasted into the face crop, feathered to the clip's face oval. */
function drawOutput() {
  if (!clip || !result) return;
  const { width: w, height: h } = clip;
  for (const c of [out, layer, mask]) if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const ctx = out.getContext("2d")!;
  ctx.drawImage(clip.video, 0, 0, w, h);
  const i = Math.min(result.frames.length - 1, Math.floor(clip.video.currentTime * result.fps));
  const frame = result.frames[i], r = result.rects[i], oval = result.oval[i];
  if (!frame || !r || !oval || $<HTMLInputElement>("show-orig").checked) return;

  // Mask: the clip's face oval, blurred for a soft edge.
  const m = mask.getContext("2d")!;
  m.clearRect(0, 0, w, h);
  m.filter = `blur(${Math.max(2, r.size / 40)}px)`;
  m.beginPath();
  OVAL.forEach((p, k) => (k ? m.lineTo : m.moveTo).call(m, oval[p * 3] * w, oval[p * 3 + 1] * h));
  m.fill();
  m.filter = "none";

  const l = layer.getContext("2d")!;
  l.globalCompositeOperation = "source-over";
  l.clearRect(0, 0, w, h);
  l.drawImage(frame, r.x, r.y, r.size, r.size);
  l.globalCompositeOperation = "destination-in";
  l.drawImage(mask, 0, 0);
  ctx.drawImage(layer, 0, 0);
}

function loop() {
  if (!clip || clip.video.paused) return;
  if (result && clip.video.currentTime >= result.frames.length / result.fps) clip.video.currentTime = 0;
  drawOutput();
  requestAnimationFrame(loop);
}

$("play").onclick = async () => {
  if (!clip) return;
  if (clip.video.paused) { await clip.video.play(); $("play").textContent = "Pause"; loop(); }
  else { clip.video.pause(); $("play").textContent = "Play"; drawOutput(); }
};
$<HTMLInputElement>("show-orig").onchange = drawOutput;
