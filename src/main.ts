import "./style.css";
import { detectImage, getLandmarker } from "./face/landmarker";
import type { LandmarkTrack } from "./face/track";
import { FPS, frameAt, loadClip, trackClip, type Clip } from "./media/clip";
import { selfieFromFile, snapWebcam, startWebcam } from "./media/selfie";
import { drawLandmarks } from "./overlay";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// Warm up the model in the background so the first detection is quick.
getLandmarker().catch((e) => showMsg("selfie-msg", `Couldn't load the face model: ${e.message ?? e}`, true));

function showMsg(id: string, text: string, error = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", error);
}

// ---------- 1. Selfie ----------
let camStream: MediaStream | null = null;

async function useSelfie(canvas: HTMLCanvasElement) {
  const view = $<HTMLCanvasElement>("selfie-view");
  view.width = canvas.width;
  view.height = canvas.height;
  const ctx = view.getContext("2d")!;
  ctx.drawImage(canvas, 0, 0);
  view.hidden = false;
  showMsg("selfie-msg", "Looking for a face…");
  const face = await detectImage(canvas);
  if (!face) {
    showMsg("selfie-msg", "No face found. Try a well-lit, front-facing photo with your whole face visible.", true);
    return;
  }
  drawLandmarks(ctx, face.points);
  showMsg("selfie-msg", "Face found.");
}

$<HTMLInputElement>("selfie-file").onchange = async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  stopCam();
  try {
    await useSelfie(await selfieFromFile(file));
  } catch (err) {
    showMsg("selfie-msg", (err as Error).message, true);
  }
};

$("cam-start").onclick = async () => {
  try {
    camStream = await startWebcam($<HTMLVideoElement>("cam"));
    $("cam").hidden = false;
    $("cam-snap").hidden = false;
    $("selfie-view").hidden = true;
    showMsg("selfie-msg", "");
  } catch {
    showMsg("selfie-msg", "Couldn't open the webcam. Check the browser's camera permission.", true);
  }
};

$("cam-snap").onclick = async () => {
  const snap = snapWebcam($<HTMLVideoElement>("cam"));
  stopCam();
  await useSelfie(snap);
};

function stopCam() {
  camStream?.getTracks().forEach((t) => t.stop());
  camStream = null;
  $("cam").hidden = true;
  $("cam-snap").hidden = true;
}

// ---------- 2. Clip ----------
let clip: Clip | null = null;
let track: LandmarkTrack | null = null;
let abort: AbortController | null = null;
let clipUrl: string | null = null;
const view = $<HTMLCanvasElement>("clip-view");
const scrub = $<HTMLInputElement>("scrub");

$<HTMLInputElement>("clip-file").onchange = async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  abort?.abort();
  clip?.video.pause();
  if (clipUrl) URL.revokeObjectURL(clipUrl);
  clipUrl = URL.createObjectURL(file);
  $("player").hidden = true;
  view.hidden = true;
  showMsg("clip-msg", "");
  try {
    clip = await loadClip(clipUrl);
    const note = clip.video.duration > 10.05 ? ` Using the first 10 s of ${clip.video.duration.toFixed(1)} s.` : "";
    showMsg("clip-msg", `Tracking faces at ${clip.width}×${clip.height}…${note}`);
    $("progress").hidden = false;
    abort = new AbortController();
    const t0 = performance.now();
    track = await trackClip(clip, (done, total) => {
      $("bar").style.width = `${(100 * done) / total}%`;
      $("progress-text").textContent = `${done} / ${total} frames`;
    }, abort.signal);
    const hits = track.landmarks.filter(Boolean).length;
    showMsg("clip-msg",
      hits === 0
        ? "No face found anywhere in this clip. Try one with a clear, mostly front-facing face."
        : `Face found in ${hits} of ${track.frameCount} frames (${((performance.now() - t0) / 1000).toFixed(1)} s).${note}`,
      hits === 0);
    view.width = clip.width;
    view.height = clip.height;
    view.hidden = false;
    scrub.max = String(track.frameCount - 1);
    $("player").hidden = false;
    drawFrame(0);
  } catch (err) {
    if ((err as Error).name !== "AbortError") showMsg("clip-msg", (err as Error).message, true);
  } finally {
    $("progress").hidden = true;
  }
};

/** Draw the current video frame plus the cached landmarks for frame i. */
function drawFrame(i: number) {
  if (!clip || !track) return;
  const ctx = view.getContext("2d")!;
  ctx.drawImage(clip.video, 0, 0, view.width, view.height);
  const p = track.landmarks[i];
  if (p) drawLandmarks(ctx, p);
  scrub.value = String(i);
  $("frame-label").textContent = `frame ${i + 1}/${track.frameCount}${p ? "" : " (no face)"}`;
}

function loop() {
  if (!clip || !track || clip.video.paused) return;
  if (clip.video.currentTime >= clip.duration) clip.video.currentTime = 0; // loop within the 10 s cap
  drawFrame(Math.min(frameAt(clip.video.currentTime), track.frameCount - 1));
  requestAnimationFrame(loop);
}

$("play").onclick = async () => {
  if (!clip) return;
  if (clip.video.paused) {
    await clip.video.play();
    $("play").textContent = "Pause";
    loop();
  } else {
    clip.video.pause();
    $("play").textContent = "Play";
  }
};

scrub.oninput = () => {
  if (!clip) return;
  clip.video.pause();
  $("play").textContent = "Play";
  const i = Number(scrub.value);
  clip.video.addEventListener("seeked", () => drawFrame(i), { once: true });
  clip.video.currentTime = (i + 0.5) / FPS;
};
