// Loading a clip and running the landmark pass over it, frame by frame.
import { detectVideoFrame } from "../face/landmarker";
import type { LandmarkTrack } from "../face/track";

export const MAX_SECONDS = 10;
export const FPS = 30; // fixed sampling rate; source fps isn't exposed by the browser
const MAX_LONG = 1280; // 720p cap: 1280 on the long side, 720 on the short side
const MAX_SHORT = 720;

export interface Clip {
  video: HTMLVideoElement;
  duration: number; // seconds actually used (<= MAX_SECONDS)
  width: number; // processing/render size (downscaled if needed)
  height: number;
}

export function fitSize(w: number, h: number) {
  const s = Math.min(1, MAX_LONG / Math.max(w, h), MAX_SHORT / Math.min(w, h));
  // Even dimensions keep video encoders happy later.
  return { width: Math.round((w * s) / 2) * 2, height: Math.round((h * s) / 2) * 2 };
}

export async function loadClip(url: string): Promise<Clip> {
  const video = document.createElement("video");
  video.src = url;
  video.playsInline = true;
  video.preload = "auto";
  video.crossOrigin = "anonymous";
  await new Promise<void>((res, rej) => {
    video.onloadeddata = () => res();
    video.onerror = () => rej(new Error("This video can't be decoded by your browser. Try an MP4 or WebM."));
  });
  if (!video.videoWidth) throw new Error("This file has no video track.");
  const duration = Math.min(video.duration || MAX_SECONDS, MAX_SECONDS);
  return { video, duration, ...fitSize(video.videoWidth, video.videoHeight) };
}

function seek(video: HTMLVideoElement, t: number) {
  return new Promise<void>((res) => {
    video.addEventListener("seeked", () => res(), { once: true });
    video.currentTime = t;
  });
}

/**
 * Seek through the clip one frame at a time and run Face Landmarker on each.
 * Seeking (rather than playing) is slower but deterministic and identical in
 * Chrome and Firefox; each await also yields to the event loop, so the tab
 * never freezes.
 */
export async function trackClip(
  clip: Clip,
  onProgress: (done: number, total: number) => void,
  signal?: AbortSignal,
): Promise<LandmarkTrack> {
  const frameCount = Math.max(1, Math.floor(clip.duration * FPS));
  const canvas = document.createElement("canvas");
  canvas.width = clip.width;
  canvas.height = clip.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: false })!;
  const track: LandmarkTrack = {
    version: 1, width: clip.width, height: clip.height, fps: FPS, frameCount,
    landmarks: [], matrices: [],
  };
  clip.video.pause();
  for (let i = 0; i < frameCount; i++) {
    if (signal?.aborted) throw new DOMException("Cancelled", "AbortError");
    // Sample the middle of each frame slot so float rounding never lands us on the previous frame.
    await seek(clip.video, (i + 0.5) / FPS);
    ctx.drawImage(clip.video, 0, 0, canvas.width, canvas.height);
    const face = await detectVideoFrame(canvas, 1000 / FPS);
    track.landmarks.push(face?.points ?? null);
    track.matrices.push(face?.matrix ?? null);
    onProgress(i + 1, frameCount);
  }
  await seek(clip.video, 0);
  return track;
}

export const frameAt = (t: number) => Math.floor(t * FPS);
