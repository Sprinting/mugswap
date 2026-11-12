// Thin wrapper around MediaPipe Face Landmarker. No DOM/UI code here beyond
// what MediaPipe needs as input, so the logic can be ported later.
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import { NUM_LANDMARKS } from "./track";

const BASE = import.meta.env.BASE_URL;
let instance: Promise<FaceLandmarker> | null = null;
let mode: "IMAGE" | "VIDEO" = "IMAGE";
// VIDEO mode requires strictly increasing timestamps across the lifetime of the
// landmarker, so we keep a running clock rather than restarting at 0 per clip.
let videoClockMs = 0;

// Cache the promise, not the result, so concurrent callers share one instance.
export function getLandmarker(): Promise<FaceLandmarker> {
  return (instance ??= create());
}

async function create(): Promise<FaceLandmarker> {
  // WASM + model are served from our own origin (see scripts/copy-mediapipe-wasm.mjs).
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}mediapipe/wasm`);
  const opts = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: `${BASE}models/face_landmarker.task`, delegate },
    runningMode: "IMAGE" as const,
    numFaces: 1, // first face only
    outputFacialTransformationMatrixes: true,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch (e) {
    // Some browsers/drivers can't create the WebGL context MediaPipe's GPU delegate needs.
    console.warn("GPU delegate failed, falling back to CPU", e);
    return FaceLandmarker.createFromOptions(fileset, opts("CPU"));
  }
}

async function setMode(m: "IMAGE" | "VIDEO") {
  const lm = await getLandmarker();
  if (mode !== m) {
    await lm.setOptions({ runningMode: m });
    mode = m;
  }
  return lm;
}

/** Flatten the first face of a result into the cache format (478*3 floats). */
export function firstFace(r: FaceLandmarkerResult): { points: Float32Array; matrix: Float32Array | null } | null {
  const face = r.faceLandmarks[0];
  if (!face || face.length < NUM_LANDMARKS) return null;
  const points = new Float32Array(NUM_LANDMARKS * 3);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    points[i * 3] = face[i].x;
    points[i * 3 + 1] = face[i].y;
    points[i * 3 + 2] = face[i].z;
  }
  const m = r.facialTransformationMatrixes?.[0];
  return { points, matrix: m ? Float32Array.from(m.data) : null };
}

export async function detectImage(src: TexImageSource) {
  const lm = await setMode("IMAGE");
  return firstFace(lm.detect(src as HTMLCanvasElement));
}

/** Detect one frame of a sequence. Call with frames in order. */
export async function detectVideoFrame(src: TexImageSource, frameMs: number) {
  const lm = await setMode("VIDEO");
  videoClockMs += Math.max(1, frameMs);
  return firstFace(lm.detectForVideo(src as HTMLCanvasElement, videoClockMs));
}
