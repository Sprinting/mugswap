// LivePortrait in the browser via ONNX Runtime Web + WebGPU. SPIKE CODE.
//
// Pipeline (see the LivePortrait paper / KwaiVGI repo):
//   appearance extractor(selfie crop)  -> feature_3d (a 3D feature volume of the selfie)
//   motion extractor(crop)             -> pitch/yaw/roll bins, translation t, expression exp,
//                                         scale, canonical keypoints kp (21 x 3)
//   keypoints = scale * (kp @ R + exp) + t
//   stitching MLP nudges the driving keypoints so the face edge stays put
//   generator(feature_3d, kp_source, kp_driving) -> animated selfie crop
//
// The per-frame math is ported from MartinDelophy/ai-video-editor
// (src/workers/liveportrait.worker.js, MIT), which runs the same ONNX exports in
// the browser. The models are the WebGPU-friendly exports that project hosts at
// huggingface.co/haixin/timeline-studio-onnx-models.
import * as ort from "onnxruntime-web/webgpu";
import ortWasmMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";

// Threads need cross-origin isolation (COOP/COEP headers, set in vite.config.ts).
// Single-threaded by default: with 8 threads, session creation hung on an M-series Mac
// in Chrome (not reproducible in headless Chrome). Try more with ?threads=N in the URL.
const wanted = Number(new URLSearchParams(location.search).get("threads")) || 1;
ort.env.wasm.numThreads = crossOriginIsolated ? wanted : 1;
export const wasmThreads = ort.env.wasm.numThreads;
ort.env.wasm.wasmPaths = { mjs: ortWasmMjsUrl, wasm: ortWasmUrl };
if (ort.env.webgpu) ort.env.webgpu.powerPreference = "high-performance";

const BASE =
  "https://huggingface.co/haixin/timeline-studio-onnx-models/resolve/a201b681c8f96672b5c3f624e32d4dc932f150af/liveportrait-webgpu/";
// Plain fp32 exports of the warping module + SPADE generator, for the CPU (wasm) fallback.
const CPU_BASE = "https://huggingface.co/dyicnc/Live-Portrait-ONNX/resolve/e6c5d2407593a39f29c92ffd5ea3eaf5e59d52a1/";

const FILES = {
  appearance: { parts: ["liveportrait-appearance_feature_extractor.onnx"], bytes: 3_355_896 },
  motion: {
    parts: ["liveportrait-motion-extractor.onnx.part-aa", "liveportrait-motion-extractor.onnx.part-ab", "liveportrait-motion-extractor.onnx.part-ac"],
    bytes: 112_593_241,
  },
  stitching: { parts: ["liveportrait-stitching.onnx"], bytes: 182_363 },
  preview: {
    parts: ["aa", "ab", "ac", "ad", "ae"].map((p) => `liveportrait-generator-preview-fp16.onnx.part-${p}?v=5fdb50d2`),
    bytes: 210_713_705,
  },
  quality: {
    parts: ["aa", "ab", "ac", "ad", "ae"].map((p) => `liveportrait-generator-quality-fp16.onnx.part-${p}?v=1b4630bf`),
    bytes: 210_713_678,
  },
  cpuWarping: { parts: [CPU_BASE + "warping.onnx"], bytes: 182_274_422 },
  cpuSpade: { parts: [CPU_BASE + "spade_generator.onnx"], bytes: 221_924_849 },
} as const;
type ModelKey = keyof typeof FILES;

export type Quality = "preview" | "quality";
const CACHE = "mugswap-models-v1";

/** Download (or read from Cache Storage) one model, reporting bytes as they arrive. */
async function fetchModel(key: ModelKey, onBytes: (n: number, label: string) => void): Promise<Uint8Array> {
  const cache = await caches.open(CACHE).catch(() => null);
  const chunks: Uint8Array[] = [];
  for (const part of FILES[key].parts) {
    const url = part.startsWith("https://") ? part : BASE + part;
    let res = (await cache?.match(url)) ?? null;
    const label = res ? "reading cache" : "downloading";
    if (!res) {
      const net = await fetch(url);
      if (!net.ok) throw new Error(`Download failed for ${part}: HTTP ${net.status}`);
      await cache?.put(url, net.clone()).catch(() => {}); // quota errors are fine; we just re-download next time
      res = net;
    }
    const reader = res.body!.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      onBytes(value.byteLength, label);
    }
  }
  const total = chunks.reduce((n, c) => n + c.byteLength, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  return out;
}

export interface Sessions {
  appearance: ort.InferenceSession;
  motion: ort.InferenceSession;
  stitching: ort.InferenceSession;
  generator?: ort.InferenceSession; // GPU path: warp + render fused in one fp16 model
  warping?: ort.InferenceSession; // CPU path: separate fp32 warp...
  spade?: ort.InferenceSession; // ...and renderer
  size: number; // generator output size in px
  backend: Backend;
}

/**
 * "gpu": only the heavy generator runs on WebGPU; the small encoders and MLP run
 *        on the CPU so the GPU holds one big model, not three.
 * "cpu": everything on the CPU (wasm). Very slow, but shows the quality on
 *        machines where WebGPU falls over.
 */
export type Backend = "gpu" | "cpu";

export async function loadSessions(quality: Quality, backend: Backend, onProgress: (loaded: number, total: number, label: string) => void): Promise<Sessions> {
  if (backend === "gpu" && !("gpu" in navigator)) throw new Error("WebGPU isn't available in this browser. Try desktop Chrome, or pick the CPU backend.");
  const heavy: ModelKey[] = backend === "gpu" ? [quality] : ["cpuWarping", "cpuSpade"];
  const keys: ModelKey[] = ["appearance", "motion", "stitching", ...heavy];
  const total = keys.reduce((n, k) => n + FILES[k].bytes, 0);
  let loaded = 0;
  const s: Partial<Record<ModelKey, ort.InferenceSession>> = {};
  for (const k of keys) {
    const bytes = await fetchModel(k, (n, how) => { loaded += n; onProgress(loaded, total, `${how} ${k}`); });
    const ep = backend === "gpu" && k === quality ? "WebGPU" : `CPU, ${ort.env.wasm.numThreads} thread(s)`;
    onProgress(loaded, total, `starting ${k} on ${ep}`);
    console.log(`[spike] creating session ${k} (${ep})`);
    // The generator is conv-heavy; NHWC is the faster layout for ORT's WebGPU kernels.
    const provider = backend === "gpu" && k === quality ? { name: "webgpu", preferredLayout: "NHWC" } : "wasm";
    const opts: ort.InferenceSession.SessionOptions = { executionProviders: [provider as never], graphOptimizationLevel: "all" };
    s[k] = await ort.InferenceSession.create(bytes, opts);
    console.log(`[spike] session ${k} ready`);
  }
  return {
    appearance: s.appearance!, motion: s.motion!, stitching: s.stitching!,
    generator: s[quality], warping: s.cpuWarping, spade: s.cpuSpade,
    size: backend === "cpu" ? 512 : quality === "quality" ? 512 : 256,
    backend,
  };
}

export async function release(s: Sessions) {
  for (const x of [s.appearance, s.motion, s.stitching, s.generator, s.warping, s.spade]) await x?.release().catch(() => {});
}

// ---------- motion math ----------

export interface Motion { pitch: number; yaw: number; roll: number; t: Float32Array; exp: Float32Array; scale: number; kp: Float32Array }

/** Head pose comes out as 66 logits over 3-degree bins; take the softmax expectation. */
function binsToDegrees(logits: ArrayLike<number>): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  let sum = 0, w = 0;
  for (let i = 0; i < logits.length; i++) { const e = Math.exp(logits[i] - max); sum += e; w += e * i; }
  return (w / sum) * 3 - 97.5;
}

const mul3 = (a: Float32Array, b: Float32Array) => {
  const o = new Float32Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return o;
};
const transpose3 = (m: Float32Array) => new Float32Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]);

/** LivePortrait's rotation convention: R = (Rz Ry Rx)^T, applied as row-vector kp @ R. */
function rotation(pitch: number, yaw: number, roll: number): Float32Array {
  const [x, y, z] = [pitch, yaw, roll].map((d) => (d * Math.PI) / 180);
  const rx = new Float32Array([1, 0, 0, 0, Math.cos(x), -Math.sin(x), 0, Math.sin(x), Math.cos(x)]);
  const ry = new Float32Array([Math.cos(y), 0, Math.sin(y), 0, 1, 0, -Math.sin(y), 0, Math.cos(y)]);
  const rz = new Float32Array([Math.cos(z), -Math.sin(z), 0, Math.sin(z), Math.cos(z), 0, 0, 0, 1]);
  return transpose3(mul3(rz, mul3(ry, rx)));
}

export async function extractMotion(sess: Sessions, img: ort.Tensor): Promise<Motion> {
  const r = await sess.motion.run({ img });
  const m: Motion = {
    pitch: binsToDegrees(r.pitch.data as Float32Array),
    yaw: binsToDegrees(r.yaw.data as Float32Array),
    roll: binsToDegrees(r.roll.data as Float32Array),
    t: Float32Array.from(r.t.data as Float32Array),
    exp: Float32Array.from(r.exp.data as Float32Array),
    scale: (r.scale.data as Float32Array)[0],
    kp: Float32Array.from(r.kp.data as Float32Array),
  };
  Object.values(r).forEach((v) => v.dispose());
  return m;
}

/** keypoints = scale * (kp @ R + exp) + t   (t only moves x and y) */
export function keypoints(kp: Float32Array, R: Float32Array, exp: Float32Array, scale: number, t: Float32Array): Float32Array {
  const out = new Float32Array(63);
  for (let p = 0; p < 21; p++) {
    const o = p * 3;
    for (let a = 0; a < 3; a++) out[o + a] = scale * (kp[o] * R[a] + kp[o + 1] * R[3 + a] + kp[o + 2] * R[6 + a] + exp[o + a]);
    out[o] += t[0];
    out[o + 1] += t[1];
  }
  return out;
}

export const sourceKeypoints = (m: Motion) => keypoints(m.kp, rotation(m.pitch, m.yaw, m.roll), m.exp, m.scale, m.t);

/**
 * Driving keypoints, relative to the clip's first frame: the selfie keeps its own
 * shape, and only the *change* in head rotation and expression is transferred.
 * Scale and translation are deliberately NOT transferred, because we paste the
 * result back into a crop that already follows the clip's face around.
 */
export function drivingKeypoints(src: Motion, drv: Motion, drv0: Motion): Float32Array {
  const rDrv = rotation(drv.pitch, drv.yaw, drv.roll);
  const rDrv0 = rotation(drv0.pitch, drv0.yaw, drv0.roll);
  const rSrc = rotation(src.pitch, src.yaw, src.roll);
  const R = mul3(mul3(rDrv, transpose3(rDrv0)), rSrc);
  const exp = new Float32Array(63);
  for (let i = 0; i < 63; i++) exp[i] = src.exp[i] + drv.exp[i] - drv0.exp[i];
  return keypoints(src.kp, R, exp, src.scale, src.t);
}

const tensor = (data: Float32Array, dims: number[]) => new ort.Tensor("float32", data, dims);

/** Stitching MLP: predicts a small correction so the animated face blends with the un-animated crop edge. */
export async function stitch(sess: Sessions, src: Float32Array, drv: Float32Array): Promise<Float32Array> {
  const input = new Float32Array(126);
  input.set(src);
  input.set(drv, 63);
  const r = await sess.stitching.run({ input: tensor(input, [1, 126]) });
  const d = r.output.data as Float32Array;
  const out = Float32Array.from(drv);
  for (let i = 0; i < 63; i++) out[i] += d[i];
  for (let p = 0; p < 21; p++) { out[p * 3] += d[63]; out[p * 3 + 1] += d[64]; }
  r.output.dispose();
  return out;
}

export async function appearance(sess: Sessions, img: ort.Tensor): Promise<ort.Tensor> {
  const r = await sess.appearance.run({ img });
  return r[sess.appearance.outputNames[0]];
}

/** Returns an RGBA ImageData of the animated crop. */
export async function generate(sess: Sessions, feature: ort.Tensor, src: Float32Array, drv: Float32Array): Promise<ImageData> {
  const feeds = { feature_3d: feature, kp_source: tensor(src, [1, 21, 3]), kp_driving: tensor(drv, [1, 21, 3]) };
  let out: ort.Tensor;
  if (sess.generator) {
    out = (await sess.generator.run(feeds))[sess.generator.outputNames[0]];
  } else {
    const warped = (await sess.warping!.run(feeds))[sess.warping!.outputNames[0]];
    out = (await sess.spade!.run({ [sess.spade!.inputNames[0]]: warped }))[sess.spade!.outputNames[0]];
    warped.dispose();
  }
  const [, , h, w] = out.dims;
  const d = (await out.getData()) as Float32Array;
  out.dispose();
  const plane = w * h;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    rgba[i * 4] = d[i] * 255;
    rgba[i * 4 + 1] = d[plane + i] * 255;
    rgba[i * 4 + 2] = d[plane * 2 + i] * 255;
    rgba[i * 4 + 3] = 255;
  }
  return new ImageData(rgba, w, h);
}

/** Canvas (any size) -> [1,3,256,256] float tensor in [0,1], channel-first. */
export function toTensor(c: HTMLCanvasElement): ort.Tensor {
  const px = c.getContext("2d")!.getImageData(0, 0, 256, 256).data;
  const plane = 256 * 256;
  const chw = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i++) {
    chw[i] = px[i * 4] / 255;
    chw[plane + i] = px[i * 4 + 1] / 255;
    chw[plane * 2 + i] = px[i * 4 + 2] / 255;
  }
  return tensor(chw, [1, 3, 256, 256]);
}
