// Thin-Plate Spline Motion Model (Zhao & Zhang, CVPR 2022, MIT) in the browser via
// ONNX Runtime Web + WebGPU. SPIKE CODE.
//
// Pipeline (see github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model, demo.py, "relative" mode):
//   kp detector(crop)          -> 50 keypoints in [-1, 1] (10 thin-plate splines x 5 points)
//   source encoder(selfie)     -> feature maps of the selfie, computed once
//   per frame: relative keypoints = selfie kp + (frame kp - first frame kp) * scale
//              solve the 10 thin-plate splines (tiny 8x8 systems, done here in JS)
//              frame model(selfie features, keypoints, splines) -> animated selfie crop
//
// Everything is 2D (Conv, GridSample on 4D tensors, InstanceNorm), so it runs fully on
// WebGPU. The ONNX files are exported from the vox 256 checkpoint by scripts/tpsmm/
// (fp16 or fp32 weights; inputs and outputs are fp32 either way) and served from
// public/models/tpsmm/.
import * as ort from "onnxruntime-web/webgpu";

const strict = new URLSearchParams(location.search).get("strict") === "1";
const BASE = "/models/tpsmm/";

// Files over 50 MB are split into parts (GitHub refuses files over 100 MB).
const FILES = {
  fp16: {
    kp: { parts: ["tpsmm_kp_fp16.onnx"], bytes: 22_458_473 },
    src: { parts: ["tpsmm_src_fp16.onnx"], bytes: 3_128_168 },
    frame: { parts: ["aa", "ab", "ac"].map((p) => `tpsmm_frame_fp16.onnx.part-${p}`), bytes: 122_307_183 },
  },
  fp32: {
    kp: { parts: ["tpsmm_kp.onnx"], bytes: 44_904_379 },
    src: { parts: ["tpsmm_src.onnx"], bytes: 6_250_200 },
    frame: { parts: ["aa", "ab", "ac", "ad", "ae"].map((p) => `tpsmm_frame.onnx.part-${p}`), bytes: 244_542_861 },
  },
} as const;
type Key = "kp" | "src" | "frame";

/**
 * fp16 halves the download and is faster on GPUs with half-precision shaders (Apple
 * Silicon, most recent GPUs). Falls back to fp32 when the GPU lacks "shader-f16".
 * Force with ?precision=fp32 or ?precision=fp16.
 */
export type Precision = "fp16" | "fp32";
export async function pickPrecision(): Promise<Precision> {
  const forced = new URLSearchParams(location.search).get("precision");
  if (forced === "fp16" || forced === "fp32") return forced;
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } }).gpu;
  const adapter = await gpu?.requestAdapter();
  return adapter?.features.has("shader-f16") ? "fp16" : "fp32";
}

async function fetchModel(precision: Precision, key: Key, onBytes: (n: number) => void): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  for (const part of FILES[precision][key].parts) {
    const res = await fetch(BASE + part);
    if (!res.ok) throw new Error(`Download failed for ${part}: HTTP ${res.status}. Did you git pull?`);
    const reader = res.body!.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      onBytes(value.byteLength);
    }
  }
  const out = new Uint8Array(chunks.reduce((n, c) => n + c.byteLength, 0));
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  return out;
}

export interface Sessions { kp: ort.InferenceSession; src: ort.InferenceSession; frame: ort.InferenceSession }

export async function loadSessions(precision: Precision, onProgress: (loaded: number, total: number, label: string) => void): Promise<Sessions> {
  if (!("gpu" in navigator)) throw new Error("WebGPU isn't available in this browser. Use desktop Chrome.");
  const keys: Key[] = ["kp", "src", "frame"];
  const total = keys.reduce((n, k) => n + FILES[precision][k].bytes, 0);
  let loaded = 0;
  const s: Partial<Sessions> = {};
  for (const k of keys) {
    const bytes = await fetchModel(precision, k, (n) => { loaded += n; onProgress(loaded, total, `loading ${k}`); });
    onProgress(loaded, total, `starting ${k} on WebGPU`);
    const opts: ort.InferenceSession.SessionOptions = {
      executionProviders: [{ name: "webgpu" } as never],
      graphOptimizationLevel: "all",
      // The selfie's feature maps stay on the GPU and feed the frame model directly.
      preferredOutputLocation: k === "src" ? "gpu-buffer" : "cpu",
    };
    if (strict) opts.extra = { session: { disable_cpu_ep_fallback: "1" } };
    console.log(`[spike] creating TPSMM session ${k}`);
    s[k] = await ort.InferenceSession.create(bytes, opts);
  }
  return s as Sessions;
}

export async function release(s: Sessions) {
  for (const x of [s.kp, s.src, s.frame]) await x.release().catch(() => {});
}

const tensor = (data: Float32Array, dims: number[]) => new ort.Tensor("float32", data, dims);

/** Canvas -> [1,3,256,256] float tensor in [0,1], channel-first. */
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

/** 50 keypoints (x, y in [-1, 1]) for a 256 px crop. */
export async function keypoints(s: Sessions, img: ort.Tensor): Promise<Float32Array> {
  const r = await s.kp.run({ image: img });
  const kp = Float32Array.from((await r.kp.getData()) as Float32Array);
  r.kp.dispose();
  return kp;
}

export interface Source { image: ort.Tensor; kp: Float32Array; feats: Record<string, ort.Tensor> }

/** Encode the selfie once: keypoints plus feature maps (kept on the GPU). */
export async function encodeSource(s: Sessions, image: ort.Tensor): Promise<Source> {
  const kp = await keypoints(s, image);
  const feats = await s.src.run({ source: image });
  return { image, kp, feats };
}

export function disposeSource(src: Source) {
  for (const t of Object.values(src.feats)) t.dispose();
}

/** Area of the convex hull of 50 2D points (Andrew's monotone chain + shoelace). */
function hullArea(kp: Float32Array): number {
  const pts: [number, number][] = [];
  for (let i = 0; i < kp.length; i += 2) pts.push([kp[i], kp[i + 1]]);
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list: [number, number][]) => {
    const h: [number, number][] = [];
    for (const p of list) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop();
      h.push(p);
    }
    h.pop();
    return h;
  };
  const hull = [...half(pts), ...half([...pts].reverse())];
  let a = 0;
  for (let i = 0; i < hull.length; i++) {
    const [x0, y0] = hull[i], [x1, y1] = hull[(i + 1) % hull.length];
    a += x0 * y1 - x1 * y0;
  }
  return Math.abs(a) / 2;
}

/**
 * The repo's "relative" mode: the selfie keeps its own face shape and only the clip's
 * *movement* since a reference frame is applied, scaled by relative face size. `gain` > 1
 * exaggerates the movement (TPSMM tends to under-open mouths).
 */
export function relativeKeypoints(src: Float32Array, drv: Float32Array, drv0: Float32Array, gain = 1): Float32Array {
  const scale = (gain * Math.sqrt(hullArea(src))) / Math.sqrt(hullArea(drv0));
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = src[i] + (drv[i] - drv0[i]) * scale;
  return out;
}

/**
 * Solve the 10 thin-plate splines that map driving keypoints to source keypoints
 * (modules/util.py, TPS mode "kp"). Each is an 8x8 linear system; ONNX has no matrix
 * inverse that runs on WebGPU, so it's done here. Returns [10, 8, 2].
 */
export function solveTps(kpDriving: Float32Array, kpSource: Float32Array): Float32Array {
  const out = new Float32Array(10 * 8 * 2);
  for (let t = 0; t < 10; t++) {
    const p = (i: number) => [kpDriving[(t * 5 + i) * 2], kpDriving[(t * 5 + i) * 2 + 1]];
    // L = [[K, P], [P^T, 0]] + 0.01 I, with K_ij = r^2 log r^2 and P rows (x, y, 1)
    const A: number[][] = Array.from({ length: 8 }, () => new Array(10).fill(0));
    for (let i = 0; i < 5; i++) {
      const [xi, yi] = p(i);
      for (let j = 0; j < 5; j++) {
        const [xj, yj] = p(j);
        const r2 = (xi - xj) ** 2 + (yi - yj) ** 2;
        A[i][j] = r2 * Math.log(r2 + 1e-9);
      }
      A[i][5] = A[5][i] = xi;
      A[i][6] = A[6][i] = yi;
      A[i][7] = A[7][i] = 1;
      A[i][8] = kpSource[(t * 5 + i) * 2]; // right-hand side Y = [kp_source; 0]
      A[i][9] = kpSource[(t * 5 + i) * 2 + 1];
    }
    for (let i = 0; i < 8; i++) A[i][i] += 0.01;
    // Gauss-Jordan with partial pivoting
    for (let c = 0; c < 8; c++) {
      let piv = c;
      for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]];
      const d = A[c][c];
      for (let k = c; k < 10; k++) A[c][k] /= d;
      for (let r = 0; r < 8; r++) {
        if (r === c || A[r][c] === 0) continue;
        const f = A[r][c];
        for (let k = c; k < 10; k++) A[r][k] -= f * A[c][k];
      }
    }
    for (let i = 0; i < 8; i++) { out[(t * 8 + i) * 2] = A[i][8]; out[(t * 8 + i) * 2 + 1] = A[i][9]; }
  }
  return out;
}

/** One output frame as RGBA ImageData (256x256). `kpDriving` should come from relativeKeypoints. */
export async function generate(s: Sessions, src: Source, kpDriving: Float32Array): Promise<ImageData> {
  const r = await s.frame.run({
    source: src.image,
    ...src.feats,
    kp_driving: tensor(kpDriving, [1, 50, 2]),
    kp_source: tensor(src.kp, [1, 50, 2]),
    tps: tensor(solveTps(kpDriving, src.kp), [1, 10, 8, 2]),
  });
  const d = (await r.out.getData()) as Float32Array;
  r.out.dispose();
  const plane = 256 * 256;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    rgba[i * 4] = d[i] * 255;
    rgba[i * 4 + 1] = d[plane + i] * 255;
    rgba[i * 4 + 2] = d[plane * 2 + i] * 255;
    rgba[i * 4 + 3] = 255;
  }
  return new ImageData(rgba, 256, 256);
}
