// Copies MediaPipe's WASM runtime into public/ so it is served from our own
// origin (no third-party CDN, nothing leaves the browser).
import { cpSync, mkdirSync } from "node:fs";
const src = "node_modules/@mediapipe/tasks-vision/wasm";
const dst = "public/mediapipe/wasm";
mkdirSync(dst, { recursive: true });
for (const f of ["vision_wasm_internal.js", "vision_wasm_internal.wasm"]) cpSync(`${src}/${f}`, `${dst}/${f}`);
console.log("copied MediaPipe wasm ->", dst);
