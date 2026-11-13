import { defineConfig } from "vite";

// Cross-origin isolation lets ONNX Runtime's wasm backend use threads (SharedArrayBuffer).
// Model fetches from Hugging Face are CORS requests, so require-corp doesn't block them.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  server: { headers: isolation },
  preview: { headers: isolation },
});
