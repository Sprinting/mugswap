# mugswap

Put your face on a short clip, entirely in the browser. Vite + TypeScript, MediaPipe Face Landmarker, WebGL2.

## Run

    npm install     # also copies MediaPipe's WASM into public/mediapipe/
    npm run dev     # http://localhost:5173 (webcam needs localhost or https)
    npm run build   # -> dist/

The Face Landmarker model (`public/models/face_landmarker.task`, float16 v1) and WASM
are served from our own origin, so nothing is fetched from or sent to third parties.

## Layout

- `src/face/`: landmark logic, no UI. `landmarker.ts` wraps MediaPipe; `track.ts`
  defines the cached landmark format (documented there).
- `src/media/`: getting the selfie and clip into canvases; the frame-by-frame tracking pass.
- `src/overlay.ts`: M1 debug mesh drawing.
- `src/main.ts`: page wiring.

## Status

- [x] M1: landmarks over the selfie and over the clip, frame by frame
- [ ] M2: static warp on one frame
- [ ] M3: full clip warp with live preview
- [ ] M4: feathering, color match, mouth cutout, yaw fade
- [ ] M5: WebM export with audio, watermark, demo clips, Worker deploy
