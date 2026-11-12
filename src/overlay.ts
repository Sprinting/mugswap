// M1 debug view: draw the 478-point mesh on a 2D canvas.
import { FaceLandmarker } from "@mediapipe/tasks-vision";

const TESS = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
const OVAL = FaceLandmarker.FACE_LANDMARKS_FACE_OVAL;
const LIPS = FaceLandmarker.FACE_LANDMARKS_LIPS;

function lines(ctx: CanvasRenderingContext2D, p: Float32Array, conns: { start: number; end: number }[], w: number, h: number) {
  ctx.beginPath();
  for (const { start, end } of conns) {
    ctx.moveTo(p[start * 3] * w, p[start * 3 + 1] * h);
    ctx.lineTo(p[end * 3] * w, p[end * 3 + 1] * h);
  }
  ctx.stroke();
}

export function drawLandmarks(ctx: CanvasRenderingContext2D, p: Float32Array) {
  const { width: w, height: h } = ctx.canvas;
  const lw = Math.max(1, w / 800);
  ctx.save();
  ctx.lineWidth = lw * 0.6;
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  lines(ctx, p, TESS, w, h);
  ctx.lineWidth = lw * 1.5;
  ctx.strokeStyle = "#ffd400";
  lines(ctx, p, OVAL, w, h);
  ctx.strokeStyle = "#ff4f7b";
  lines(ctx, p, LIPS, w, h);
  ctx.restore();
}
