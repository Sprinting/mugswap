// Cameos-style compositing helpers for the spike: make the generated face sit in the
// clip's lighting rather than look like a sticker. SPIKE CODE.
//
//  1. Colour match: per frame, shift the generated face's per-channel mean and contrast
//     (Reinhard-style, in RGB) to the clip actor's face under the same lighting.
//     Stats are smoothed over neighbouring frames so the colour doesn't flicker.
//  2. Upscale + sharpen: the model outputs 256 px; we resize once to the paste size
//     with high-quality smoothing and add an unsharp mask so it's not softer than the clip.
//  3. The blend edge itself is handled at paste time (faceMask below).

export interface Stats { mean: [number, number, number]; std: [number, number, number] }

// Sample region: an ellipse over the central face (cheeks, nose, eyes, mouth) of a 256 px
// head crop as built by cropRect (face box ~135 px, centred ~26 px below crop centre).
const CX = 128, CY = 150, RX = 46, RY = 58;

export function faceStats(px: Uint8ClampedArray): Stats {
  const sum = [0, 0, 0], sq = [0, 0, 0];
  let n = 0;
  for (let y = CY - RY; y <= CY + RY; y++) {
    for (let x = CX - RX; x <= CX + RX; x++) {
      if (((x - CX) / RX) ** 2 + ((y - CY) / RY) ** 2 > 1) continue;
      const o = (y * 256 + x) * 4;
      for (let c = 0; c < 3; c++) { sum[c] += px[o + c]; sq[c] += px[o + c] ** 2; }
      n++;
    }
  }
  const mean = sum.map((s) => s / n) as Stats["mean"];
  const std = sq.map((s, c) => Math.sqrt(Math.max(1, s / n - mean[c] ** 2))) as Stats["std"];
  return { mean, std };
}

/** Moving average of stats over +-radius frames (nulls skipped). */
export function smoothStats(list: (Stats | null)[], radius = 4): (Stats | null)[] {
  return list.map((s, i) => {
    if (!s) return null;
    const m = [0, 0, 0], d = [0, 0, 0];
    let n = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(list.length - 1, i + radius); j++) {
      const q = list[j];
      if (!q) continue;
      for (let c = 0; c < 3; c++) { m[c] += q.mean[c]; d[c] += q.std[c]; }
      n++;
    }
    return { mean: m.map((v) => v / n), std: d.map((v) => v / n) } as Stats;
  });
}

/** Recolour `img` in place so its face stats move from `from` to `to`. */
export function colorMatch(img: ImageData, from: Stats, to: Stats, strength = 1) {
  const a = [0, 0, 0], b = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    // Clamp the contrast change: big swings come from shadows or hair in the sample.
    const k = Math.min(1.5, Math.max(0.67, to.std[c] / from.std[c]));
    a[c] = 1 + (k - 1) * strength;
    b[c] = (to.mean[c] - from.mean[c] * k) * strength; // lerp between identity and (x - m1) * k + m2
  }
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    for (let c = 0; c < 3; c++) d[i + c] = d[i + c] * a[c] + b[c];
  }
}

const tmp = document.createElement("canvas");
tmp.width = tmp.height = 256;

/** 256 px ImageData -> bitmap at `size` px (capped at 640), high-quality resize + unsharp mask. */
export async function upscaleSharpen(img: ImageData, size: number, amount = 0.7): Promise<ImageBitmap> {
  const s = Math.max(64, Math.min(640, Math.round(size)));
  tmp.getContext("2d")!.putImageData(img, 0, 0);
  const big = new OffscreenCanvas(s, s);
  const g = big.getContext("2d", { willReadFrequently: true })!;
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  g.drawImage(tmp, 0, 0, s, s);
  const sharp = g.getImageData(0, 0, s, s);
  // Blur radius ~ one source pixel at the new scale.
  const blurCanvas = new OffscreenCanvas(s, s);
  const bg = blurCanvas.getContext("2d", { willReadFrequently: true })!;
  bg.filter = `blur(${Math.max(1, s / 256)}px)`;
  bg.drawImage(big, 0, 0);
  const blur = bg.getImageData(0, 0, s, s).data;
  const d = sharp.data;
  for (let i = 0; i < d.length; i += 4) {
    for (let c = 0; c < 3; c++) d[i + c] = d[i + c] + amount * (d[i + c] - blur[i + c]);
  }
  g.putImageData(sharp, 0, 0);
  return createImageBitmap(big);
}
