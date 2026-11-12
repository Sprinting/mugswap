// Getting the selfie into a canvas, from a file or the webcam.
const MAX_SELFIE = 1024;

export function toCanvas(src: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const s = Math.min(1, MAX_SELFIE / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.round(w * s);
  c.height = Math.round(h * s);
  c.getContext("2d")!.drawImage(src, 0, 0, c.width, c.height);
  return c;
}

export async function selfieFromFile(file: File): Promise<HTMLCanvasElement> {
  // imageOrientation: respect EXIF rotation from phone photos.
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" }).catch(() => {
    throw new Error("Couldn't read that image. Try a JPG or PNG.");
  });
  const c = toCanvas(bmp, bmp.width, bmp.height);
  bmp.close();
  return c;
}

export async function startWebcam(video: HTMLVideoElement): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function snapWebcam(video: HTMLVideoElement): HTMLCanvasElement {
  return toCanvas(video, video.videoWidth, video.videoHeight);
}
