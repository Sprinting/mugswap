/**
 * Cached landmark data for one clip. Kept deliberately plain so it can be
 * serialized (JSON / binary) and ported to other runtimes later.
 *
 * - Frames are sampled at a fixed `fps`; frame i covers time i / fps seconds.
 * - `landmarks[i]` is a Float32Array of 478 * 3 values (x, y, z per point) in
 *   MediaPipe's normalized image coords: x, y in [0, 1] relative to the frame
 *   width/height, z roughly in the same scale as x (depth, smaller = closer).
 *   It is null when no face was found in that frame.
 * - `matrices[i]` is the 4x4 facial transformation matrix (column-major,
 *   16 floats) from MediaPipe, or null. Used later for head yaw.
 * - Only the first detected face is kept.
 */
export interface LandmarkTrack {
  version: 1;
  width: number; // processed frame size in pixels
  height: number;
  fps: number;
  frameCount: number;
  landmarks: (Float32Array | null)[];
  matrices: (Float32Array | null)[];
}

export const NUM_LANDMARKS = 478;
