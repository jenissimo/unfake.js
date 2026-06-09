/**
 * Shared codec for the unfake-core WASM packed image format:
 *   [u32 LE width, u32 LE height, RGBA bytes...]
 *
 * Returns a payload whose `buffer` contains ONLY the pixels at offset 0,
 * which is what `payloadToImageData` (below) expects. The copy
 * also makes the buffer cleanly transferable back to the main thread.
 */
/** ImageData -> worker payload (copies the buffer so the original isn't detached). */
export function imageDataToPayload(img) {
  if (!img) return null;
  const copy = new Uint8ClampedArray(img.data);
  return { width: img.width, height: img.height, buffer: copy.buffer };
}

/** Worker payload (pixels at offset 0) -> ImageData. */
export function payloadToImageData(payload) {
  if (!payload) return null;
  const expected = payload.width * payload.height * 4;
  const bytes = new Uint8ClampedArray(payload.buffer, 0, expected);
  return new ImageData(bytes, payload.width, payload.height);
}

export function unpackCoreImage(packed) {
  const view = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  const width = view.getUint32(0, true);
  const height = view.getUint32(4, true);
  const pixels = new Uint8ClampedArray(
    packed.buffer,
    packed.byteOffset + 8,
    width * height * 4,
  );
  const copy = pixels.slice(); // fresh buffer, offset 0, exact length
  return { width, height, buffer: copy.buffer };
}
