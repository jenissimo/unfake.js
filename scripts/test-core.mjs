#!/usr/bin/env bun
/**
 * Core glue tests (self-contained, no external fixtures).
 *
 * Round-trips the packed [u32 w, u32 h, RGBA...] format the WASM core returns
 * through unpackCoreImage and asserts the pixels land at offset 0 — i.e. what
 * payloadToImageData reads. Guards the worker <-> main-thread glue where the
 * off-by-8 bug lived.
 */
import { unpackCoreImage } from '../browser-tool/lib/core-codec.js';

let failed = 0;

{
  const w = 3;
  const h = 2;
  const pixels = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < pixels.length; i++) pixels[i] = (i * 7 + 1) & 0xff;

  const packed = new Uint8Array(8 + w * h * 4);
  const dv = new DataView(packed.buffer);
  dv.setUint32(0, w, true);
  dv.setUint32(4, h, true);
  packed.set(pixels, 8);

  const img = unpackCoreImage(packed);
  const got = new Uint8ClampedArray(img.buffer, 0, w * h * 4); // as payloadToImageData reads
  const ok = img.width === w && img.height === h
    && Array.from(got).join(',') === Array.from(pixels).join(',');
  if (!ok) {
    console.error(`FAIL core-codec round-trip: ${img.width}x${img.height} pixels=${Array.from(got)}`);
    failed++;
  }
}

if (failed) {
  console.error(`core checks: ${failed} failure(s)`);
  process.exit(1);
}
console.log('core checks: OK (core-codec round-trip)');
