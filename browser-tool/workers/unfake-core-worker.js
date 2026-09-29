/**
 * unfake-core WASM worker (detect / snap / downscale — no OpenCV).
 */
import init, {
  detect_tiled,
  detect_legacy,
  detect_runs,
  detect_auto,
  crop_offset,
  detect_grid,
  grid_for_scale_vec,
  elastic_bounds_vec,
  snap_rgba_packed,
  downscale_rgba_packed,
  downscale_grid_packed,
  downscale_bounds_packed,
  morph_cleanup_packed,
  detect_color_count,
  content_adaptive_packed,
  vector_preprocess_packed,
  vector_fill_bg_packed,
  vector_postprocess_packed,
  quantize_packed,
  encode_png_bytes,
} from '../vendor/unfake-core/unfake_wasm.js';
import { unpackCoreImage } from '../lib/core-codec.js';

let ready = false;

function rgbaView(image) {
  return new Uint8Array(image.buffer, 0, image.width * image.height * 4);
}

/** [periodX, periodY, offsetX, offsetY, cellsX, cellsY] → object, empty → null. */
function gridFromVec(v) {
  if (!v || v.length < 6) return null;
  return { periodX: v[0], periodY: v[1], offsetX: v[2], offsetY: v[3], cellsX: v[4], cellsY: v[5] };
}

const OPS = {
  detectTiled({ image }) {
    return {
      type: 'number',
      value: detect_tiled(rgbaView(image), image.width, image.height),
    };
  },
  detectLegacy({ image }) {
    return {
      type: 'number',
      value: detect_legacy(rgbaView(image), image.width, image.height),
    };
  },
  detectRuns({ image }) {
    return {
      type: 'number',
      value: detect_runs(rgbaView(image), image.width, image.height),
    };
  },
  detectAuto({ image }) {
    return {
      type: 'number',
      value: detect_auto(rgbaView(image), image.width, image.height),
    };
  },
  detectGrid({ image }) {
    return { type: 'grid', grid: gridFromVec(detect_grid(rgbaView(image), image.width, image.height)) };
  },
  gridForScale({ image, scale }) {
    return {
      type: 'grid',
      grid: gridFromVec(grid_for_scale_vec(rgbaView(image), image.width, image.height, scale)),
    };
  },
  elasticBounds({ image, grid }) {
    const v = elastic_bounds_vec(
      rgbaView(image), image.width, image.height,
      grid.periodX, grid.periodY, grid.offsetX, grid.offsetY,
    );
    if (!v || v.length < 3) return { type: 'bounds', bounds: null };
    const nx = v[2];
    const x = Array.from(v.subarray(3, 3 + nx));
    const y = Array.from(v.subarray(3 + nx));
    return {
      type: 'bounds',
      bounds: { x, y, adapted: [v[0] === 1, v[1] === 1], cellsX: x.length - 1, cellsY: y.length - 1 },
    };
  },
  cropOffset({ image, scale }) {
    const xy = crop_offset(rgbaView(image), image.width, image.height, scale);
    return { type: 'offset', x: xy[0], y: xy[1] };
  },
  snapToGrid({ image, scale }) {
    const packed = snap_rgba_packed(rgbaView(image), image.width, image.height, scale);
    const img = unpackCoreImage(packed);
    return { type: 'image', width: img.width, height: img.height, buffer: img.buffer };
  },
  vectorPreProcess({ image, options }) {
    const o = options || {};
    const filter = o.filter === 'bilateral' ? 1 : o.filter === 'median' ? 2 : 0;
    const value = o.value ?? (filter === 1 ? 15 : 5);
    const packed = vector_preprocess_packed(
      rgbaView(image), image.width, image.height,
      filter, value, o.morphology ? 1 : 0, o.morphologyKernel ?? 3,
    );
    const img = unpackCoreImage(packed);
    return { type: 'image', width: img.width, height: img.height, buffer: img.buffer };
  },
  vectorFillBackground({ image, color }) {
    const c = color || { r: 255, g: 0, b: 255, a: 255 };
    const packed = vector_fill_bg_packed(rgbaView(image), image.width, image.height, c.r, c.g, c.b, c.a);
    const img = unpackCoreImage(packed);
    return { type: 'image', width: img.width, height: img.height, buffer: img.buffer };
  },
  vectorPostProcess({ image, options }) {
    const o = options || {};
    const filter = o.filter === 'gaussian' ? 3 : o.filter === 'median' ? 2 : 0;
    const packed = vector_postprocess_packed(rgbaView(image), image.width, image.height, filter, o.value ?? 3);
    const img = unpackCoreImage(packed);
    return { type: 'image', width: img.width, height: img.height, buffer: img.buffer };
  },
  encodePng({ image }) {
    const bytes = encode_png_bytes(rgbaView(image), image.width, image.height);
    return { type: 'bytes', buffer: bytes.buffer };
  },
  quantize({ image, maxColors, fixedPalette }) {
    const fixed = (fixedPalette && fixedPalette.length)
      ? Uint8Array.from(fixedPalette.flatMap((c) => [c.r, c.g, c.b, c.a ?? 255]))
      : new Uint8Array(0);
    const packed = quantize_packed(rgbaView(image), image.width, image.height, maxColors, fixed);
    const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
    const w = dv.getUint32(0, true);
    const h = dv.getUint32(4, true);
    const n = dv.getUint32(8, true);
    const palette = [];
    for (let i = 0; i < n; i++) {
      const o = 12 + i * 4;
      palette.push({ r: packed[o], g: packed[o + 1], b: packed[o + 2], a: packed[o + 3] });
    }
    const imgStart = 12 + n * 4;
    const pixels = new Uint8ClampedArray(packed.buffer, packed.byteOffset + imgStart, w * h * 4).slice();
    return { type: 'quantize', width: w, height: h, buffer: pixels.buffer, palette, colorsUsed: n };
  },
  contentAdaptive({ image, targetW, targetH }) {
    const packed = content_adaptive_packed(rgbaView(image), image.width, image.height, targetW, targetH);
    const img = unpackCoreImage(packed);
    return { type: 'image', width: img.width, height: img.height, buffer: img.buffer };
  },
  morphCleanup({ image }) {
    const packed = morph_cleanup_packed(rgbaView(image), image.width, image.height);
    const img = unpackCoreImage(packed);
    return { type: 'image', width: img.width, height: img.height, buffer: img.buffer };
  },
  detectColorCount({ image, downsampleTo, quantize, dominance, maxColors }) {
    const v = detect_color_count(
      rgbaView(image),
      image.width,
      image.height,
      downsampleTo ?? 64,
      quantize ?? 48,
      dominance ?? 0.015,
      maxColors ?? 32,
    );
    return { type: 'number', value: v };
  },
  downscale({ image, scale, method, domThreshold, alignGrid }) {
    const packed = downscale_rgba_packed(
      rgbaView(image),
      image.width,
      image.height,
      scale,
      method,
      domThreshold ?? 0.15,
      alignGrid ? 1 : 0,
    );
    const img = unpackCoreImage(packed);
    return { type: 'image', width: img.width, height: img.height, buffer: img.buffer };
  },
  downscaleBounds({ image, bounds, method, domThreshold }) {
    const packed = downscale_bounds_packed(
      rgbaView(image),
      image.width,
      image.height,
      Float64Array.from(bounds.x),
      Float64Array.from(bounds.y),
      method,
      domThreshold ?? 0.15,
    );
    const img = unpackCoreImage(packed);
    return { type: 'image', width: img.width, height: img.height, buffer: img.buffer };
  },
  downscaleGrid({ image, grid, method, domThreshold }) {
    const packed = downscale_grid_packed(
      rgbaView(image),
      image.width,
      image.height,
      grid.periodX,
      grid.periodY,
      grid.offsetX,
      grid.offsetY,
      method,
      domThreshold ?? 0.15,
    );
    const img = unpackCoreImage(packed);
    return { type: 'image', width: img.width, height: img.height, buffer: img.buffer };
  },
};

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'init') {
    try {
      const wasmUrl = new URL(msg.wasmUrl, import.meta.url).href;
      await init(wasmUrl);
      ready = true;
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'init-error', error: String(err?.message || err) });
    }
    return;
  }

  if (msg.type === 'rpc') {
    if (!ready) {
      self.postMessage({
        type: 'rpc-result',
        id: msg.id,
        ok: false,
        error: 'unfake-core worker not ready',
      });
      return;
    }
    const fn = OPS[msg.op];
    if (!fn) {
      self.postMessage({
        type: 'rpc-result',
        id: msg.id,
        ok: false,
        error: `Unknown core op: ${msg.op}`,
      });
      return;
    }
    try {
      const result = fn(msg.payload);
      const transfers = [];
      if (result?.type === 'image' || result?.type === 'quantize' || result?.type === 'bytes') {
        transfers.push(result.buffer);
      }
      self.postMessage({ type: 'rpc-result', id: msg.id, ok: true, result }, transfers);
    } catch (err) {
      self.postMessage({
        type: 'rpc-result',
        id: msg.id,
        ok: false,
        error: String(err?.message || err),
      });
    }
  }
};
