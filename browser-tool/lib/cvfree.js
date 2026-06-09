/**
 * Pure-JS replacements for OpenCV steps in the pixel-art pipeline.
 * Ported from experiments/lib/cvfree.mjs + detectors.mjs (experiment/headless).
 */

function maxPeriod(n) {
  return Math.max(2, Math.min(Math.floor(n / 8), 128));
}

function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function reconcileClose(a, b) {
  if (Math.abs(a - b) <= Math.max(1, 0.12 * Math.min(a, b))) return Math.min(a, b);
  if (a === b) return a;
  const g = gcd(a, b);
  return g >= 2 ? g : Math.min(a, b);
}

function gcd(a, b) {
  while (b) { [a, b] = [b, a % b]; }
  return a;
}

export function boundarySignals(img) {
  const { data, width, height } = img;
  const col = new Float64Array(width);
  const row = new Float64Array(height);
  for (let y = 0; y < height; y++) {
    for (let x = 1; x < width; x++) {
      const i = (y * width + x) * 4;
      const j = i - 4;
      col[x] += Math.abs(data[i] - data[j]) + Math.abs(data[i + 1] - data[j + 1])
        + Math.abs(data[i + 2] - data[j + 2]) + Math.abs(data[i + 3] - data[j + 3]);
    }
  }
  for (let x = 0; x < width; x++) {
    for (let y = 1; y < height; y++) {
      const i = (y * width + x) * 4;
      const j = ((y - 1) * width + x) * 4;
      row[y] += Math.abs(data[i] - data[j]) + Math.abs(data[i + 1] - data[j + 1])
        + Math.abs(data[i + 2] - data[j + 2]) + Math.abs(data[i + 3] - data[j + 3]);
    }
  }
  return { col, row };
}

export function peakLag(sig) {
  const n = sig.length;
  const maxLag = maxPeriod(n);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += sig[i];
  mean /= n;
  const s = new Float64Array(n);
  let norm = 0;
  for (let i = 0; i < n; i++) { s[i] = sig[i] - mean; norm += s[i] * s[i]; }
  norm = norm || 1e-9;
  const ac = new Float64Array(maxLag + 2);
  for (let lag = 2; lag <= maxLag; lag++) {
    let r = 0;
    for (let i = 0; i + lag < n; i++) r += s[i] * s[i + lag];
    ac[lag] = r / norm;
  }
  let start = 2;
  const cap = Math.max(3, Math.floor(maxLag / 2));
  while (start < cap && ac[start + 1] <= ac[start]) start++;

  let gmax = 0;
  let gp = start;
  for (let p = start; p <= maxLag; p++) if (ac[p] > gmax) { gmax = ac[p]; gp = p; }
  if (gmax <= 0) return 1;
  const thr = 0.6 * gmax;
  for (let p = start; p <= maxLag; p++) {
    if (ac[p] >= thr && ac[p] >= ac[p - 1] && ac[p] >= ac[p + 1]) return p;
  }
  return gp;
}

/** Cv-free edge scale detector (acpeak). */
export function acpeakDetect(img) {
  const { col, row } = boundarySignals(img);
  return reconcileClose(peakLag(col), peakLag(row));
}

function morph2x2(img, useMax) {
  const { data, width: W, height: H } = img;
  const out = new Uint8ClampedArray(data.length);
  const nb = [[-1, -1], [0, -1], [-1, 0], [0, 0]];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      for (let ch = 0; ch < 4; ch++) {
        let v = useMax ? 0 : 255;
        for (let k = 0; k < 4; k++) {
          const nx = x + nb[k][0];
          const ny = y + nb[k][1];
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          const s = data[(ny * W + nx) * 4 + ch];
          v = useMax ? (s > v ? s : v) : (s < v ? s : v);
        }
        out[o + ch] = v;
      }
    }
  }
  return new ImageData(out, W, H);
}

const erode = (img) => morph2x2(img, false);
const dilate = (img) => morph2x2(img, true);

/** OPEN then CLOSE with 2×2 kernel, anchor (1,1) — matches OpenCV preset. */
export function morphologicalCleanupCvFree(img) {
  return erode(dilate(dilate(erode(img))));
}

export function findOptimalCropOffset(img, scale) {
  const { data, width: W, height: H } = img;
  const g = new Float64Array(W * H);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    g[p] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
  }
  const refl = (i, n) => {
    if (n === 1) return 0;
    while (i < 0 || i >= n) {
      if (i < 0) i = -i;
      if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
  };
  const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  const profX = new Float64Array(W);
  const profY = new Float64Array(H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let sx = 0;
      let sy = 0;
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++, k++) {
          const v = g[refl(y + dy, H) * W + refl(x + dx, W)];
          sx += v * kx[k];
          sy += v * ky[k];
        }
      }
      profX[x] += Math.abs(sx);
      profY[y] += Math.abs(sy);
    }
  }
  const best = (prof, s) => {
    let bo = 0;
    let ms = -1;
    for (let o = 0; o < s; o++) {
      let c = 0;
      for (let i = o; i < prof.length; i += s) c += prof[i] || 0;
      if (c > ms) { ms = c; bo = o; }
    }
    return bo;
  };
  return { x: best(profX, scale), y: best(profY, scale) };
}

export function cropImageData(img, x, y, w, h) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let ty = 0; ty < h; ty++) {
    for (let tx = 0; tx < w; tx++) {
      const si = ((y + ty) * img.width + (x + tx)) * 4;
      const di = (ty * w + tx) * 4;
      out[di] = img.data[si];
      out[di + 1] = img.data[si + 1];
      out[di + 2] = img.data[si + 2];
      out[di + 3] = img.data[si + 3];
    }
  }
  return new ImageData(out, w, h);
}

/** Snap image to pixel grid using cv-free crop search. */
export function snapToGrid(img, scale, logger) {
  const crop = findOptimalCropOffset(img, scale);
  const newWidth = Math.floor((img.width - crop.x) / scale) * scale;
  const newHeight = Math.floor((img.height - crop.y) / scale) * scale;
  if (newWidth < scale || newHeight < scale) {
    logger?.warn?.(`Snapping failed: resulting image size (${newWidth}x${newHeight}) is too small. Skipping snap.`);
    return img;
  }
  logger?.log?.(`Image cropped to: ${newWidth} x ${newHeight} (from offset x:${crop.x}, y:${crop.y})`);
  return cropImageData(img, crop.x, crop.y, newWidth, newHeight);
}
