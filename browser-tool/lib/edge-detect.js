/**
 * Canonical tiled edge detector (synced from experiments/lib/edge-detect.mjs).
 * JS fallback when unfake-core WASM is unavailable.
 */
import { peakLag } from './cvfree.js';

function mode(arr) {
  if (!arr.length) return 0;
  const counts = {};
  let max = 0;
  let res = arr[0];
  for (const v of arr) {
    counts[v] = (counts[v] || 0) + 1;
    if (counts[v] > max) { max = counts[v]; res = v; }
  }
  return res;
}

const KX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
const KY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

function reflect(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * (n - 1) - i;
  }
  return i;
}

function toGray(img) {
  const { data, width: W, height: H } = img;
  const g = new Float64Array(W * H);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    g[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return { g, W, H };
}

function sobelProfiles(g, W, H, x0, y0, rw, rh) {
  const profX = new Float64Array(rw);
  const profY = new Float64Array(rh);
  for (let ly = 0; ly < rh; ly++) {
    for (let lx = 0; lx < rw; lx++) {
      const y = y0 + ly;
      const x = x0 + lx;
      let sx = 0;
      let sy = 0;
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++, k++) {
          const v = g[reflect(y + dy, H) * W + reflect(x + dx, W)];
          sx += v * KX[k];
          sy += v * KY[k];
        }
      }
      profX[lx] += Math.abs(sx);
      profY[ly] += Math.abs(sy);
    }
  }
  return { profX, profY };
}

function regionStddev(g, W, x0, y0, rw, rh) {
  let sum = 0;
  let sum2 = 0;
  const n = rw * rh;
  for (let ly = 0; ly < rh; ly++) {
    for (let lx = 0; lx < rw; lx++) {
      const v = g[(y0 + ly) * W + (x0 + lx)];
      sum += v;
      sum2 += v * v;
    }
  }
  const mean = sum / n;
  return Math.sqrt(Math.max(0, sum2 / n - mean * mean));
}

function scalesFromRoi(g, W, H, x0, y0, rw, rh) {
  if (rw < 3 || rh < 3) return { h: 1, v: 1 };
  const { profX, profY } = sobelProfiles(g, W, H, x0, y0, rw, rh);
  return { h: peakLag(profX), v: peakLag(profY) };
}

export function singleRegionEdgeDetect(img) {
  const { g, W, H } = toGray(img);
  const x0 = Math.floor(W * 0.125);
  const y0 = Math.floor(H * 0.125);
  const rw = Math.floor(W * 0.75);
  const rh = Math.floor(H * 0.75);
  if (rw < 3 || rh < 3) return 1;
  const { h: hScale, v: vScale } = scalesFromRoi(g, W, H, x0, y0, rw, rh);
  if (hScale > 1 && vScale > 1 && Math.abs(hScale - vScale) <= 2) {
    return Math.round((hScale + vScale) / 2);
  }
  return Math.max(hScale, vScale, 1);
}

export function tiledEdgeDetect(img) {
  const { g, W, H } = toGray(img);
  const TILE_COUNT = 3;
  const OVERLAP = 0.25;
  const tileW = Math.floor(W / TILE_COUNT);
  const tileH = Math.floor(H / TILE_COUNT);
  const overlapW = Math.floor(tileW * OVERLAP);
  const overlapH = Math.floor(tileH * OVERLAP);

  if (tileW < 50 || tileH < 50) return singleRegionEdgeDetect(img);

  const allScales = [];
  for (let ty = 0; ty < TILE_COUNT; ty++) {
    for (let tx = 0; tx < TILE_COUNT; tx++) {
      const x0 = Math.max(0, tx * tileW - overlapW);
      const y0 = Math.max(0, ty * tileH - overlapH);
      const rw = Math.min(W - x0, tileW + 2 * overlapW);
      const rh = Math.min(H - y0, tileH + 2 * overlapH);
      if (rw < 30 || rh < 30) continue;
      if (regionStddev(g, W, x0, y0, rw, rh) < 5.0) continue;
      const { h: hScale, v: vScale } = scalesFromRoi(g, W, H, x0, y0, rw, rh);
      if (hScale > 1) allScales.push(hScale);
      if (vScale > 1) allScales.push(vScale);
    }
  }

  if (allScales.length === 0) return singleRegionEdgeDetect(img);
  return mode(allScales) || 1;
}
