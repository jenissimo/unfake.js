/* global self, SVD */
/**
 * OpenCV operations for Web Worker (classic script, no DOM).
 * Registers handlers on self.openCvOps.
 */
(function () {
  'use strict';

  // --- Math helpers ----------------------------------------------------------

  function median(arr) {
    if (!arr.length) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return arr.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

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

  function multiply2x2(a, b) {
    const [a00, a01] = a[0];
    const [a10, a11] = a[1];
    const [b00, b01] = b[0];
    const [b10, b11] = b[1];
    return [
      [a00 * b00 + a01 * b10, a00 * b01 + a01 * b11],
      [a10 * b00 + a11 * b10, a10 * b01 + a11 * b11],
    ];
  }

  function detectScale(signal) {
    if (signal.length < 3) return 1;
    const meanVal = signal.reduce((a, b) => a + b, 0) / signal.length;
    const std = Math.sqrt(signal.reduce((a, b) => a + (b - meanVal) ** 2, 0) / signal.length);
    const threshold = meanVal + 1.5 * std;
    const peaks = [];
    for (let i = 1; i < signal.length - 1; i++) {
      if (signal[i] > threshold && signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
        if (peaks.length === 0 || i - peaks[peaks.length - 1] > 2) peaks.push(i);
      }
    }
    if (peaks.length <= 2) return 1;
    const spacings = peaks.slice(1).map((p, i) => p - peaks[i]);
    const medianSpacing = median(spacings);
    const closeSpacings = spacings.filter((s) => Math.abs(s - medianSpacing) <= 2);
    if (closeSpacings.length / spacings.length > 0.7) return Math.round(medianSpacing);
    const modeSpacing = mode(spacings);
    return modeSpacing > 1 ? modeSpacing : 1;
  }

  // --- ImageData / Mat helpers -----------------------------------------------

  function imageDataFrom(payload) {
    const expected = payload.width * payload.height * 4;
    const bytes = new Uint8ClampedArray(payload.buffer, 0, expected);
    return new ImageData(bytes, payload.width, payload.height);
  }

  /** Copy Mat RGBA bytes row-by-row (handles non-contiguous ROI / stride). */
  function matToRgbaClamped(mat) {
    const width = mat.cols;
    const height = mat.rows;
    const channels = mat.channels();
    const rowBytes = width * channels;
    const out = new Uint8ClampedArray(width * height * channels);
    const src = mat.data;
    const step = mat.step[0];

    if (typeof mat.isContinuous === 'function' && mat.isContinuous() && step === rowBytes) {
      out.set(src.subarray(0, rowBytes * height));
      return out;
    }

    for (let y = 0; y < height; y++) {
      const srcOff = y * step;
      const dstOff = y * rowBytes;
      out.set(src.subarray(srcOff, srcOff + rowBytes), dstOff);
    }
    return out;
  }

  function matToImagePayload(mat) {
    const out = matToRgbaClamped(mat);
    return { type: 'image', width: mat.cols, height: mat.rows, buffer: out.buffer };
  }

  function withMats(cv, fn) {
    const mats = new Set();
    const track = (m) => {
      if (m && typeof m.delete === 'function') mats.add(m);
      return m;
    };
    try {
      return fn(track);
    } finally {
      mats.forEach((m) => { if (m && !m.isDeleted?.()) m.delete(); });
    }
  }

  // --- Edge detection helpers ------------------------------------------------

  function getProfile(mat, direction, cv, track) {
    const diff = track(new cv.Mat());
    if (direction === 'horizontal') cv.Sobel(mat, diff, cv.CV_32F, 1, 0);
    else cv.Sobel(mat, diff, cv.CV_32F, 0, 1);
    const axis = direction === 'horizontal' ? 0 : 1;
    const sums = new Float32Array(axis === 0 ? mat.cols : mat.rows).fill(0);
    const data = diff.data32F;
    for (let y = 0; y < mat.rows; y++) {
      for (let x = 0; x < mat.cols; x++) {
        const val = Math.abs(data[y * mat.cols + x]);
        if (axis === 0) sums[x] += val;
        else sums[y] += val;
      }
    }
    return sums;
  }

  function singleRegionEdgeDetect(grayMat, cv, track) {
    const w = grayMat.cols;
    const h = grayMat.rows;
    const roiRect = new cv.Rect(Math.floor(w * 0.125), Math.floor(h * 0.125), Math.floor(w * 0.75), Math.floor(h * 0.75));
    if (roiRect.width < 3 || roiRect.height < 3) return 1;
    const roiMat = track(grayMat.roi(roiRect));
    const hScale = detectScale(Array.from(getProfile(roiMat, 'horizontal', cv, track)));
    const vScale = detectScale(Array.from(getProfile(roiMat, 'vertical', cv, track)));
    if (hScale > 1 && vScale > 1 && Math.abs(hScale - vScale) <= 2) {
      return Math.round((hScale + vScale) / 2);
    }
    return Math.max(hScale, vScale, 1);
  }

  function findOptimalCrop(grayMat, scale, cv) {
    const sobelX = new cv.Mat();
    const sobelY = new cv.Mat();
    try {
      cv.Sobel(grayMat, sobelX, cv.CV_32F, 1, 0, 3);
      cv.Sobel(grayMat, sobelY, cv.CV_32F, 0, 1, 3);
      const profileX = new Float32Array(grayMat.cols).fill(0);
      const profileY = new Float32Array(grayMat.rows).fill(0);
      const dataX = sobelX.data32F;
      const dataY = sobelY.data32F;
      for (let y = 0; y < grayMat.rows; y++) {
        for (let x = 0; x < grayMat.cols; x++) {
          const idx = y * grayMat.cols + x;
          profileX[x] += Math.abs(dataX[idx]);
          profileY[y] += Math.abs(dataY[idx]);
        }
      }
      const findBestOffset = (profile, s) => {
        let bestOffset = 0;
        let maxScore = -1;
        for (let offset = 0; offset < s; offset++) {
          let currentScore = 0;
          for (let i = offset; i < profile.length; i += s) currentScore += profile[i] || 0;
          if (currentScore > maxScore) { maxScore = currentScore; bestOffset = offset; }
        }
        return bestOffset;
      };
      return { x: findBestOffset(profileX, scale), y: findBestOffset(profileY, scale) };
    } finally {
      sobelX.delete();
      sobelY.delete();
    }
  }

  // --- Content-adaptive core ---------------------------------------------------

  function contentAdaptiveCore(srcLab, targetW, targetH, cv, track) {
    const NUM_ITERATIONS = 5;
    const { cols: wi, rows: hi } = srcLab;
    const wo = targetW;
    const ho = targetH;
    const rx = wi / wo;
    const ry = hi / ho;
    const r_avg = (rx + ry) / 2.0;
    const min_singular_value = 0.5;
    const max_singular_value = Math.max(1.0, 0.5 * r_avg);

    const labPlanes = track(new cv.MatVector());
    cv.split(srcLab, labPlanes);
    const L_plane = track(labPlanes.get(0));
    const a_plane = track(labPlanes.get(1));
    const b_plane = track(labPlanes.get(2));

    const mu_k = [];
    const Sigma_k = [];
    const nu_k = [];

    for (let yk = 0; yk < ho; yk++) {
      for (let xk = 0; xk < wo; xk++) {
        const k_idx = yk * wo + xk;
        mu_k[k_idx] = [(xk + 0.5) * rx, (yk + 0.5) * ry];
        const initial_sx = (rx / 3) * (rx / 3);
        const initial_sy = (ry / 3) * (ry / 3);
        Sigma_k[k_idx] = [initial_sx, 0, 0, initial_sy];
        nu_k[k_idx] = [50.0, 0.0, 0.0];
      }
    }

    for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
      const gamma_sum_per_pixel = new Float32Array(wi * hi).fill(1e-9);
      const w_ki = new Array(wo * ho).fill(0).map(() => new Map());

      for (let k = 0; k < wo * ho; k++) {
        const [s0, s1, s2, s3] = Sigma_k[k];
        const det = s0 * s3 - s1 * s2;
        const inv_det = 1.0 / (det + 1e-9);
        const sigma_inv = [s3 * inv_det, -s1 * inv_det, -s2 * inv_det, s0 * inv_det];
        const [mu_x, mu_y] = mu_k[k];
        const i_min_x = Math.max(0, Math.floor(mu_x - 2 * rx));
        const i_max_x = Math.min(wi, Math.ceil(mu_x + 2 * rx));
        const i_min_y = Math.max(0, Math.floor(mu_y - 2 * ry));
        const i_max_y = Math.min(hi, Math.ceil(mu_y + 2 * ry));
        let w_sum = 1e-9;
        for (let yi = i_min_y; yi < i_max_y; yi++) {
          for (let xi = i_min_x; xi < i_max_x; xi++) {
            const dx = xi - mu_x;
            const dy = yi - mu_y;
            const exponent = dx * dx * sigma_inv[0] + 2 * dx * dy * sigma_inv[1] + dy * dy * sigma_inv[3];
            const weight = Math.exp(-0.5 * exponent);
            if (weight > 1e-5) {
              const i = yi * wi + xi;
              w_ki[k].set(i, weight);
              w_sum += weight;
            }
          }
        }
        for (const [i, weight] of w_ki[k].entries()) {
          const normalized_w = weight / w_sum;
          w_ki[k].set(i, normalized_w);
          gamma_sum_per_pixel[i] += normalized_w;
        }
      }

      const next_mu_k = new Array(wo * ho);
      const next_Sigma_k = new Array(wo * ho);
      const next_nu_k = new Array(wo * ho);

      for (let k = 0; k < wo * ho; k++) {
        let w_sum = 1e-9;
        const new_mu = [0, 0];
        const new_nu = [0, 0, 0];
        for (const [i, wk] of w_ki[k].entries()) {
          const gamma_k_i = wk / gamma_sum_per_pixel[i];
          w_sum += gamma_k_i;
          const yi = Math.floor(i / wi);
          const xi = i % wi;
          new_mu[0] += gamma_k_i * xi;
          new_mu[1] += gamma_k_i * yi;
          new_nu[0] += gamma_k_i * L_plane.data32F[i];
          new_nu[1] += gamma_k_i * a_plane.data32F[i];
          new_nu[2] += gamma_k_i * b_plane.data32F[i];
        }
        new_mu[0] /= w_sum;
        new_mu[1] /= w_sum;
        new_nu[0] /= w_sum;
        new_nu[1] /= w_sum;
        new_nu[2] /= w_sum;
        next_mu_k[k] = new_mu;
        next_nu_k[k] = new_nu;

        const new_Sigma_arr = [0, 0, 0, 0];
        for (const [i, wk] of w_ki[k].entries()) {
          const gamma_k_i = wk / gamma_sum_per_pixel[i];
          const yi = Math.floor(i / wi);
          const xi = i % wi;
          const dx = xi - new_mu[0];
          const dy = yi - new_mu[1];
          new_Sigma_arr[0] += gamma_k_i * dx * dx;
          new_Sigma_arr[1] += gamma_k_i * dx * dy;
          new_Sigma_arr[3] += gamma_k_i * dy * dy;
        }
        new_Sigma_arr[0] /= w_sum;
        new_Sigma_arr[1] /= w_sum;
        new_Sigma_arr[2] = new_Sigma_arr[1];
        new_Sigma_arr[3] /= w_sum;
        next_Sigma_k[k] = new_Sigma_arr;
      }

      for (let k = 0; k < wo * ho; k++) {
        const sigma_arr = next_Sigma_k[k];
        const sigma_mat2d = [[sigma_arr[0], sigma_arr[1]], [sigma_arr[2], sigma_arr[3]]];
        const { u, q, v } = SVD(sigma_mat2d);
        q[0] = Math.max(min_singular_value, Math.min(q[0], max_singular_value));
        q[1] = Math.max(min_singular_value, Math.min(q[1], max_singular_value));
        const s_diag = [[q[0], 0], [0, q[1]]];
        const v_t = [[v[0][0], v[1][0]], [v[0][1], v[1][1]]];
        const temp = multiply2x2(u, s_diag);
        const new_sigma_mat2d = multiply2x2(temp, v_t);
        mu_k[k] = next_mu_k[k];
        Sigma_k[k] = [new_sigma_mat2d[0][0], new_sigma_mat2d[0][1], new_sigma_mat2d[1][0], new_sigma_mat2d[1][1]];
        nu_k[k] = next_nu_k[k];
      }
    }

    const outLab = track(new cv.Mat(ho, wo, cv.CV_32FC3));
    for (let yk = 0; yk < ho; yk++) {
      for (let xk = 0; xk < wo; xk++) {
        const k_idx = yk * wo + xk;
        const [l, a, b] = nu_k[k_idx];
        outLab.data32F[(yk * wo + xk) * 3] = l;
        outLab.data32F[(yk * wo + xk) * 3 + 1] = a;
        outLab.data32F[(yk * wo + xk) * 3 + 2] = b;
      }
    }
    return outLab;
  }

  // --- RPC handlers ----------------------------------------------------------

  self.openCvOps = {
    morphologicalCleanup(payload, cv) {
      const imgData = imageDataFrom(payload.image);
      return withMats(cv, (track) => {
        const mat = track(cv.matFromImageData(imgData));
        const kernel = track(cv.Mat.ones(2, 2, cv.CV_8U));
        cv.morphologyEx(mat, mat, cv.MORPH_OPEN, kernel);
        cv.morphologyEx(mat, mat, cv.MORPH_CLOSE, kernel);
        return matToImagePayload(mat);
      });
    },

    detectOptimalColorCount(payload, cv) {
      const imgData = imageDataFrom(payload.image);
      const {
        downsampleTo = 64,
        colorQuantizeFactor = 48,
        dominanceThreshold = 0.015,
        maxColors = 32,
      } = payload;
      return withMats(cv, (track) => {
        const src = track(cv.matFromImageData(imgData));
        const aspectRatio = src.rows / src.cols;
        const targetWidth = downsampleTo;
        const targetHeight = Math.round(targetWidth * aspectRatio);
        const smallMat = track(new cv.Mat());
        cv.resize(src, smallMat, new cv.Size(targetWidth, targetHeight), 0, 0, cv.INTER_AREA);
        cv.medianBlur(smallMat, smallMat, 5);
        cv.GaussianBlur(smallMat, smallMat, new cv.Size(5, 5), 1, 1);

        const colorCounts = new Map();
        const totalPixels = smallMat.rows * smallMat.cols;
        const d = smallMat.data;
        for (let i = 0; i < d.length; i += 4) {
          if (d[i + 3] < 200) continue;
          const r = Math.round(d[i] / colorQuantizeFactor) * colorQuantizeFactor;
          const g = Math.round(d[i + 1] / colorQuantizeFactor) * colorQuantizeFactor;
          const b = Math.round(d[i + 2] / colorQuantizeFactor) * colorQuantizeFactor;
          const colorKey = `${r},${g},${b}`;
          colorCounts.set(colorKey, (colorCounts.get(colorKey) || 0) + 1);
        }
        const minPixelsForDominance = Math.max(3, Math.round(totalPixels * dominanceThreshold));
        const sortedColors = Array.from(colorCounts.entries())
          .filter(([, count]) => count >= minPixelsForDominance);
        let significantColors = sortedColors.length;
        if (significantColors > maxColors) {
          const strictThreshold = Math.max(minPixelsForDominance, Math.round(totalPixels * 0.02));
          significantColors = sortedColors.filter(([, count]) => count >= strictThreshold).length;
        }
        const result = Math.max(2, Math.min(significantColors, maxColors));
        return { type: 'number', value: result };
      });
    },

    legacyEdgeDetect(payload, cv) {
      const imgData = imageDataFrom(payload.image);
      return withMats(cv, (track) => {
        const srcMat = track(cv.matFromImageData(imgData));
        const gray = track(new cv.Mat());
        cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);
        const scale = singleRegionEdgeDetect(gray, cv, track);
        return { type: 'number', value: scale };
      });
    },

    tiledEdgeDetect(payload, cv) {
      const imgData = imageDataFrom(payload.image);
      return withMats(cv, (track) => {
        const srcMat = track(cv.matFromImageData(imgData));
        const gray = track(new cv.Mat());
        cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);

        const allScales = [];
        const TILE_COUNT = 3;
        const OVERLAP = 0.25;
        const tileW = Math.floor(gray.cols / TILE_COUNT);
        const tileH = Math.floor(gray.rows / TILE_COUNT);
        const overlapW = Math.floor(tileW * OVERLAP);
        const overlapH = Math.floor(tileH * OVERLAP);

        if (tileW < 50 || tileH < 50) {
          const scale = singleRegionEdgeDetect(gray, cv, track);
          return { type: 'number', value: scale };
        }

        for (let y = 0; y < TILE_COUNT; y++) {
          for (let x = 0; x < TILE_COUNT; x++) {
            const roiX = Math.max(0, x * tileW - overlapW);
            const roiY = Math.max(0, y * tileH - overlapH);
            const roiW = Math.min(gray.cols - roiX, tileW + 2 * overlapW);
            const roiH = Math.min(gray.rows - roiY, tileH + 2 * overlapH);
            if (roiW < 30 || roiH < 30) continue;
            const tileMat = track(gray.roi(new cv.Rect(roiX, roiY, roiW, roiH)));
            const mean = track(new cv.Mat());
            const stddev = track(new cv.Mat());
            cv.meanStdDev(tileMat, mean, stddev);
            if (stddev.data64F[0] < 5.0) continue;
            const hScale = detectScale(Array.from(getProfile(tileMat, 'horizontal', cv, track)));
            const vScale = detectScale(Array.from(getProfile(tileMat, 'vertical', cv, track)));
            if (hScale > 1) allScales.push(hScale);
            if (vScale > 1) allScales.push(vScale);
          }
        }

        if (allScales.length === 0) {
          const scale = singleRegionEdgeDetect(gray, cv, track);
          return { type: 'number', value: scale };
        }
        const bestScale = mode(allScales) || 1;
        return { type: 'number', value: bestScale };
      });
    },

    snapToGrid(payload, cv) {
      const imgData = imageDataFrom(payload.image);
      const scale = payload.scale;
      return withMats(cv, (track) => {
        const srcMat = track(cv.matFromImageData(imgData));
        const grayMat = track(new cv.Mat());
        cv.cvtColor(srcMat, grayMat, cv.COLOR_RGBA2GRAY);
        const crop = findOptimalCrop(grayMat, scale, cv);
        const newWidth = Math.floor((imgData.width - crop.x) / scale) * scale;
        const newHeight = Math.floor((imgData.height - crop.y) / scale) * scale;
        if (newWidth < scale || newHeight < scale) {
          return matToImagePayload(srcMat);
        }
        const rect = new cv.Rect(crop.x, crop.y, newWidth, newHeight);
        const croppedMat = track(srcMat.roi(rect).clone());
        return matToImagePayload(croppedMat);
      });
    },

    contentAdaptiveDownscale(payload, cv) {
      const imgData = imageDataFrom(payload.image);
      const { targetW, targetH } = payload;
      if (typeof SVD !== 'function') {
        throw new Error('SVD.js is required for content-adaptive downscaling.');
      }
      return withMats(cv, (track) => {
        const srcMat = track(cv.matFromImageData(imgData));
        const channels = track(new cv.MatVector());
        cv.split(srcMat, channels);
        const alpha = track(channels.get(3));
        const outAlpha = track(new cv.Mat());
        cv.resize(alpha, outAlpha, new cv.Size(targetW, targetH), 0, 0, cv.INTER_AREA);

        const srcRGB = track(new cv.Mat());
        const rgbVec = track(new cv.MatVector());
        rgbVec.push_back(channels.get(0));
        rgbVec.push_back(channels.get(1));
        rgbVec.push_back(channels.get(2));
        cv.merge(rgbVec, srcRGB);

        const srcRGB32f = track(new cv.Mat());
        srcRGB.convertTo(srcRGB32f, cv.CV_32F, 1.0 / 255.0);
        const srcLab = track(new cv.Mat());
        cv.cvtColor(srcRGB32f, srcLab, cv.COLOR_RGB2Lab);

        const outLab = contentAdaptiveCore(srcLab, targetW, targetH, cv, track);

        const outRGB_32f = track(new cv.Mat());
        cv.cvtColor(outLab, outRGB_32f, cv.COLOR_Lab2RGB);
        const outRGB_8u = track(new cv.Mat());
        outRGB_32f.convertTo(outRGB_8u, cv.CV_8U, 255.0, 0);

        const outRgbSplit = track(new cv.MatVector());
        cv.split(outRGB_8u, outRgbSplit);
        const finalRgbaMat = track(new cv.Mat());
        const rgbaVec = track(new cv.MatVector());
        rgbaVec.push_back(outRgbSplit.get(0));
        rgbaVec.push_back(outRgbSplit.get(1));
        rgbaVec.push_back(outRgbSplit.get(2));
        rgbaVec.push_back(outAlpha);
        cv.merge(rgbaVec, finalRgbaMat);

        return matToImagePayload(finalRgbaMat);
      });
    },

    vectorPreProcess(payload, cv) {
      const imgData = imageDataFrom(payload.image);
      const options = payload.options || {};
      return withMats(cv, (track) => {
        const src = track(cv.matFromImageData(imgData));
        const dst = track(new cv.Mat());

        if (options.filter === 'bilateral') {
          const d = options.value || 15;
          const sigmaColor = d * 2;
          const sigmaSpace = d / 2;
          const channels = track(new cv.MatVector());
          cv.split(src, channels);
          const rgb = track(new cv.Mat());
          const rgbChannels = track(new cv.MatVector());
          rgbChannels.push_back(channels.get(0));
          rgbChannels.push_back(channels.get(1));
          rgbChannels.push_back(channels.get(2));
          cv.merge(rgbChannels, rgb);
          const filteredRgb = track(new cv.Mat());
          cv.bilateralFilter(rgb, filteredRgb, d, sigmaColor, sigmaSpace, cv.BORDER_DEFAULT);
          const filteredChannels = track(new cv.MatVector());
          cv.split(filteredRgb, filteredChannels);
          const finalChannels = track(new cv.MatVector());
          finalChannels.push_back(filteredChannels.get(0));
          finalChannels.push_back(filteredChannels.get(1));
          finalChannels.push_back(filteredChannels.get(2));
          finalChannels.push_back(channels.get(3));
          cv.merge(finalChannels, dst);
        } else if (options.filter === 'median') {
          let ksize = options.value || 5;
          if (ksize % 2 === 0) ksize++;
          cv.medianBlur(src, dst, ksize);
        } else {
          return matToImagePayload(src);
        }

        if (options.morphology) {
          const kernelSize = options.morphologyKernel || 3;
          const kernel = track(cv.Mat.ones(kernelSize, kernelSize, cv.CV_8U));
          cv.morphologyEx(dst, dst, cv.MORPH_CLOSE, kernel);
        }

        return matToImagePayload(dst);
      });
    },

    vectorFillBackground(payload, cv) {
      const imgData = imageDataFrom(payload.image);
      const color = payload.color || { r: 255, g: 0, b: 255, a: 255 };
      return withMats(cv, (track) => {
        const src = track(cv.matFromImageData(imgData));
        const { width, height } = imgData;
        const bgMat = track(new cv.Mat(height, width, cv.CV_8UC4, [color.r, color.g, color.b, color.a]));
        const channels = track(new cv.MatVector());
        cv.split(src, channels);
        const alphaMask = track(channels.get(3));
        src.copyTo(bgMat, alphaMask);
        return matToImagePayload(bgMat);
      });
    },

    vectorPostProcess(payload, cv) {
      const imgData = imageDataFrom(payload.image);
      const options = payload.options || {};
      return withMats(cv, (track) => {
        const src = track(cv.matFromImageData(imgData));
        const dst = track(new cv.Mat());
        let ksize = options.value || 3;
        if (ksize % 2 === 0) ksize++;
        if (options.filter === 'gaussian') {
          cv.GaussianBlur(src, dst, new cv.Size(ksize, ksize), 0, 0, cv.BORDER_DEFAULT);
        } else if (options.filter === 'median') {
          cv.medianBlur(src, dst, ksize);
        } else {
          return matToImagePayload(src);
        }
        return matToImagePayload(dst);
      });
    },
  };
})();
