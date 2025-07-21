/**
 * @fileoverview Shared utilities for pixel art and vector processing.
 * This file contains common functions used across the library for tasks like
 * image manipulation, color quantization, and mathematical calculations.
 */

import * as IQ from 'image-q';

// --- Logger Utility ----------------------------------------------------------

/**
 * Logger utility with debug control.
 * Controlled by `window.DEBUG_UNFAKE` flag.
 */
export const logger = {
  _isEnabled: () => typeof window !== 'undefined' ? window.DEBUG_UNFAKE !== false : true,
  log: (...args) => logger._isEnabled() && console.log('[unfake.js]', ...args),
  warn: (...args) => logger._isEnabled() && console.warn('[unfake.js]', ...args),
  error: (...args) => console.error('[unfake.js]', ...args),
};

// --- OpenCV Integration ------------------------------------------------------

let cvPromise = null;

/**
 * Ensures OpenCV is ready and returns the `cv` namespace.
 * This function is memoized and returns a singleton promise.
 * @returns {Promise<object>} A promise that resolves with the `cv` object.
 */
export async function cvReady() {
  if (cvPromise) return cvPromise;

  cvPromise = new Promise((resolve, reject) => {
    // Case 1: cv is already a promise (common in modern opencv.js)
    if (typeof cv !== 'undefined' && cv instanceof Promise) {
      logger.log('OpenCV is loading (promise)...');
      return cv.then(resolve).catch(reject);
    }

    // Case 2: cv is already loaded and initialized
    if (typeof cv !== 'undefined' && cv.getBuildInformation) {
      logger.log('OpenCV is already ready.');
      return resolve(cv);
    }

    // Case 3: In a non-browser environment
    if (typeof window === 'undefined') {
      return reject(new Error('OpenCV is not available in a non-browser environment.'));
    }

    // Case 4: window.onOpenCvReady is defined (legacy support)
    if (typeof window.onOpenCvReady === 'function') {
      const originalOnReady = window.onOpenCvReady;
      window.onOpenCvReady = () => {
        originalOnReady();
        if (cv.getBuildInformation) {
          logger.log('OpenCV initialized via onOpenCvReady callback.');
          resolve(cv);
        } else {
          reject(new Error('onOpenCvReady was called, but cv object is invalid.'));
        }
      };
      return; // The callback will resolve/reject the promise
    }

    // Case 5: Poll for cv object
    logger.log('Waiting for OpenCV to initialize...');
    const startTime = Date.now();
    const interval = setInterval(() => {
      if (Date.now() - startTime > 20000) { // 20-second timeout
        clearInterval(interval);
        return reject(new Error('OpenCV failed to initialize within 20 seconds.'));
      }

      if (typeof cv !== 'undefined') {
        clearInterval(interval);
        // It might have loaded and become a promise
        if (cv instanceof Promise) {
            logger.log('OpenCV is loading (promise detected)...');
            return cv.then(resolve).catch(reject);
        }
        if (cv.getBuildInformation) {
            logger.log('OpenCV initialized successfully.');
            resolve(cv);
        }
      }
    }, 100);
  });
  return cvPromise;
}

/**
 * Resource management wrapper for OpenCV operations.
 * Ensures that OpenCV is ready and all created Mat objects are deleted.
 * @param {function(cv: object, track: Function): Promise<any>} fn - An async function that receives `cv` and a `track` function.
 * @returns {Promise<any>} The result of the async function execution.
 */
export async function withCv(fn) {
    const mats = new Set();
    const cv = await cvReady();
    const track = (mat) => {
        if (mat && typeof mat.delete === 'function') {
            mats.add(mat);
        }
        return mat;
    };
    try {
        return await fn(cv, track);
    } finally {
        mats.forEach(mat => {
            // isDeleted might not exist on all cv versions, check for it.
            if (mat && !mat.isDeleted?.()) {
                mat.delete();
            }
        });
    }
}

// --- Image Loading and Conversion --------------------------------------------

/**
 * Reads a user-supplied File/Blob into an ImageData object.
 * @param {File|Blob} file The file to read.
 * @returns {Promise<ImageData>} A promise that resolves with the image data.
 */
export async function fileToImageData(file) {
  logger.log(`Starting conversion of file: ${file.name}, size: ${file.size}`);

  if (file.size > 50 * 1024 * 1024) { // 50MB limit
    throw new Error(`File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB. Max 50MB.`);
  }

  const bitmap = await Promise.race([
    createImageBitmap(file),
    new Promise((_, reject) => setTimeout(() => reject(new Error('Image loading timeout')), 30000))
  ]);
  logger.log(`ImageBitmap created: ${bitmap.width}x${bitmap.height}`);

  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(bitmap.width, bitmap.height)
    : document.createElement('canvas');

  if (!(canvas instanceof OffscreenCanvas)) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
  }
  
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * Encodes ImageData to a PNG buffer using UPNG.js, with a canvas fallback.
 * @param {ImageData} imgData The image data to encode.
 * @returns {Promise<Uint8Array>} A Uint8Array containing the PNG file data.
 */
export async function encodePng(imgData) {
  const { UPNG } = window;

  if (UPNG?.encode) {
    const buffer = UPNG.encode([imgData.data.buffer], imgData.width, imgData.height, 0);
    return new Uint8Array(buffer);
  }

  logger.warn('UPNG.js not found, using slower canvas fallback for PNG encoding.');
  const canvas = document.createElement('canvas');
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  canvas.getContext('2d').putImageData(imgData, 0, 0);

  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('Canvas fallback for PNG encoding failed.');
  return new Uint8Array(await blob.arrayBuffer());
}

// --- Image Processing and Cleanup --------------------------------------------

/**
 * Applies morphological cleanup to an image.
 * Uses OPEN to remove noise and CLOSE to fill small gaps.
 * @param {ImageData} imgData The input image data.
 * @returns {Promise<ImageData>} The cleaned image data.
 */
export async function morphologicalCleanup(imgData) {
  return withCv((cv, track) => {
    const mat = track(cv.matFromImageData(imgData));
    const kernel = track(cv.Mat.ones(2, 2, cv.CV_8U)); // 2x2 for finer details

    // OPEN operation removes noise and small artifacts
    cv.morphologyEx(mat, mat, cv.MORPH_OPEN, kernel);

    // CLOSE operation fills small gaps and smooths jagged edges
    cv.morphologyEx(mat, mat, cv.MORPH_CLOSE, kernel);

    return new ImageData(new Uint8ClampedArray(mat.data), mat.cols, mat.rows);
  });
}

/**
 * Converts the alpha channel to binary (0 or 255) based on a threshold.
 * @param {ImageData} imgData The input image data.
 * @param {number} [threshold=128] Alpha threshold (0-255).
 * @returns {ImageData} The processed image data.
 */
export function alphaBinarization(imgData, threshold = 128) {
  const { data, width, height } = imgData;
  const out = new Uint8ClampedArray(data);

  for (let i = 3; i < out.length; i += 4) {
    out[i] = out[i] >= threshold ? 255 : 0;
  }

  return new ImageData(out, width, height);
}

/**
 * Removes isolated diagonal pixels ("jaggies") from pixel art.
 * @param {ImageData} imgData The input image data.
 * @returns {ImageData} The processed image data.
 */
export function jaggyCleaner(imgData) {
  const { data, width, height } = imgData;
  const out = new Uint8ClampedArray(data);

  const get = (x, y) => {
    if (x < 0 || x >= width || y < 0 || y >= height) return null;
    const i = (y * width + x) * 4;
    return { r: out[i], g: out[i + 1], b: out[i + 2], a: out[i + 3] };
  };

  const set = (x, y, color) => {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    const i = (y * width + x) * 4;
    Object.assign(out, [color.r, color.g, color.b, color.a], i);
  };

  const isOpaque = (p) => p?.a > 128;

  // Apply rule-based cleaning for orphaned diagonal pixels
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      if (!isOpaque(get(x, y))) continue;

      const N  = isOpaque(get(x, y - 1)), S  = isOpaque(get(x, y + 1));
      const E  = isOpaque(get(x + 1, y)), W  = isOpaque(get(x - 1, y));
      const NE = isOpaque(get(x + 1, y - 1)), NW = isOpaque(get(x - 1, y - 1));
      const SE = isOpaque(get(x + 1, y + 1)), SW = isOpaque(get(x - 1, y + 1));
      
      const opaqueOrth = N + S + E + W;
      const opaqueDiag = NE + NW + SE + SW;
      
      // Remove pixel if it has no orthogonal neighbors and only one diagonal one
      if (opaqueOrth === 0 && opaqueDiag === 1) {
        set(x, y, { r: 0, g: 0, b: 0, a: 0 });
      }
    }
  }

  return new ImageData(out, width, height);
}

/**
 * Ensures final pixels have binary alpha and transparent pixels are black.
 * @param {ImageData} imgData The image to clean.
 * @returns {ImageData} The cleaned image.
 */
export function finalizePixels(imgData) {
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 128) {
      data[i] = data[i + 1] = data[i + 2] = data[i + 3] = 0;
    } else {
      data[i + 3] = 255;
    }
  }
  return imgData;
}

// --- Color and Palette -------------------------------------------------------

/**
 * Automatically detects the optimal number of colors in an image.
 * Uses aggressive clustering and dominant color analysis.
 * @param {ImageData} imgData The input image data.
 * @param {object} [options] Options for detection.
 * @returns {Promise<number>} The estimated optimal number of colors.
 */
export async function detectOptimalColorCount(imgData, {
    downsampleTo = 64,
    colorQuantizeFactor = 48,
    dominanceThreshold = 0.015,
    maxColors = 32
} = {}) {
    logger.log('Detecting optimal color count...');
    return withCv(async (cv, track) => {
        const src = track(cv.matFromImageData(imgData));
        
        // Downsample for faster analysis
        const aspectRatio = src.rows / src.cols;
        const targetWidth = downsampleTo;
        const targetHeight = Math.round(targetWidth * aspectRatio);
        const dsize = new cv.Size(targetWidth, targetHeight);
        const smallMat = track(new cv.Mat());
        cv.resize(src, smallMat, dsize, 0, 0, cv.INTER_AREA);
        
        // Blur to remove noise and gradients
        cv.medianBlur(smallMat, smallMat, 5);
        cv.GaussianBlur(smallMat, smallMat, new cv.Size(5, 5), 1, 1);

        // Gather color statistics with aggressive quantization
        const colorCounts = new Map();
        const totalPixels = smallMat.rows * smallMat.cols;
        const d = smallMat.data;

        for (let i = 0; i < d.length; i += 4) {
            if (d[i + 3] < 200) continue; // Ignore transparent pixels

            const r = Math.round(d[i] / colorQuantizeFactor) * colorQuantizeFactor;
            const g = Math.round(d[i + 1] / colorQuantizeFactor) * colorQuantizeFactor;
            const b = Math.round(d[i + 2] / colorQuantizeFactor) * colorQuantizeFactor;

            const colorKey = `${r},${g},${b}`;
            colorCounts.set(colorKey, (colorCounts.get(colorKey) || 0) + 1);
        }
        
        // Analyze for dominant colors
        const minPixelsForDominance = Math.max(3, Math.round(totalPixels * dominanceThreshold));
        logger.log(`Found ${colorCounts.size} unique quantized colors (pre-filter)`);
        
        const sortedColors = Array.from(colorCounts.entries())
            .filter(([, count]) => count >= minPixelsForDominance);
        
        let significantColors = sortedColors.length;
        
        // If there are too many colors, apply a stricter threshold
        if (significantColors > maxColors) {
            const strictThreshold = Math.max(minPixelsForDominance, Math.round(totalPixels * 0.02));
            significantColors = sortedColors.filter(([, count]) => count >= strictThreshold).length;
            logger.log(`Filtered down to ${significantColors} dominant colors`);
        }
        
        const result = Math.max(2, Math.min(significantColors, maxColors));
        logger.log(`Final auto-detected color count: ${result}`);
        return result;
    });
}


/**
 * Quantizes image colors using image-q library with a fallback.
 * @param {ImageData} imgData The input image data.
 * @param {number} maxColors The maximum number of colors.
 * @param {string[]|null} [fixedPalette=null] Optional fixed palette of hex colors.
 * @returns {{quantized: ImageData, colorsUsed: number}}
 */
export function quantizeImage(imgData, maxColors, fixedPalette = null) {
    try {
        const inPointContainer = IQ.utils.PointContainer.fromImageData(imgData);
        const palette = fixedPalette?.length
            ? IQ.utils.Palette.createByColors(fixedPalette)
            : IQ.buildPaletteSync([inPointContainer], {
                colors: maxColors,
                colorDistanceFormula: 'euclidean',
                paletteQuantization: 'wuquant',
              });

        const outPointContainer = IQ.applyPaletteSync(inPointContainer, palette, {
            imageQuantization: 'nearest',
            ditherKern: 'None', // No dithering for pixel art
        });
        const quantized = new ImageData(new Uint8ClampedArray(outPointContainer.toUint8Array()), imgData.width, imgData.height);
        
        // Extract palette in {r, g, b, a} format for tracers
        const tracerPalette = palette.getPointContainer().getPointArray().map(p => ({
            r: Math.round(p.r), g: Math.round(p.g), b: Math.round(p.b), a: Math.round(p.a)
        }));
        
        return {
            quantized: finalizePixels(quantized),
            colorsUsed: tracerPalette.length,
            palette: tracerPalette,
        };
    } catch (error) {
        logger.warn('image-q quantization failed, using fallback:', error);
        const { data, width, height } = imgData;
        const result = new Uint8ClampedArray(data.length);
        const step = 256 / Math.max(1, maxColors - 1);
        for (let i = 0; i < data.length; i += 4) {
            result[i] = Math.round(data[i] / step) * step;
            result[i + 1] = Math.round(data[i + 1] / step) * step;
            result[i + 2] = Math.round(data[i + 2] / step) * step;
            result[i + 3] = data[i + 3];
        }
        const quantized = finalizePixels(new ImageData(result, width, height));
        const palette = getPaletteAsObjects(quantized);
        
        return {
            quantized,
            colorsUsed: palette.length,
            palette,
        };
    }
} 

/**
 * Counts the number of unique opaque colors in an ImageData object.
 * @param {ImageData} imgData The image data.
 * @returns {number} The count of unique colors.
 */
export function countColors(imgData) {
  const seen = new Set();
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] > 128) { // Count only opaque colors
      seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    }
    if (seen.size > 256) break; // Optimization for many-colored images
  }
  return seen.size;
}

/**
 * Extracts the unique colors from an ImageData object into an array of objects.
 * @param {ImageData} imgData The image data.
 * @returns {Array<{r: number, g: number, b: number, a: number}>} An array of color objects.
 */
export function getPaletteAsObjects(imgData) {
  const seen = new Set();
  const palette = [];
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2], a = d[i + 3];
    if (a > 128) { // Count only opaque colors
      const key = `${r}|${g}|${b}|${a}`;
      if (!seen.has(key)) {
        seen.add(key);
        palette.push({ r, g, b, a });
      }
    }
  }
  return palette;
}

/**
 * Extracts the unique colors from an ImageData object into a hex palette.
 * @param {ImageData} imgData The image data.
 * @returns {string[]} An array of hex color strings (e.g., "#rrggbbaa").
 */
export function getPaletteFromImage(imgData) {
  const seen = new Set();
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    seen.add((d[i] << 24) | (d[i + 1] << 16) | (d[i + 2] << 8) | d[i + 3]);
  }
  const toHex = val => val.toString(16).padStart(2, '0');
  return Array.from(seen).map(key => {
    const r = (key >>> 24);
    const g = (key >>> 16) & 0xFF;
    const b = (key >>> 8) & 0xFF;
    const a = key & 0xFF;
    return `#${toHex(r)}${toHex(g)}${toHex(b)}${toHex(a)}`;
  });
}

// --- Scaling and Resampling --------------------------------------------------

/**
 * Detects pixel grid scale from a signal using peak analysis.
 * @param {number[]} signal An array of numbers representing e.g., gradient profile.
 * @returns {number} The detected scale factor (at least 1).
 */
export function detectScale(signal) {
  if (signal.length < 3) return 1;

  const meanVal = signal.reduce((a, b) => a + b, 0) / signal.length;
  const std = Math.sqrt(signal.reduce((a, b) => a + (b - meanVal) ** 2, 0) / signal.length);
  const threshold = meanVal + 1.5 * std;

  const peaks = [];
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > threshold && signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
      // Ensure peaks are spaced out
      if (peaks.length === 0 || i - peaks[peaks.length - 1] > 2) {
          peaks.push(i);
      }
    }
  }

  if (peaks.length <= 2) {
    logger.log('detectScale: Not enough peaks found, returning 1.');
    return 1;
  }

  const spacings = peaks.slice(1).map((p, i) => p - peaks[i]);
  logger.log(`detectScale: Found ${peaks.length} peaks, spacings: [${spacings}]`);
  
  // Heuristic: if most spacings are close to the median, use the median.
  const medianSpacing = median(spacings);
  const closeSpacings = spacings.filter(s => Math.abs(s - medianSpacing) <= 2);
  
  if (closeSpacings.length / spacings.length > 0.7) {
    logger.log(`detectScale: Using median spacing: ${medianSpacing}`);
    return Math.round(medianSpacing);
  }

  // Otherwise, return the mode as a fallback.
  const modeSpacing = mode(spacings);
  logger.log(`detectScale: Using fallback mode spacing: ${modeSpacing}`);
  return modeSpacing > 1 ? modeSpacing : 1;
}

/**
 * Finds the optimal crop offset to align an image with a pixel grid.
 * @param {object} grayMat An OpenCV grayscale Mat object.
 * @param {number} scale The grid scale factor.
 * @param {object} cv The OpenCV namespace.
 * @returns {{x: number, y: number}} The optimal (x, y) crop offset.
 */
export function findOptimalCrop(grayMat, scale, cv) {
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
      let bestOffset = 0, maxScore = -1;
      for (let offset = 0; offset < s; offset++) {
        let currentScore = 0;
        for (let i = offset; i < profile.length; i += s) {
          currentScore += profile[i] || 0;
        }
        if (currentScore > maxScore) {
          maxScore = currentScore;
          bestOffset = offset;
        }
      }
      return bestOffset;
    };

    const bestDx = findBestOffset(profileX, scale);
    const bestDy = findBestOffset(profileY, scale);
    logger.log(`Optimal crop found: x=${bestDx}, y=${bestDy}`);
    return { x: bestDx, y: bestDy };

  } finally {
    sobelX.delete();
    sobelY.delete();
  }
}

/**
 * Downscales an image by sampling pixels within blocks.
 * @param {ImageData} imgData The source image data.
 * @param {number} hScale Horizontal scale factor.
 * @param {number} vScale Vertical scale factor.
 * @param {string} [method='median'] Sampling method: 'median', 'mode', 'mean', 'domMean', 'nearest'.
 * @param {number} [domMeanThreshold=0.05] Threshold for 'domMean' method.
 * @returns {ImageData} The downscaled image data.
 */
export function downscaleBlock(imgData, hScale, vScale, method = 'median', domMeanThreshold = 0.05) {
  const targetW = Math.floor(imgData.width / hScale);
  const targetH = Math.floor(imgData.height / vScale);
  if (targetW <= 0 || targetH <= 0) return new ImageData(1, 1);

  const out = new Uint8ClampedArray(targetW * targetH * 4);
  const d = imgData.data;

  for (let ty = 0; ty < targetH; ty++) {
    for (let tx = 0; tx < targetW; tx++) {
      const colorsR = [], colorsG = [], colorsB = [], colorsA = [];

      for (let dy = 0; dy < vScale; dy++) {
        for (let dx = 0; dx < hScale; dx++) {
          const sx = tx * hScale + dx, sy = ty * vScale + dy;
          if (sx >= imgData.width || sy >= imgData.height) continue;

          const idx = (sy * imgData.width + sx) * 4;
          if (d[idx + 3] > 128) { // Only consider mostly opaque pixels for color
            colorsR.push(d[idx]);
            colorsG.push(d[idx + 1]);
            colorsB.push(d[idx + 2]);
          }
          colorsA.push(d[idx + 3]); // Always consider alpha for aggregation
        }
      }

      if (colorsA.length === 0) continue;

      const offset = (ty * targetW + tx) * 4;
      const hasColor = colorsR.length > 0;

      let aggregator;
      switch (method) {
        case 'nearest': aggregator = colors => colors[0] || 0; break;
        case 'mode': aggregator = mode; break;
        case 'mean': aggregator = mean; break;
        case 'median': aggregator = median; break;
        case 'domMean': aggregator = colors => dominantOrMean(colors, domMeanThreshold); break;
        default:
          logger.warn(`Unknown downscale method "${method}", falling back to "median".`);
          aggregator = median;
      }
      
      out[offset]     = hasColor ? aggregator(colorsR) : 0;
      out[offset + 1] = hasColor ? aggregator(colorsG) : 0;
      out[offset + 2] = hasColor ? aggregator(colorsB) : 0;
      // Median is robust for alpha to preserve hard edges
      out[offset + 3] = median(colorsA); 
    }
  }
  return new ImageData(out, targetW, targetH);
}

// --- Math Helpers ------------------------------------------------------------

export const median = (arr) => {
  if (arr.length === 0) return 0;
  const mid = Math.floor(arr.length / 2);
  const sorted = [...arr].sort((a, b) => a - b);
  return arr.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

export const mode = (arr) => {
  if (arr.length === 0) return 0;
  const counts = {};
  let max = 0, res = arr[0];
  for (const v of arr) {
    counts[v] = (counts[v] || 0) + 1;
    if (counts[v] > max) {
      max = counts[v];
      res = v;
    }
  }
  return res;
};

export const mean = (arr) => {
  if (arr.length === 0) return 0;
  return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);
};

/**
 * Uses dominant color if frequent enough, otherwise falls back to the mean.
 * @param {number[]} arr Array of color values.
 * @param {number} [threshold=0.05] Frequency threshold for dominant color (0-1).
 * @returns {number} The aggregated color value.
 */
export function dominantOrMean(arr, threshold = 0.05) {
  if (arr.length === 0) return 0;
  const freq = {};
  arr.forEach(v => freq[v] = (freq[v] || 0) + 1);
  
  const [dominant, count] = Object.entries(freq).reduce(
    (best, cur) => (cur[1] > best[1] ? cur : best),
    [null, 0]
  );
  
  if (dominant !== null && count / arr.length >= threshold) {
    return +dominant;
  }
  return mean(arr);
}

/**
 * Calculates the Greatest Common Divisor (GCD) of an array of numbers.
 * @param {number[]} arr The array of numbers.
 * @returns {number} The GCD of the array.
 */
export function gcdArray(arr) {
  if (!arr.length) return 1;
  let result = arr[0];
  for (let i = 1; i < arr.length; i++) {
    result = gcd(result, arr[i]);
    if (result === 1) return 1; // Early exit
  }
  return result;
}

/**
 * Calculates the GCD of two numbers using the Euclidean algorithm.
 */
function gcd(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) {
    [a, b] = [b, a % b];
  }
  return a;
}

/**
 * Multiplies two 2x2 matrices (a * b).
 * @param {number[][]} a First matrix.
 * @param {number[][]} b Second matrix.
 * @returns {number[][]} The resulting matrix.
 */
export function multiply2x2(a, b) {
  const [a00, a01] = a[0];
  const [a10, a11] = a[1];
  const [b00, b01] = b[0];
  const [b10, b11] = b[1];
  return [
      [a00 * b00 + a01 * b10, a00 * b01 + a01 * b11],
      [a10 * b00 + a11 * b10, a10 * b01 + a11 * b11]
  ];
} 