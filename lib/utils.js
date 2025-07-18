// utils.js – Common utilities for pixel art and vector processing
// Shared functions used by both pixel.js and vector.js modules

/**
 * Waits until OpenCV WASM is ready. Returns the cv namespace.
 * A simplified and robust version.
 */
export async function cvReady() {
  return new Promise((resolve, reject) => {
    console.log('cvReady: checking OpenCV status...');
    console.log('cvReady: typeof cv =', typeof cv);
    console.log('cvReady: cv instanceof Promise =', cv instanceof Promise);

    // Handle case where cv is a Promise (like in the example)
    if (cv instanceof Promise) {
      console.log('cvReady: cv is a Promise, awaiting resolution...');
      cv.then(resolvedCv => {
        console.log('cvReady: Promise resolved, cv object:', typeof resolvedCv);
        resolve(resolvedCv);
      }).catch(error => {
        console.log('cvReady: Promise rejected:', error);
        reject(error);
      });
      return;
    }

    // Check if OpenCV is already ready
    if (typeof cv !== 'undefined' && cv.getBuildInformation) {
      console.log('cvReady: OpenCV is already ready.');
      console.log('cvReady: Resolving with cv object');
      return resolve(cv);
    }

    // Check if we're in a browser environment
    if (typeof window === 'undefined') {
      return reject(new Error('OpenCV is not available in this environment'));
    }

    // Check if OpenCV script is loaded
    if (typeof cv === 'undefined') {
      console.log('cvReady: OpenCV not loaded yet, waiting...');
      
      // Wait for OpenCV to be available (simplified approach)
      const checkInterval = setInterval(() => {
        if (typeof cv !== 'undefined') {
          clearInterval(checkInterval);
          console.log('cvReady: OpenCV script loaded');
          
          // Handle Promise case
          if (cv instanceof Promise) {
            cv.then(resolvedCv => {
              console.log('cvReady: Promise resolved after loading');
              resolve(resolvedCv);
            }).catch(reject);
          } else if (cv.getBuildInformation) {
            console.log('cvReady: OpenCV ready after loading');
            resolve(cv);
          } else {
            reject(new Error('OpenCV loaded but not properly initialized'));
          }
        }
      }, 100);

      // Timeout for script loading
      setTimeout(() => {
        clearInterval(checkInterval);
        reject(new Error('OpenCV script failed to load within 10 seconds'));
      }, 10000);
      
      return;
    }

    // OpenCV is loaded but not ready, wait for initialization
    if (!cv.getBuildInformation) {
      console.log('cvReady: OpenCV loaded but not ready, waiting for initialization...');
      
      const initInterval = setInterval(() => {
        if (cv.getBuildInformation) {
          console.log('cvReady: OpenCV initialized');
          clearInterval(initInterval);
          resolve(cv);
        }
      }, 100);

      setTimeout(() => {
        clearInterval(initInterval);
        reject(new Error('OpenCV failed to initialize within 20 seconds'));
      }, 20000);
      
      return;
    }
  });
}

/**
 * Reads user‑supplied File/Blob into ImageData (RGBA Uint8ClampedArray).
 */
export async function fileToImageData(file) {
  logger.log('fileToImageData: Starting conversion of file:', file.name, 'size:', file.size);
  
  // Safety check for file size
  if (file.size > 50 * 1024 * 1024) { // 50MB limit
    throw new Error(`File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB. Maximum supported size is 50MB.`);
  }
  
  logger.log('fileToImageData: About to call createImageBitmap...');
  
  // Add timeout for createImageBitmap
  const timeoutPromise = new Promise((_, reject) => {
    setTimeout(() => reject(new Error('Image loading timeout - file may be corrupted or too large')), 30000); // 30 seconds
  });
  
  const bitmapPromise = createImageBitmap(file);
  const bitmap = await Promise.race([bitmapPromise, timeoutPromise]);
  
  logger.log('fileToImageData: ImageBitmap created, dimensions:', bitmap.width, 'x', bitmap.height);
  logger.log('fileToImageData: Creating OffscreenCanvas...');
  
  // Check if OffscreenCanvas is supported, fallback to regular canvas
  let canvas;
  if (typeof OffscreenCanvas !== 'undefined') {
    canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  } else {
    // Fallback for older browsers
    canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }
  logger.log('fileToImageData: Getting 2D context...');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  logger.log('fileToImageData: Drawing image to canvas...');
  ctx.drawImage(bitmap, 0, 0);
  logger.log('fileToImageData: Getting ImageData...');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  logger.log('fileToImageData: ImageData created successfully');
  return imageData;
}

/**
 * Apply morphological opening to clean up noise before quantization
 */
export async function morphologicalCleanup(imgData) {
  const cv = await cvReady();
  const mat = cv.matFromImageData(imgData);
  const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.morphologyEx(mat, mat, cv.MORPH_OPEN, kernel);
  const cleanedData = new ImageData(new Uint8ClampedArray(mat.data), mat.cols, mat.rows);
  mat.delete(); kernel.delete();
  return cleanedData;
}

// ---------- Math Helpers ------------------------------------------------------
export const median = (arr) => {
  const mid = Math.floor(arr.length / 2);
  const sorted = [...arr].sort((a, b) => a - b);
  return arr.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

export const mode = (arr) => {
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
  return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);
};

/**
 * Count unique colors in ImageData
 */
export function countColors(imgData) {
  const seen = new Set();
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    if (seen.size > 256) break; // cap for perf
  }
  return seen.size;
}

/**
 * Detect scale from signal analysis
 * SIMPLIFIED: Back to basics with reliable peak detection
 */
export function detectScale(signal) {
  logger.log('detectScale called with signal length:', signal.length);
  if (signal.length < 10) {
    logger.log('detectScale: signal too short, returning 1');
    return 1;
  }
  
  // SIMPLIFIED: Single robust threshold
  const sorted = [...signal].sort((a, b) => a - b);
  const q80 = sorted[Math.floor(sorted.length * 0.8)];
  const q20 = sorted[Math.floor(sorted.length * 0.2)];
  const threshold = q80 + (q80 - q20) * 0.2; // Conservative but not too high
  
  logger.log('detectScale: calculated threshold:', threshold);
  
  // SIMPLIFIED: Basic peak detection
  const peaks = [];
  const minPeakSpacing = Math.max(2, Math.floor(signal.length / 50));
  
  for (let i = 2; i < signal.length - 2; i++) {
    if (signal[i] > threshold && 
        signal[i] > signal[i - 1] && signal[i] > signal[i - 2] &&
        signal[i] > signal[i + 1] && signal[i] > signal[i + 2]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minPeakSpacing) {
        peaks.push(i);
      }
    }
  }
  
  logger.log('detectScale: found peaks:', peaks.length);
  
  if (peaks.length >= 3) {
    const spacings = peaks.slice(1).map((p, i) => p - peaks[i]);
    const result = Math.round(median(spacings));
    logger.log('detectScale: calculated scale:', result, 'from spacings:', spacings);
    
    // SIMPLIFIED: Basic validation
    if (result >= 2 && result <= 20) {
      return result;
    }
  }
  
  if (peaks.length >= 2) {
    const spacing = peaks[1] - peaks[0];
    logger.log('detectScale: using fallback spacing:', spacing);
    if (spacing >= 2 && spacing <= 20) {
      return spacing;
    }
  }
  
  logger.log('detectScale: no clear pattern found, returning 1');
  return 1;
}

/**
 * Find optimal crop position for grid snapping
 */
export function findOptimalCrop(grayMat, scale, cv) {
  const sobelX = new cv.Mat(); const sobelY = new cv.Mat();
  cv.Sobel(grayMat, sobelX, cv.CV_32F, 1, 0, 3);
  cv.Sobel(grayMat, sobelY, cv.CV_32F, 0, 1, 3);

  const profileX = new Float32Array(grayMat.cols).fill(0);
  const profileY = new Float32Array(grayMat.rows).fill(0);
  const dataX = sobelX.data32F; const dataY = sobelY.data32F;
  for (let y = 0; y < grayMat.rows; y++) {
    for (let x = 0; x < grayMat.cols; x++) {
      const idx = y * grayMat.cols + x;
      // Using absolute gradient value for correct profile
      profileX[x] += Math.abs(dataX[idx]);
      profileY[y] += Math.abs(dataY[idx]);
    }
  }

  const findBestOffset = (profile, s) => {
    let bestOffset = 0, maxScore = -1;
    for (let offset = 0; offset < s; offset++) {
      let currentScore = 0;
      for (let i = offset; i < profile.length; i += s) {
        if (profile[i]) currentScore += profile[i];
      }
      if (currentScore > maxScore) { maxScore = currentScore; bestOffset = offset; }
    }
    return bestOffset;
  };

  const bestDx = findBestOffset(profileX, scale);
  const bestDy = findBestOffset(profileY, scale);
  sobelX.delete(); sobelY.delete();
  console.log(`Optimal crop found: x=${bestDx}, y=${bestDy}`);
  return { x: bestDx, y: bestDy };
}

/**
 * Logger utility with debug control
 */
export const logger = {
  log: (...args) => {
    const DEBUG = typeof window !== 'undefined' ? window.DEBUG_PIXEL_PROCESSOR !== false : true;
    DEBUG && console.log(...args);
  },
  warn: (...args) => {
    const DEBUG = typeof window !== 'undefined' ? window.DEBUG_PIXEL_PROCESSOR !== false : true;
    DEBUG && console.warn(...args);
  },
  error: (...args) => console.error(...args), // Always show errors
}; 