
// pixelProcessor.js – Browser‑first ES‑module for pixel art processing and vectorization
// Dependencies (all ESM‑compatible):
//   opencv.js  – WASM build, expose `cv` global           (https://docs.opencv.org/4.x/opencv.js)
//   image-q    – Palette quantisation                    (npm i image-q)
//   upng-js    – PNG encoder                             (npm i upng-js)
//   potrace-wasm – Vector tracing                        (npm i potrace-wasm)
//
// Exports two main functions:
// 1. processImage - Pixel art downscaling and optimization
//   Example: import { processImage } from './pixelProcessor.js';
//            const { png, manifest } = await processImage({ file, palette: true });
//
// 2. vectorizeImage - Convert raster images to SVG vectors
//   Example: import { vectorizeImage } from './pixelProcessor.js';
//            const { svg, manifest } = await vectorizeImage({ file, numColors: 'auto' });
//
// Debug logging can be disabled by setting: window.DEBUG_PIXEL_PROCESSOR = false
//--------------------------------------------------------------------------

import * as IQ from 'image-q';
import * as UPNG from 'upng-js';
import { loadFromCanvas } from 'potrace-wasm';

// Система логирования с возможностью включения/выключения
// Для отключения отладочных сообщений установите: window.DEBUG_PIXEL_PROCESSOR = false
const DEBUG = typeof window !== 'undefined' ? window.DEBUG_PIXEL_PROCESSOR !== false : true;
const logger = {
  log: (...args) => DEBUG && console.log(...args),
  warn: (...args) => DEBUG && console.warn(...args),
  error: (...args) => console.error(...args), // Ошибки показываем всегда
};

/**
 * Waits until OpenCV WASM is ready. Returns the cv namespace.
 * A simplified and robust version.
 */
async function cvReady() {
  return new Promise((resolve, reject) => {
    logger.log('cvReady: checking OpenCV status...');

    if (typeof cv !== 'undefined' && cv.getBuildInformation) {
      logger.log('cvReady: OpenCV is already ready.');
      return resolve(cv);
    }

    const interval = setInterval(() => {
      // Modern OpenCV 4.x creates a global promise `cv`
      if (typeof cv !== 'undefined' && typeof cv.then === 'function') {
        logger.log('cvReady: Found cv promise, awaiting resolution...');
        clearInterval(interval);
        cv.then(resolve).catch(reject);
        return;
      }
      // Classic check for older versions or when `onRuntimeInitialized` has already fired
      if (typeof cv !== 'undefined' && cv.getBuildInformation) {
        logger.log('cvReady: OpenCV has initialized.');
        clearInterval(interval);
        resolve(cv);
      }
    }, 100);

    setTimeout(() => {
      clearInterval(interval);
      reject(new Error('OpenCV failed to load within 30 seconds'));
    }, 30000);
  });
}

/**
 * Reads user‑supplied File/Blob into ImageData (RGBA Uint8ClampedArray).
 */
export async function fileToImageData(file) {
  const bitmap = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

// ---------- Helpers ------------------------------------------------------
const median = (arr) => {
  const mid = Math.floor(arr.length / 2);
  const sorted = [...arr].sort((a, b) => a - b);
  return arr.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

const mode = (arr) => {
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

const mean = (arr) => {
  return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);
};

function countColors(imgData) {
  const seen = new Set();
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    if (seen.size > 256) break; // cap for perf
  }
  return seen.size;
}

function downscaleBlock(imgData, hScale, vScale, targetW, targetH, method = 'median', alphaThreshold = 0) {
  const out = new Uint8ClampedArray(targetW * targetH * 4);
  const d = imgData.data;
  for (let ty = 0; ty < targetH; ty++) {
    for (let tx = 0; tx < targetW; tx++) {
      const colorsR = [], colorsG = [], colorsB = [], colorsA = [], pixels = [];
      for (let dy = 0; dy < vScale; dy++) {
        for (let dx = 0; dx < hScale; dx++) {
          const sx = tx * hScale + dx;
          const sy = ty * vScale + dy;
          // Boundary check for safety
          if (sx >= imgData.width || sy >= imgData.height) continue;
          const idx = (sy * imgData.width + sx) * 4;
          let pixel = [d[idx], d[idx + 1], d[idx + 2], d[idx + 3]];
          if (pixel[3] < alphaThreshold) pixel = [0, 0, 0, 0];
          colorsR.push(pixel[0]);
          colorsG.push(pixel[1]);
          colorsB.push(pixel[2]);
          colorsA.push(pixel[3]);
          pixels.push(pixel);
        }
      }
      if (pixels.length === 0) continue;

      const offset = (ty * targetW + tx) * 4;
      if (method === 'mode') {
        out[offset] = mode(colorsR);
        out[offset + 1] = mode(colorsG);
        out[offset + 2] = mode(colorsB);
        out[offset + 3] = mode(colorsA);
      } else if (method === 'mean') {
        out[offset] = mean(colorsR);
        out[offset + 1] = mean(colorsG);
        out[offset + 2] = mean(colorsB);
        out[offset + 3] = mean(colorsA);
      } else if (method === 'contrast') {
        const med = [median(colorsR), median(colorsG), median(colorsB), median(colorsA)];
        let maxDist = -1, best = med;
        for (const px of pixels) {
          const dist = Math.abs(px[0] - med[0]) + Math.abs(px[1] - med[1]) + Math.abs(px[2] - med[2]);
          if (dist > maxDist) { maxDist = dist; best = px; }
        }
        out[offset] = best[0]; out[offset + 1] = best[1]; out[offset + 2] = best[2]; out[offset + 3] = best[3];
      } else if (method === 'nearest') {
        const cx = Math.floor(hScale / 2), cy = Math.floor(vScale / 2);
        const cidx = cy * hScale + cx;
        const px = pixels[cidx] || pixels[0];
        out[offset] = px[0]; out[offset + 1] = px[1]; out[offset + 2] = px[2]; out[offset + 3] = px[3];
      } else { // 'median' is default
        out[offset] = median(colorsR);
        out[offset + 1] = median(colorsG);
        out[offset + 2] = median(colorsB);
        out[offset + 3] = median(colorsA);
      }
    }
  }
  return new ImageData(out, targetW, targetH);
}

/**
 * Apply morphological opening to clean up noise before quantization
 */
async function morphologicalCleanup(imgData) {
  const cv = await cvReady();
  const mat = cv.matFromImageData(imgData);
  const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.morphologyEx(mat, mat, cv.MORPH_OPEN, kernel);
  const cleanedData = new ImageData(new Uint8ClampedArray(mat.data), mat.cols, mat.rows);
  mat.delete(); kernel.delete();
  return cleanedData;
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
  if (peaks.length > 2) {
    const spacings = peaks.slice(1).map((p, i) => p - peaks[i]);
    return Math.round(median(spacings));
  }
  return 1;
}

async function edgeAwareDetect(imgData, manualScale) {
  if (manualScale) {
    const [hScale, vScale] = manualScale;
    const baseScale = Math.max(hScale, vScale);
    return {
      hScale, vScale, baseScale,
      targetW: Math.floor(imgData.width / hScale),
      targetH: Math.floor(imgData.height / vScale)
    };
  }

  const cv = await cvReady();
  const cvImg = cv.matFromImageData(imgData);
  const gray = new cv.Mat();
  cv.cvtColor(cvImg, gray, cv.COLOR_RGBA2GRAY);
  const edges = new cv.Mat();
  cv.Canny(gray, edges, 50, 150);
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  let roiRect;
  if (contours.size() > 0) {
    let largest = contours.get(0);
    for (let i = 1; i < contours.size(); i++) {
      if (cv.contourArea(contours.get(i)) > cv.contourArea(largest)) largest = contours.get(i);
    }
    const rect = cv.boundingRect(largest);
    const pad = Math.min(20, rect.width * 0.1, rect.height * 0.1);
    roiRect = new cv.Rect(
      Math.max(0, rect.x - pad), Math.max(0, rect.y - pad),
      Math.min(imgData.width - (rect.x - pad), rect.width + 2 * pad),
      Math.min(imgData.height - (rect.y - pad), rect.height + 2 * pad)
    );
    largest.delete();
  } else {
    roiRect = new cv.Rect(0, 0, imgData.width, imgData.height);
  }

  const roiMat = gray.roi(roiRect);
  const diffH = new cv.Mat(); const diffV = new cv.Mat();
  cv.Sobel(roiMat, diffH, cv.CV_32F, 1, 0);
  cv.Sobel(roiMat, diffV, cv.CV_32F, 0, 1);

  const reduceSum = (mat, axis) => {
    const sums = new Float32Array(axis === 0 ? mat.cols : mat.rows).fill(0);
    const data = mat.data32F;
    for (let y = 0; y < mat.rows; y++) {
      for (let x = 0; x < mat.cols; x++) {
        const val = Math.abs(data[y * mat.cols + x]);
        if (axis === 0) sums[x] += val; else sums[y] += val;
      }
    }
    return sums;
  };
  const hSum = reduceSum(diffH, 0);
  const vSum = reduceSum(diffV, 1);

  const hScale = detectScale(Array.from(hSum));
  const vScale = detectScale(Array.from(vSum));

  const candidates = new Set();
  for (let s = Math.max(3, hScale - 2); s <= hScale + 2; s++) candidates.add(s);
  for (let s = Math.max(3, vScale - 2); s <= vScale + 2; s++) candidates.add(s);
  if (candidates.size === 0) candidates.add(Math.round((hScale + vScale) / 2));

  let bestScale = Math.round((hScale + vScale) / 2) || 1;
  let maxFitScore = -1;
  for (const s of candidates) {
    if (s <= 1) continue;
    const fitW = 1.0 - ((roiRect.width % s) / s);
    const fitH = 1.0 - ((roiRect.height % s) / s);
    const score = fitW + fitH;
    if (score > maxFitScore) { maxFitScore = score; bestScale = s; }
  }
  const finalScale = bestScale;

  cvImg.delete(); gray.delete(); edges.delete();
  // ИСПРАВЛЕНО: Добавлена очистка для contours и hierarchy
  contours.delete(); hierarchy.delete();
  roiMat.delete(); diffH.delete(); diffV.delete();

  return {
    hScale: finalScale, vScale: finalScale, baseScale: finalScale,
    targetW: Math.floor(imgData.width / finalScale),
    targetH: Math.floor(imgData.height / finalScale)
  };
}

function quantizeImage(imgData, max) {
  if (typeof IQ === 'undefined') {
    logger.warn('image-q library not loaded, using fallback quantization');
    return fallbackQuantize(imgData, max);
  }
  
  try {
    // Используем высокоуровневый API согласно документации
    // https://ibezkrovnyi.github.io/image-quantization/
    
    // Создаем PointContainer из ImageData
    const inPointContainer = IQ.utils.PointContainer.fromImageData(imgData);
    
    // Создаем палитру с помощью buildPaletteSync
    const palette = IQ.buildPaletteSync([inPointContainer], {
      colorDistanceFormula: 'euclidean',
      paletteQuantization: 'wuquant',
      colors: max
    });
    
    // Применяем палитру к изображению с помощью applyPaletteSync
    const outPointContainer = IQ.applyPaletteSync(inPointContainer, palette, {
      colorDistanceFormula: 'euclidean',
      imageQuantization: 'nearest'
    });
    
    // Получаем результат как Uint8Array
    const quantizedData = outPointContainer.toUint8Array();
    
    // Конвертируем обратно в ImageData
    const quantizedImg = new ImageData(
      new Uint8ClampedArray(quantizedData), 
      imgData.width, 
      imgData.height
    );
    
    return { 
      quantized: quantizedImg, 
      colorsUsed: palette.getPointContainer().getPointArray().length 
    };
  } catch (error) {
    logger.warn('image-q quantization failed, using fallback:', error);
    return fallbackQuantize(imgData, max);
  }
}

/**
 * Простая fallback функция квантования без внешних зависимостей
 */
function fallbackQuantize(imgData, maxColors) {
  const { data, width, height } = imgData;
  const colorMap = new Map();
  
  // Собираем все цвета
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    
    // Огрубляем цвета для уменьшения их количества
    const quantizeFactor = Math.max(1, Math.floor(256 / maxColors));
    const qr = Math.floor(r / quantizeFactor) * quantizeFactor;
    const qg = Math.floor(g / quantizeFactor) * quantizeFactor;
    const qb = Math.floor(b / quantizeFactor) * quantizeFactor;
    const qa = Math.floor(a / quantizeFactor) * quantizeFactor;
    
    const key = `${qr},${qg},${qb},${qa}`;
    colorMap.set(key, { r: qr, g: qg, b: qb, a: qa });
  }
  
  // Если цветов все еще слишком много, берем только самые частые
  const colors = Array.from(colorMap.values());
  if (colors.length > maxColors) {
    // Простая стратегия: берем первые maxColors цветов
    colors.splice(maxColors);
  }
  
  // Применяем квантование
  const result = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    
    // Находим ближайший цвет из палитры
    let minDist = Infinity;
    let bestColor = colors[0];
    
    for (const color of colors) {
      const dist = Math.abs(r - color.r) + Math.abs(g - color.g) + Math.abs(b - color.b) + Math.abs(a - color.a);
      if (dist < minDist) {
        minDist = dist;
        bestColor = color;
      }
    }
    
    result[i] = bestColor.r;
    result[i + 1] = bestColor.g;
    result[i + 2] = bestColor.b;
    result[i + 3] = bestColor.a;
  }
  
  return {
    quantized: new ImageData(result, width, height),
    colorsUsed: colors.length
  };
}

function findOptimalCrop(grayMat, scale, cv) {
  const sobelX = new cv.Mat(); const sobelY = new cv.Mat();
  cv.Sobel(grayMat, sobelX, cv.CV_32F, 1, 0, 3);
  cv.Sobel(grayMat, sobelY, cv.CV_32F, 0, 1, 3);

  const profileX = new Float32Array(grayMat.cols).fill(0);
  const profileY = new Float32Array(grayMat.rows).fill(0);
  const dataX = sobelX.data32F; const dataY = sobelY.data32F;
  for (let y = 0; y < grayMat.rows; y++) {
    for (let x = 0; x < grayMat.cols; x++) {
      const idx = y * grayMat.cols + x;
      // Используем абсолютное значение градиента для правильного профиля
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
 * Автоматически определяет оптимальное количество цветов в изображении,
 * используя более агрессивную кластеризацию и анализ доминирующих цветов.
 */
async function detectOptimalColorCount(imgData, { 
    downsampleTo = 64, // Уменьшили для более быстрого анализа
    colorQuantizeFactor = 48, // Увеличили для более агрессивной группировки
    dominanceThreshold = 0.015 // 1.5% от общего количества пикселей
} = {}) {
    const cv = await cvReady();
    let src = cv.matFromImageData(imgData);
    
    // 1. Уменьшаем изображение еще сильнее для анализа общей структуры
    const aspectRatio = src.rows / src.cols;
    const targetWidth = downsampleTo;
    const targetHeight = Math.round(targetWidth * aspectRatio);
    const dsize = new cv.Size(targetWidth, targetHeight);
    let smallMat = new cv.Mat();
    cv.resize(src, smallMat, dsize, 0, 0, cv.INTER_AREA);
    
    // 2. Применяем более агрессивное размытие для устранения градиентов и шума
    cv.medianBlur(smallMat, smallMat, 5); // Увеличили с 3 до 5
    
    // 3. Дополнительное гауссово размытие для сглаживания
    const kernel = new cv.Size(5, 5);
    cv.GaussianBlur(smallMat, smallMat, kernel, 1, 1);

    const colorCounts = new Map();
    const totalPixels = smallMat.rows * smallMat.cols;
    const d = smallMat.data;

    // 4. Собираем статистику цветов с более агрессивной квантизацией
    for (let i = 0; i < d.length; i += 4) {
        const alpha = d[i + 3];
        if (alpha < 200) continue; // Игнорируем прозрачные/полупрозрачные пиксели

        // Более агрессивная квантизация цветов
        const r = Math.round(d[i] / colorQuantizeFactor) * colorQuantizeFactor;
        const g = Math.round(d[i + 1] / colorQuantizeFactor) * colorQuantizeFactor;
        const b = Math.round(d[i + 2] / colorQuantizeFactor) * colorQuantizeFactor;

        const colorKey = `${r},${g},${b}`;
        colorCounts.set(colorKey, (colorCounts.get(colorKey) || 0) + 1);
    }
    
    // 5. Анализируем доминирующие цвета
    const minPixelsForDominance = Math.max(3, Math.round(totalPixels * dominanceThreshold));
    const dominantColors = [];
    
    console.log(`Total pixels: ${totalPixels}, min pixels for dominance: ${minPixelsForDominance}`);
    console.log(`Found ${colorCounts.size} unique quantized colors`);
    
    // Сортируем цвета по частоте использования
    const sortedColors = Array.from(colorCounts.entries())
        .sort((a, b) => b[1] - a[1]) // Сортируем по убыванию количества пикселей
        .filter(([color, count]) => count >= minPixelsForDominance);
    
    console.log('Top 10 colors by frequency:');
    sortedColors.slice(0, 10).forEach(([color, count], index) => {
        const percentage = ((count / totalPixels) * 100).toFixed(1);
        console.log(`  ${index + 1}. rgb(${color}) - ${count} pixels (${percentage}%)`);
    });
    
    // 6. Применяем дополнительную фильтрацию для удаления переходных цветов
    let significantColors = sortedColors.length;
    
    // Если у нас слишком много цветов, применяем более строгие критерии
    if (significantColors > 32) {
        // Оставляем только цвета, составляющие больше 2% от изображения
        const strictThreshold = Math.max(minPixelsForDominance, Math.round(totalPixels * 0.02));
        significantColors = sortedColors.filter(([, count]) => count >= strictThreshold).length;
        console.log(`After strict filtering (2% threshold): ${significantColors} colors`);
    }
    
    src.delete();
    smallMat.delete();

    // 7. Возвращаем разумное количество цветов
    const result = Math.max(2, Math.min(significantColors, 32)); // Уменьшили максимум с 64 до 32
    console.log(`Final auto-detected color count: ${result}`);
    return result;
}

/**
 * Извлекает уникальные цвета из квантованного ImageData
 */
function getPaletteFromQuantized(imgData) {
    const seen = new Map();
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
        const r = d[i], g = d[i+1], b = d[i+2], a = d[i+3];
        const key = (r << 24) | (g << 16) | (b << 8) | a;
        if (!seen.has(key)) {
            seen.set(key, {r, g, b, a});
        }
    }
    return Array.from(seen.values());
}

/**
 * Создает бинарную маску (черное на белом) для одного цвета
 */
function createMask(imgData, targetColor) {
    console.log(`Creating mask for color: rgb(${targetColor.r}, ${targetColor.g}, ${targetColor.b})`);
    const mask = new ImageData(imgData.width, imgData.height);
    const d = imgData.data;
    const m = mask.data;
    let blackPixels = 0;
    
    for (let i = 0; i < d.length; i += 4) {
        const r = d[i], g = d[i+1], b = d[i+2], a = d[i+3];
        // Если пиксель совпадает с целевым цветом, делаем его черным, иначе - белым.
        if (r === targetColor.r && g === targetColor.g && b === targetColor.b && a === targetColor.a) {
            m[i] = 0; m[i+1] = 0; m[i+2] = 0; m[i+3] = 255; // Black
            blackPixels++;
        } else {
            m[i] = 255; m[i+1] = 255; m[i+2] = 255; m[i+3] = 255; // White
        }
    }
    
    console.log(`Mask created: ${blackPixels} black pixels out of ${imgData.width * imgData.height}`);
    return mask;
}

/**
 * Трассирует один цвет в SVG путь
 */
async function traceColor(imgData, color, options) {
    console.log(`traceColor called for: rgb(${color.r}, ${color.g}, ${color.b})`);
    try {
        // Проверяем, загружен ли potrace-wasm
        if (typeof loadFromCanvas === 'undefined') {
            console.warn('potrace-wasm not loaded, using fallback');
            return { paths: createFallbackPath(imgData, color), viewBox: null };
        }
        
        console.log('Creating canvas for potrace...');
        // Создаем canvas для potrace (с поддержкой Web Workers)
        const canvas = typeof OffscreenCanvas !== 'undefined' 
            ? new OffscreenCanvas(imgData.width, imgData.height) 
            : document.createElement('canvas'); // Fallback для старых браузеров/сред

        if (canvas instanceof HTMLCanvasElement) { // Только если это не OffscreenCanvas
            canvas.width = imgData.width;
            canvas.height = imgData.height;
        }

        const ctx = canvas.getContext('2d');
        
        // Рисуем маску на canvas
        const mask = createMask(imgData, color);
        ctx.putImageData(mask, 0, 0);
        
        console.log('Canvas created, calling potrace-wasm...');
        // Используем potrace-wasm
        const result = await loadFromCanvas(canvas, {
            turdsize: options.turdsize,
            alphamax: options.alphamax,
            opticurve: options.opticurve,
        });
        
        console.log('Potrace result type:', typeof result);
        console.log('Potrace result preview:', typeof result === 'string' ? result.substring(0, 200) : result);
        
        // potrace-wasm возвращает SVG строку, нужно извлечь из неё пути и viewBox
        if (typeof result === 'string' && result.includes('<path')) {
            console.log('Parsing SVG string from potrace...');
            
            // Используем DOMParser для извлечения элементов
            const parser = new DOMParser();
            const svgDoc = parser.parseFromString(result, 'image/svg+xml');
            
            // Извлекаем размеры из корневого SVG элемента
            const svgElement = svgDoc.querySelector('svg');
            let viewBox = svgElement ? svgElement.getAttribute('viewBox') : null;
            const width = svgElement ? svgElement.getAttribute('width') : null;
            const height = svgElement ? svgElement.getAttribute('height') : null;
            
            console.log('SVG attributes from potrace:', { viewBox, width, height });
            
            // Если нет viewBox, но есть width/height, создаем viewBox
            if (!viewBox && width && height) {
                viewBox = `0 0 ${width.replace('pt', '')} ${height.replace('pt', '')}`;
                console.log('Generated viewBox from width/height:', viewBox);
            }
            
                         const pathElements = svgDoc.querySelectorAll('path');
             
             // Если координаты путей намного больше viewBox, пересчитываем viewBox
             if (pathElements.length > 0 && viewBox) {
                 const samplePath = pathElements[0].getAttribute('d');
                 const coordMatch = samplePath?.match(/M(\d+)\s+(\d+)/);
                 if (coordMatch) {
                     const maxCoord = Math.max(parseInt(coordMatch[1]), parseInt(coordMatch[2]));
                     const viewBoxParts = viewBox.split(' ').map(parseFloat);
                     const viewBoxMax = Math.max(viewBoxParts[2], viewBoxParts[3]);
                     
                     console.log('Coordinate analysis:', { maxCoord, viewBoxMax, ratio: maxCoord / viewBoxMax });
                     
                     // Если координаты намного больше viewBox, используем координаты для viewBox
                     if (maxCoord > viewBoxMax * 2) {
                         // Найдем максимальные координаты среди всех путей
                         let maxX = 0, maxY = 0;
                         pathElements.forEach(pathEl => {
                             const d = pathEl.getAttribute('d');
                             const coords = d?.match(/[ML](\d+)\s+(\d+)/g);
                             coords?.forEach(coord => {
                                 const match = coord.match(/[ML](\d+)\s+(\d+)/);
                                 if (match) {
                                     maxX = Math.max(maxX, parseInt(match[1]));
                                     maxY = Math.max(maxY, parseInt(match[2]));
                                 }
                             });
                         });
                         
                         // Добавляем небольшой отступ
                         const newViewBox = `0 0 ${Math.ceil(maxX * 1.1)} ${Math.ceil(maxY * 1.1)}`;
                         console.log('Recalculated viewBox based on coordinates:', newViewBox);
                         viewBox = newViewBox;
                     }
                 }
             }
             
             console.log('Final viewBox for this color:', viewBox);
            console.log('Found path elements:', pathElements.length);
            
            if (pathElements.length > 0) {
                // Собираем все d-атрибуты в один массив путей
                const pathStrings = [];
                pathElements.forEach(pathEl => {
                    const d = pathEl.getAttribute('d');
                    if (d && d.trim().length > 0) {
                        pathStrings.push(d.trim());
                    }
                });
                
                console.log('Valid paths found:', pathStrings.length);
                
                if (pathStrings.length > 0) {
                    // Фильтруем проблемные гигантские пути от potrace
                    const validPathStrings = pathStrings.filter(d => {
                        // Игнорируем пути, которые начинаются с команды рисования гигантского квадрата
                        // Эти пути обычно выходят далеко за пределы изображения
                        const startsWithGiantSquare = d.startsWith('M0 5120') || 
                                                     d.startsWith('M0 2560') || 
                                                     d.startsWith('M0 1280') ||
                                                     d.includes('l0 -5120') ||
                                                     d.includes('l0 -2560') ||
                                                     d.includes('l0 -1280');
                        
                        // Также игнорируем очень длинные пути (потенциально проблемные)
                        const tooLong = d.length > 100000;
                        
                        if (startsWithGiantSquare) {
                            console.log('⚠️  Filtered out giant square path:', d.substring(0, 100) + '...');
                        }
                        if (tooLong) {
                            console.log('⚠️  Filtered out very long path:', d.length, 'chars');
                        }
                        
                        return !startsWithGiantSquare && !tooLong;
                    });
                    
                    console.log(`Path filtering: ${pathStrings.length} → ${validPathStrings.length} paths`);
                    
                    if (validPathStrings.length > 0) {
                        // Правильно форматируем цвет с поддержкой альфа-канала
                        const rgbaColor = color.a !== undefined && color.a !== 255 
                            ? `rgba(${color.r}, ${color.g}, ${color.b}, ${(color.a / 255).toFixed(3)})` 
                            : `rgb(${color.r}, ${color.g}, ${color.b})`;
                        
                        console.log(`Using color: ${rgbaColor} for path generation`);
                        
                        // Возвращаем каждый путь как отдельный <path> элемент для лучшего контроля
                        // Добавляем более толстую обводку того же цвета для заполнения зазоров
                        const strokeColor = rgbaColor; // Используем тот же цвет для обводки
                        const strokeWidth = Math.max(1, Math.ceil(Math.min(imgData.width, imgData.height) / 200)); // Адаптивная толщина
                        
                        const svgPaths = validPathStrings.map(pathData => 
                            `<path d="${pathData}" fill="${rgbaColor}" fill-rule="nonzero" stroke="${strokeColor}" stroke-width="${strokeWidth}" stroke-linejoin="round" stroke-linecap="round" shape-rendering="geometricPrecision" />`
                        );
                        
                        const result = svgPaths.join('\n');
                        console.log('Generated SVG paths length:', result.length);
                        return { paths: result, viewBox };
                    }
                }
            }
        }
        
        console.log('No valid paths found in potrace result');
        return { paths: '', viewBox: null };
    } catch (error) {
        console.warn(`Failed to trace color ${color.r},${color.g},${color.b}:`, error);
        // Fallback: создаем простой прямоугольник для этого цвета
        return { paths: createFallbackPath(imgData, color), viewBox: null };
    }
}

/**
 * Создает простой fallback path для цвета (минимальная растеризация)
 */
function createFallbackPath(imgData, color) {
    logger.warn(`Creating fallback path for color: rgb(${color.r}, ${color.g}, ${color.b})`);
    
    // Вместо одного большого прямоугольника, создаем набор маленьких квадратов
    // только для пикселей этого цвета (простая растеризация)
    const paths = [];
    const { data, width, height } = imgData;
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const r = data[idx], g = data[idx + 1], b = data[idx + 2], a = data[idx + 3];
            
            if (r === color.r && g === color.g && b === color.b && a === color.a) {
                paths.push(`M${x},${y}h1v1h-1z`); // Квадрат 1x1 пиксель
            }
        }
    }
    
    if (paths.length === 0) {
        logger.warn('No pixels found for color, returning empty path');
        return '';
    }
    
    // Правильно форматируем цвет с поддержкой альфа-канала
    const rgbaColor = color.a !== undefined && color.a !== 255 
        ? `rgba(${color.r}, ${color.g}, ${color.b}, ${(color.a / 255).toFixed(3)})` 
        : `rgb(${color.r}, ${color.g}, ${color.b})`;
        
    // Используем тот же цвет для обводки в fallback режиме с адаптивной толщиной
    const strokeColor = rgbaColor;
    const strokeWidth = Math.max(1, Math.ceil(Math.min(width, height) / 200));
        
    console.log(`Creating fallback path with color: ${rgbaColor}, stroke-width: ${strokeWidth}`);
    const fallbackPath = `<path d="${paths.join('')}" fill="${rgbaColor}" stroke="${strokeColor}" stroke-width="${strokeWidth}" stroke-linejoin="round" stroke-linecap="round" shape-rendering="geometricPrecision" />`;
    logger.log(`Fallback path created with ${paths.length} pixel rectangles`);
    return fallbackPath;
}

/**
 * Векторизует изображение в SVG
 */
export async function vectorizeImage({
  file,
  numColors = 'auto',
  turdSize = 10,
  alphaMax = 1.0,
  opticurve = true,
  preProcess = {
    enabled: true,
    filter: 'bilateral',
    value: 15,
    morphology: true, // Новый параметр: включить морфологию
    morphologyKernel: 5 // Новый параметр: размер ядра
  }
}) {
    logger.log('vectorizeImage called with:', { file: file.name, numColors, preProcess });
    const t0 = performance.now();

    const cv = await cvReady();
    const imgData = await fileToImageData(file);
    const originalSize = [imgData.width, imgData.height];
    
    // Определение количества цветов
    let finalNumColors;
    if (numColors === 'auto' || typeof numColors !== 'number') {
        logger.log("Detecting optimal color count automatically...");
        finalNumColors = await detectOptimalColorCount(imgData, { 
            downsampleTo: 128, 
            colorQuantizeFactor: 24, 
            minPixelCount: 50
        });
    } else {
        finalNumColors = numColors;
    }
    
    let src = cv.matFromImageData(imgData);

    // --- ПРЕДОБРАБОТКА ---
    if (preProcess.enabled) {
        console.log(`Applying pre-processing filter: ${preProcess.filter}`);
        const processedMat = new cv.Mat();
        if (preProcess.filter === 'bilateral') {
            // Билатеральный фильтр
            const d = preProcess.value; // Диаметр
            const sigmaColor = d * 2;
            const sigmaSpace = d / 2;
            const channels = new cv.MatVector();
            cv.split(src, channels);
            const rgbMat = new cv.Mat();
            const rgbChannels = new cv.MatVector();
            rgbChannels.push_back(channels.get(0));
            rgbChannels.push_back(channels.get(1));
            rgbChannels.push_back(channels.get(2));
            cv.merge(rgbChannels, rgbMat);
            const filteredRgb = new cv.Mat();
            cv.bilateralFilter(rgbMat, filteredRgb, d, sigmaColor, sigmaSpace);
            const filteredChannels = new cv.MatVector();
            cv.split(filteredRgb, filteredChannels);
            const finalChannels = new cv.MatVector();
            finalChannels.push_back(filteredChannels.get(0));
            finalChannels.push_back(filteredChannels.get(1));
            finalChannels.push_back(filteredChannels.get(2));
            finalChannels.push_back(channels.get(3));
            cv.merge(finalChannels, processedMat);
            rgbMat.delete();
            filteredRgb.delete();
            channels.delete();
            rgbChannels.delete();
            filteredChannels.delete();
            finalChannels.delete();
            // --- Новый блок: морфологическое закрытие ---
            if (preProcess.morphology !== false) {
                const kernelSize = preProcess.morphologyKernel || 5;
                const kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8U);
                cv.morphologyEx(processedMat, processedMat, cv.MORPH_CLOSE, kernel);
                kernel.delete();
                console.log(`Applied morphological closing with kernel size ${kernelSize}`);
            }
        } else { // 'median'
            let ksize = preProcess.value;
            if (ksize % 2 === 0) ksize++;
            cv.medianBlur(src, processedMat, ksize);
        }
        src.delete();
        src = processedMat;
    }

    // Конвертируем Mat обратно в ImageData для квантования
    const preProcessedImageData = new ImageData(
        new Uint8ClampedArray(src.data),
        src.cols,
        src.rows
    );
    src.delete(); // Очищаем память OpenCV

    // Квантование цветов
    console.log(`Quantizing image to ${finalNumColors} colors...`);
    const { quantized, colorsUsed } = quantizeImage(preProcessedImageData, finalNumColors);
    console.log(`Quantization complete. Colors used: ${colorsUsed}`);

    // Трассировка
    console.log('Tracing color layers...');
    const { data, width, height } = quantized;

    // Получаем уникальные цвета из палитры
    let palette = getPaletteFromQuantized(quantized);
    console.log('Palette colors found:', palette.length);
    console.log('Palette preview:', palette.slice(0, 5));
    
    // Добавляем детальное логирование палитры
    console.log('Full palette details:');
    palette.forEach((color, index) => {
        const rgbaString = color.a !== undefined && color.a !== 255 
            ? `rgba(${color.r}, ${color.g}, ${color.b}, ${(color.a / 255).toFixed(3)})` 
            : `rgb(${color.r}, ${color.g}, ${color.b})`;
        console.log(`  Color ${index}: ${rgbaString}`);
    });

    // Сортируем слои по площади (количеству пикселей) для правильного Z-индекса
    console.log('Sorting palette by pixel count for proper layering...');
    const pixelCounts = new Map(palette.map(c => [JSON.stringify(c), 0]));
    
    // Подсчитываем количество пикселей для каждого цвета
    for (let i = 0; i < quantized.data.length; i += 4) {
        const r = quantized.data[i], g = quantized.data[i+1], b = quantized.data[i+2], a = quantized.data[i+3];
        if (a === 0) continue; // Пропускаем прозрачные пиксели
        const key = JSON.stringify({ r, g, b, a });
        if (pixelCounts.has(key)) {
            pixelCounts.set(key, pixelCounts.get(key) + 1);
        }
    }
    
    // Определяем фоновый цвет (самый частый)
    console.log('Detecting background color...');
    let backgroundColor = null;
    let maxCount = -1;
    
    for (const [colorStr, count] of pixelCounts.entries()) {
        const color = JSON.parse(colorStr);
        // Исключаем полностью прозрачные цвета
        if (color.a === 0) continue;
        
        if (count > maxCount) {
            maxCount = count;
            backgroundColor = { 
                ...color, 
                string: color.a !== undefined && color.a !== 255 
                    ? `rgba(${color.r}, ${color.g}, ${color.b}, ${(color.a / 255).toFixed(3)})` 
                    : `rgb(${color.r}, ${color.g}, ${color.b})`
            };
        }
    }
    
    console.log('Background color detected:', backgroundColor);
    console.log(`Background covers ${maxCount} pixels (${((maxCount / (width * height)) * 100).toFixed(1)}% of image)`);
    
    // Отфильтруем палитру: исключаем фоновый цвет из трассировки
    const paletteForTracing = palette.filter(color => {
        if (!backgroundColor) return true;
        
        return !(color.r === backgroundColor.r && 
                 color.g === backgroundColor.g && 
                 color.b === backgroundColor.b && 
                 color.a === backgroundColor.a);
    });
    
    console.log(`Palette filtered: ${palette.length} → ${paletteForTracing.length} colors (background excluded)`);
    console.log('Colors for tracing:', paletteForTracing.slice(0, 3).map(c => 
        c.a !== undefined && c.a !== 255 
            ? `rgba(${c.r}, ${c.g}, ${c.b}, ${(c.a / 255).toFixed(3)})` 
            : `rgb(${c.r}, ${c.g}, ${c.b})`
    ));

    // Вычисляем яркость для каждого цвета (perceived brightness)
    const getBrightness = (color) => {
        // Используем формулу для perceived brightness
        return (0.299 * color.r + 0.587 * color.g + 0.114 * color.b);
    };
    
    // Функция сортировки палитры с разными стратегиями
    const sortPalette = (palette, pixelCounts, strategy = 'intelligent') => {
        console.log(`Applying ${strategy} layer sorting...`);
        
        switch (strategy) {
            case 'byCount': // От частых к редким (оригинальная логика)
                return [...palette].sort((a, b) => {
                    const countA = pixelCounts.get(JSON.stringify(a)) || 0;
                    const countB = pixelCounts.get(JSON.stringify(b)) || 0;
                    return countB - countA;
                });
                
            case 'byCountReverse': // От редких к частым
                return [...palette].sort((a, b) => {
                    const countA = pixelCounts.get(JSON.stringify(a)) || 0;
                    const countB = pixelCounts.get(JSON.stringify(b)) || 0;
                    return countA - countB;
                });
                
            case 'byBrightness': // От ярких к темным
                return [...palette].sort((a, b) => {
                    return getBrightness(b) - getBrightness(a);
                });
                
            case 'byBrightnessReverse': // От темных к ярким
                return [...palette].sort((a, b) => {
                    return getBrightness(a) - getBrightness(b);
                });
                
            case 'intelligent': // Интеллектуальная комбинация
            default:
                return [...palette].sort((a, b) => {
                    const countA = pixelCounts.get(JSON.stringify(a)) || 0;
                    const countB = pixelCounts.get(JSON.stringify(b)) || 0;
                    const brightnessA = getBrightness(a);
                    const brightnessB = getBrightness(b);
                    
                    // Если один цвет значительно ярче другого, более яркий идет первым (фон)
                    const brightnessDiff = Math.abs(brightnessA - brightnessB);
                    if (brightnessDiff > 50) { // порог различия яркости
                        return brightnessB - brightnessA; // более яркие первыми
                    }
                    
                    // Если яркости похожи, сортируем по количеству пикселей
                    // Более частые цвета (фон) идут первыми
                    return countB - countA;
                });
        }
    };
    
    // Попробуем разные стратегии сортировки
    // TODO: Сделать это настраиваемым через параметры
    const sortingStrategy = 'byCount'; // можно поменять на 'byCount', 'byCountReverse', 'byBrightness', 'byBrightnessReverse'
    palette = sortPalette(palette, pixelCounts, sortingStrategy);
    
    console.log(`Palette sorted using '${sortingStrategy}' strategy:`);
    palette.slice(0, 5).forEach((c, i) => {
        const count = pixelCounts.get(JSON.stringify(c)) || 0;
        const brightness = getBrightness(c);
        const rgbaString = c.a !== undefined && c.a !== 255 
            ? `rgba(${c.r}, ${c.g}, ${c.b}, ${(c.a / 255).toFixed(3)})` 
            : `rgb(${c.r}, ${c.g}, ${c.b})`;
        console.log(`  ${i}: ${rgbaString} - ${count} pixels, brightness: ${brightness.toFixed(1)}`);
    });

    const svgPaths = [];

    for (const color of paletteForTracing) { // Используем отфильтрованную палитру
        if (color.a === 0) continue; // Пропускаем полностью прозрачный цвет

        const rgbaString = color.a !== undefined && color.a !== 255 
            ? `rgba(${color.r}, ${color.g}, ${color.b}, ${(color.a / 255).toFixed(3)})` 
            : `rgb(${color.r}, ${color.g}, ${color.b})`;
        console.log(`Tracing color: ${rgbaString}`);
        
        try {
            // Вычисляем адаптивные параметры для уменьшения зазоров
            const adaptiveTurdSize = Math.max(turdSize, Math.floor(Math.min(width, height) / 30)); // Увеличили делитель для более грубой трассировки
            const adaptiveAlphaMax = Math.min(alphaMax, 0.5); // Уменьшили для более гладких кривых
            console.log(`Using adaptive params: turdSize=${adaptiveTurdSize} (original: ${turdSize}), alphaMax=${adaptiveAlphaMax} (original: ${alphaMax}), image: ${width}x${height}`);
            
            const { paths, viewBox } = await traceColor(quantized, color, {
                turdsize: adaptiveTurdSize,
                alphamax: adaptiveAlphaMax,
                opticurve: opticurve,
            });
            console.log(`Trace result length: ${paths.length}`);
            if (paths.length > 0) {
                console.log(`✓ Successfully created SVG path for ${rgbaString}`);
                svgPaths.push({ paths, viewBox });
            } else {
                console.log(`✗ No SVG path created for ${rgbaString} (empty result)`);
            }
        } catch (error) {
            console.warn(`✗ Failed to trace color ${rgbaString}:`, error);
        }
    }

    console.log(`Total SVG paths generated: ${svgPaths.length}`);
    console.log('Sample path:', svgPaths[0]?.paths?.substring(0, 100));

    // Собираем финальный SVG
    console.log('Assembling final SVG...');

    // Используем viewBox от potrace если он есть, иначе оригинальные размеры
    let finalViewBox = `0 0 ${width} ${height}`;
    if (svgPaths.length > 0 && svgPaths[0].viewBox) {
        finalViewBox = svgPaths[0].viewBox;
        console.log('Using viewBox from potrace:', finalViewBox);
    } else {
        console.log('Using original viewBox:', finalViewBox);
    }

    // Создаем responsive SVG с правильным viewBox
    // Добавляем transform для исправления переворота потrace (Y-координаты инвертированы)
    const viewBoxParts = finalViewBox.split(' ');
    const viewBoxHeight = parseFloat(viewBoxParts[3]) || height;
    const svgHeader = `<svg viewBox="${finalViewBox}" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%; max-width: 100%; max-height: 100%;" shape-rendering="geometricPrecision"><g transform="scale(1,-1) translate(0,-${viewBoxHeight})">`;
    
    // Добавляем фон как простой прямоугольник (ПРАВИЛЬНЫЙ СПОСОБ)
    // Нужно использовать размеры из viewBox для фона
    const bgWidth = viewBoxParts[2] || width;
    const bgHeight = viewBoxParts[3] || height;
    
    const backgroundRect = backgroundColor 
        ? `<rect x="${viewBoxParts[0] || 0}" y="${viewBoxParts[1] || 0}" width="${bgWidth}" height="${bgHeight}" fill="${backgroundColor.string}" />` 
        : '';
    
    console.log('Background rect:', backgroundRect || 'No background');
    
    const svgFooter = `</g></svg>`;
    const finalSVG = `${svgHeader}\n${backgroundRect}\n${svgPaths.map(item => item.paths).join('\n')}\n${svgFooter}`;

    const t1 = performance.now();
    const manifest = {
        original_size: originalSize,
        final_size: [width, height],
        processing_steps: { 
            num_colors_mode: typeof numColors === 'string' ? 'auto' : 'manual',
            num_colors_target: finalNumColors, 
            colors_used: colorsUsed,
            preprocess: preProcess,
            morphology_applied: preProcess.enabled && preProcess.filter === 'bilateral' && preProcess.morphology !== false
        },
        processing_time_ms: Math.round(t1 - t0),
        timestamp: new Date().toISOString()
    };

    return { svg: finalSVG, manifest };
}

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

export async function processImage({
  file,
  maxColors = 128,
  palette = false,
  snapGrid = false,
  manualScale,
  downscaleMethod = 'median',
  alphaThreshold = 0,
  cleanup = false
}) {
  console.log('processImage called with:', { file: file.name, maxColors, palette, snapGrid, manualScale, downscaleMethod, alphaThreshold });
  const t0 = performance.now();
  const cv = await cvReady();

  const imgData = await fileToImageData(file);
  const originalSize = [imgData.width, imgData.height];
  let currentImgMat = cv.matFromImageData(imgData);

  // 1. Determine the scale.
  console.log('Detecting pixel scale...');
  const { hScale, vScale, baseScale } = await edgeAwareDetect(imgData, manualScale);
  console.log('Scales detected:', { hScale, vScale, baseScale });

  // 2. (NEW STEP) If snapGrid, find and apply cropping.
  if (snapGrid && baseScale > 1) {
    console.log('Snapping to grid via auto-crop...');
    const grayMat = new cv.Mat(); // Create gray version for cropping analysis
    cv.cvtColor(currentImgMat, grayMat, cv.COLOR_RGBA2GRAY);
    const crop = findOptimalCrop(grayMat, baseScale, cv);
    grayMat.delete();

    const rect = new cv.Rect(crop.x, crop.y, currentImgMat.cols - crop.x, currentImgMat.rows - crop.y);
    let croppedMat = currentImgMat.roi(rect).clone(); // Clone to be safe
    currentImgMat.delete();
    currentImgMat = croppedMat;
    console.log(`Image cropped to: ${currentImgMat.cols} x ${currentImgMat.rows}`);
  }

  // 3. Convert Mat to ImageData for further processing.
  let current;
  if (typeof cv.imageDataFromMat === 'function') {
    current = cv.imageDataFromMat(currentImgMat);
  } else {
    const canvas = document.createElement('canvas'); // Use regular canvas in workers
    canvas.width = currentImgMat.cols;
    canvas.height = currentImgMat.rows;
    cv.imshow(canvas, currentImgMat);
    current = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, currentImgMat.cols, currentImgMat.rows);
  }
  // ИСПРАВЛЕНО: Удаляем матрицу после всех возможных использований.
  currentImgMat.delete();

  // 4. Morphological cleanup (if needed).
  if (cleanup) {
    console.log('Applying morphological cleanup...');
    current = await morphologicalCleanup(current);
  }

  // 5. Quantization (if needed).
  let colorsUsed = countColors(current);
  if (palette) {
    console.log('Quantizing palette to', Math.min(maxColors, 256), 'colors...');
    const q = quantizeImage(current, Math.min(maxColors, 256));
    current = q.quantized;
    colorsUsed = q.colorsUsed;
    console.log('Palette quantized, colors used:', colorsUsed);
  }

  // 6. Downscale.
  const targetW = Math.floor(current.width / hScale);
  const targetH = Math.floor(current.height / vScale);
  console.log(`Downscaling from ${current.width}x${current.height} to ${targetW}x${targetH} using hScale=${hScale}, vScale=${vScale}...`);
  current = downscaleBlock(current, hScale, vScale, targetW, targetH, downscaleMethod, alphaThreshold);

  // 7. PNG Encoding.
  console.log('Encoding PNG...');
  if (typeof UPNG === 'undefined') throw new Error('UPNG library not loaded.');
  const pngBuf = UPNG.encode([current.data.buffer], current.width, current.height, 0);
  if (!pngBuf || pngBuf.byteLength === 0) throw new Error('PNG encoding failed - empty result');
  console.log('PNG encoded, size:', pngBuf.byteLength, 'bytes');

  // === Palette extraction ===
  const paletteArr = getPaletteFromQuantized(current).map(c => {
    if (c.a !== undefined && c.a !== 255) {
      return `rgba(${c.r},${c.g},${c.b},${(c.a / 255).toFixed(3)})`;
    } else {
      return `#${((1 << 24) + (c.r << 16) + (c.g << 8) + c.b).toString(16).slice(1)}`;
    }
  });

  const manifest = {
    original_size: originalSize,
    detected_scale: { horizontal: hScale, vertical: vScale, base: baseScale },
    final_size: [current.width, current.height],
    processing_steps: { snap_grid: snapGrid, palette, cleanup, downscale_method: downscaleMethod },
    colors_used: colorsUsed,
    processing_time_ms: Math.round(performance.now() - t0),
    timestamp: new Date().toISOString()
  };
  return { png: new Uint8Array(pngBuf), manifest, palette: paletteArr, imageData: current };
}

export default { processImage, vectorizeImage };