/**
 * pixel.js - A module for advanced pixel art optimization.
 * Handles intelligent scale detection, color quantization, downscaling, and PNG encoding.
 */

import * as IQ from 'image-q';
import {
    fileToImageData,
    morphologicalCleanup,
    jaggyCleaner,
    alphaBinarization,
    countColors,
    detectScale,
    gcdArray,
    median,
    mode,
    mean,
    logger,
    findOptimalCrop,
    dominantOrMean,
    finalizePixels,
    encodePng,
    getPaletteFromImage,
    downscaleBlock,
    multiply2x2,
    quantizeImage,
    withCv
} from './utils.js';
import SVD from 'svd';

/**
 * Detects pixel art scale by analyzing color run lengths. This method is very reliable for
 * "clean" pixel art with uniform block sizes.
 * @param {ImageData} imgData - The input image data.
 * @returns {number} The detected scale factor (Greatest Common Divisor of run lengths).
 */
export function runsBasedDetect(imgData) {
    const { data, width, height } = imgData;
    const allRunLens = [];
    const scanRuns = (isHorizontal) => {
        const primaryDim = isHorizontal ? height : width;
        const secondaryDim = isHorizontal ? width : height;
        for (let i = 0; i < primaryDim; i++) {
            let currentRunLength = 1;
            for (let j = 1; j < secondaryDim; j++) {
                const idx1 = isHorizontal ? (i * width + j) * 4 : (j * width + i) * 4;
                const idx2 = isHorizontal ? idx1 - 4 : ((j - 1) * width + i) * 4;
                const isSamePixel = data[idx1] === data[idx2] && data[idx1 + 1] === data[idx2 + 1] && data[idx1 + 2] === data[idx2 + 2] && data[idx1 + 3] === data[idx2 + 3];
                if (isSamePixel) {
                    currentRunLength++;
                } else {
                    if (currentRunLength > 1) allRunLens.push(currentRunLength);
                    currentRunLength = 1;
                }
            }
            if (currentRunLength > 1) allRunLens.push(currentRunLength);
        }
    };
    scanRuns(true);
    scanRuns(false);
    if (allRunLens.length < 10) {
        return 1;
    }
    const detectedScale = gcdArray(allRunLens);
    logger.log(`Runs-based detection found scale: ${detectedScale}`);
    return Math.max(1, detectedScale);
}

/**
 * Detects pixel art scale using a robust, memory-efficient edge-aware algorithm.
 * @param {ImageData} imgData The input image data.
 * @returns {Promise<number>} The detected scale factor.
 */
export async function edgeAwareDetect(imgData) {
    if (imgData.width * imgData.height > 4_000_000) {
        logger.warn('Image > 4MP, using runs-based detection for performance.');
        return runsBasedDetect(imgData);
    }
    return withCv(async (cv, track) => {
        try {
            const srcMat = track(cv.matFromImageData(imgData));
            const gray = track(new cv.Mat());
            cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);
            const edges = track(new cv.Mat());
            cv.Canny(gray, edges, 50, 150);
            const contours = track(new cv.MatVector());
            const hierarchy = track(new cv.Mat());
            cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            let roiRect;
            if (contours.size() > 0) {
                let largest = contours.get(0);
                for (let i = 1; i < contours.size(); i++) {
                    if (cv.contourArea(contours.get(i)) > cv.contourArea(largest)) {
                        largest = contours.get(i);
                    }
                }
                const rect = cv.boundingRect(largest);
                const pad = Math.min(20, rect.width * 0.1, rect.height * 0.1);
                roiRect = new cv.Rect(Math.max(0, rect.x - pad), Math.max(0, rect.y - pad), Math.min(imgData.width - (rect.x - pad), rect.width + 2 * pad), Math.min(imgData.height - (rect.y - pad), rect.height + 2 * pad));
            } else {
                logger.warn('No contours found for ROI, using image center.');
                const w = gray.cols, h = gray.rows;
                roiRect = new cv.Rect(Math.floor(w * 0.25), Math.floor(h * 0.25), Math.floor(w * 0.5), Math.floor(h * 0.5));
            }
            if (roiRect.width < 3 || roiRect.height < 3) {
                logger.warn(`Calculated ROI is too small (${roiRect.width}x${roiRect.height}), cannot perform edge detection. Returning 1.`);
                return 1;
            }
            const roiMat = track(gray.roi(roiRect));
            const diffH = track(new cv.Mat());
            const diffV = track(new cv.Mat());
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
                if (score > maxFitScore) {
                    maxFitScore = score;
                    bestScale = s;
                }
            }
            logger.log(`Edge-aware detection (ROI: ${roiRect.width}x${roiRect.height}) found scale: ${bestScale} (h: ${hScale}, v: ${vScale})`);
            return bestScale;
        } catch (error) {
            logger.error('Edge-aware detection failed critically:', error);
            return 1;
        }
    });
}

/**
 * Enhanced downscaling method using dominant color.
 * Prevents artifact colors by working with full colors (RGB)
 * rather than individual channels. Works with already quantized colors,
 * which guarantees using only colors from the original palette.
 * @param {ImageData} imgData - Source image data (already quantized).
 * @param {number} scale - Scale factor (same for H and V).
 * @param {number} threshold - Threshold for using dominant color (0.05 = 5%).
 * @returns {ImageData} Downscaled image.
 */
export function downscaleByDominantColor(imgData, scale, threshold = 0.05) {
    const targetW = Math.floor(imgData.width / scale);
    const targetH = Math.floor(imgData.height / scale);
    const outData = new Uint8ClampedArray(targetW * targetH * 4);
    const srcData = imgData.data;
    const srcWidth = imgData.width;

    for (let ty = 0; ty < targetH; ty++) {
        for (let tx = 0; tx < targetW; tx++) {
            const colorCounts = new Map();
            const alphaValues = [];
            let dominantColor = 0; // Black
            let maxCount = 0;

            // Scan scale x scale block
            for (let dy = 0; dy < scale; dy++) {
                for (let dx = 0; dx < scale; dx++) {
                    const sx = tx * scale + dx;
                    const sy = ty * scale + dy;
                    const idx = (sy * srcWidth + sx) * 4;

                    const a = srcData[idx + 3];
                    alphaValues.push(a);

                    // Only consider opaque pixels for color determination
                    if (a > 128) {
                        const r = srcData[idx];
                        const g = srcData[idx + 1];
                        const b = srcData[idx + 2];
                        // Use 24-bit integer as color key
                        const colorInt = (r << 16) | (g << 8) | b;
                        const newCount = (colorCounts.get(colorInt) || 0) + 1;
                        colorCounts.set(colorInt, newCount);

                        if (newCount > maxCount) {
                            maxCount = newCount;
                            dominantColor = colorInt;
                        }
                    }
                }
            }

            const outIdx = (ty * targetW + tx) * 4;

            if (maxCount > 0) {
                // Collect all full colors (RGB) for analysis
                const colors = [];
                for (let dy = 0; dy < scale; dy++) {
                    for (let dx = 0; dx < scale; dx++) {
                        const sx = tx * scale + dx;
                        const sy = ty * scale + dy;
                        const idx = (sy * srcWidth + sx) * 4;
                        const a = srcData[idx + 3];
                        if (a > 128) {
                            const r = srcData[idx];
                            const g = srcData[idx + 1];
                            const b = srcData[idx + 2];
                            // Use 24-bit integer as color key
                            const colorInt = (r << 16) | (g << 8) | b;
                            colors.push(colorInt);
                        }
                    }
                }

                // Find dominant color or use mean
                let finalColor;
                if (colors.length > 0) {
                    const freq = {};
                    colors.forEach(color => freq[color] = (freq[color] || 0) + 1);
                    const [dominantColor, count] = Object.entries(freq)
                        .reduce((best, cur) => cur[1] > best[1] ? cur : best);

                    if (count / colors.length >= threshold) {
                        // Use dominant color
                        finalColor = +dominantColor;
                    } else {
                        // Use mean value for each channel
                        const rSum = colors.reduce((sum, color) => sum + ((color >> 16) & 0xFF), 0);
                        const gSum = colors.reduce((sum, color) => sum + ((color >> 8) & 0xFF), 0);
                        const bSum = colors.reduce((sum, color) => sum + (color & 0xFF), 0);
                        const avgR = Math.round(rSum / colors.length);
                        const avgG = Math.round(gSum / colors.length);
                        const avgB = Math.round(bSum / colors.length);
                        finalColor = (avgR << 16) | (avgG << 8) | avgB;
                    }
                } else {
                    finalColor = 0; // Black color
                }

                // Write result
                outData[outIdx] = (finalColor >> 16) & 0xFF; // R
                outData[outIdx + 1] = (finalColor >> 8) & 0xFF;  // G
                outData[outIdx + 2] = finalColor & 0xFF;         // B
                outData[outIdx + 3] = median(alphaValues) > 128 ? 255 : 0; // Binary alpha
            } else {
                // If block is completely transparent, make pixel black and transparent
                outData[outIdx] = 0;
                outData[outIdx + 1] = 0;
                outData[outIdx + 2] = 0;
                outData[outIdx + 3] = 0;
            }
        }
    }

    return new ImageData(outData, targetW, targetH);
}

/**
 * The core content-adaptive downscaling algorithm.
 * This function now processes an OpenCV Mat in Lab color space.
 * @param {cv.Mat} srcLab - The source image as a 32-bit float Mat in CIELAB color space.
 * @param {number} targetW - The target width.
 * @param {number} targetH - The target height.
 * @param {cv} cv - The OpenCV instance.
 * @param {function} track - The memory tracking function from withCv.
 * @returns {cv.Mat} The downscaled image as a Mat in CIELAB color space.
 */
function _contentAdaptiveCore(srcLab, targetW, targetH, cv, track) {
    logger.warn('This method is computationally intensive and may take a while.');
    const NUM_ITERATIONS = 5;

    const { cols: wi, rows: hi } = srcLab;
    const wo = targetW, ho = targetH;
    const rx = wi / wo, ry = hi / ho;

    let labPlanes = track(new cv.MatVector());
    cv.split(srcLab, labPlanes);
    const L_plane = track(labPlanes.get(0));
    const a_plane = track(labPlanes.get(1));
    const b_plane = track(labPlanes.get(2));

    let mu_k = [], Sigma_k = [], nu_k = [];

    // Initialization
    for (let yk = 0; yk < ho; yk++) {
        for (let xk = 0; xk < wo; xk++) {
            const k_idx = yk * wo + xk;
            mu_k[k_idx] = [(xk + 0.5) * rx, (yk + 0.5) * ry];
            const initial_sx = (rx / 3) * (rx / 3);
            const initial_sy = (ry / 3) * (ry / 3);
            Sigma_k[k_idx] = [initial_sx, 0, 0, initial_sy];
            // Initialize with a neutral color (gray)
            nu_k[k_idx] = [50.0, 0.0, 0.0];
        }
    }

    // EM-C Iterations
    for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
        logger.log(`Iteration ${iter + 1} / ${NUM_ITERATIONS}`);

        // E-Step
        const gamma_sum_per_pixel = new Float32Array(wi * hi).fill(1e-9);
        let w_ki = new Array(wo * ho).fill(0).map(() => new Map());

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

        // M-Step
        let next_mu_k = new Array(wo * ho);
        let next_Sigma_k = new Array(wo * ho);
        let next_nu_k = new Array(wo * ho);

        for (let k = 0; k < wo * ho; k++) {
            let w_sum = 1e-9;
            let new_mu = [0, 0];
            let new_nu = [0, 0, 0];
            for (const [i, wk] of w_ki[k].entries()) {
                const gamma_k_i = wk / gamma_sum_per_pixel[i];
                w_sum += gamma_k_i;
                const yi = Math.floor(i / wi);
                const xi = i % wi;
                new_mu[0] += gamma_k_i * xi; new_mu[1] += gamma_k_i * yi;
                new_nu[0] += gamma_k_i * L_plane.data32F[i];
                new_nu[1] += gamma_k_i * a_plane.data32F[i];
                new_nu[2] += gamma_k_i * b_plane.data32F[i];
            }
            new_mu[0] /= w_sum; new_mu[1] /= w_sum;
            new_nu[0] /= w_sum; new_nu[1] /= w_sum; new_nu[2] /= w_sum;
            next_mu_k[k] = new_mu;
            next_nu_k[k] = new_nu;

            let new_Sigma_arr = [0, 0, 0, 0];
            for (const [i, wk] of w_ki[k].entries()) {
                const gamma_k_i = wk / gamma_sum_per_pixel[i];
                const yi = Math.floor(i / wi); const xi = i % wi;
                const dx = xi - new_mu[0]; const dy = yi - new_mu[1];
                new_Sigma_arr[0] += gamma_k_i * dx * dx;
                new_Sigma_arr[1] += gamma_k_i * dx * dy;
                new_Sigma_arr[3] += gamma_k_i * dy * dy;
            }
            new_Sigma_arr[0] /= w_sum; new_Sigma_arr[1] /= w_sum;
            new_Sigma_arr[2] = new_Sigma_arr[1];
            new_Sigma_arr[3] /= w_sum;
            next_Sigma_k[k] = new_Sigma_arr;
        }

        // C-Step (Clamping)
        for (let k = 0; k < wo * ho; k++) {
            const sigma_arr = next_Sigma_k[k];
            const sigma_mat2d = [[sigma_arr[0], sigma_arr[1]], [sigma_arr[2], sigma_arr[3]]];

            const { u, q, v } = SVD(sigma_mat2d);

            // !!! RESTORE ORIGINAL VALUES FOR SHARPNESS !!!
            // These values force kernels to be small, which prevents blurring.
            q[0] = Math.max(0.05, Math.min(q[0], 0.1));
            q[1] = Math.max(0.05, Math.min(q[1], 0.1));

            const s_diag = [[q[0], 0], [0, q[1]]];
            const v_t = [[v[0][0], v[1][0]], [v[0][1], v[1][1]]];

            const temp = multiply2x2(u, s_diag);
            const new_sigma_mat2d = multiply2x2(temp, v_t);

            const final_sigma = [new_sigma_mat2d[0][0], new_sigma_mat2d[0][1], new_sigma_mat2d[1][0], new_sigma_mat2d[1][1]];

            mu_k[k] = next_mu_k[k];
            Sigma_k[k] = final_sigma;
            nu_k[k] = next_nu_k[k];
        }
    }

    // Final image construction from kernels
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

/**
 * EXPERIMENTAL: Content-adaptive downscaling.
 * High-quality but computationally expensive. Separates Alpha channel for proper handling.
 * based on https://johanneskopf.de/publications/downscaling/
 * @param {ImageData} imgData - Source image.
 * @param {number} targetW - Target width.
 * @param {number} targetH - Target height.
 * @returns {Promise<ImageData>} Downscaled image.
 */
export async function contentAdaptiveDownscale(imgData, targetW, targetH) {
    if (typeof SVD !== 'function') {
        throw new Error("SVD.js library is required for content-adaptive downscaling.");
    }
    logger.warn('Using experimental, CPU-intensive content-adaptive downscaling.');

    return withCv(async (cv, track) => {
        const srcMat = track(cv.matFromImageData(imgData));
        const channels = track(new cv.MatVector());
        cv.split(srcMat, channels);

        // 1. Downscale Alpha channel separately using area interpolation (best for shrinking).
        const alpha = track(channels.get(3));
        const outAlpha = track(new cv.Mat());
        cv.resize(alpha, outAlpha, new cv.Size(targetW, targetH), 0, 0, cv.INTER_AREA);

        // 2. Process RGB channels with the content-adaptive algorithm.
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

        const outLab = _contentAdaptiveCore(srcLab, targetW, targetH, cv, track);

        const outRGB_32f = track(new cv.Mat());
        cv.cvtColor(outLab, outRGB_32f, cv.COLOR_Lab2RGB);
        const outRGB_8u = track(new cv.Mat());
        outRGB_32f.convertTo(outRGB_8u, cv.CV_8U, 255.0, 0);

        // 3. Merge downscaled RGB with downscaled Alpha.
        const outRgbSplit = track(new cv.MatVector());
        cv.split(outRGB_8u, outRgbSplit);
        const finalRgbaMat = track(new cv.Mat());
        const rgbaVec = track(new cv.MatVector());
        rgbaVec.push_back(outRgbSplit.get(0));
        rgbaVec.push_back(outRgbSplit.get(1));
        rgbaVec.push_back(outRgbSplit.get(2));
        rgbaVec.push_back(outAlpha);
        cv.merge(rgbaVec, finalRgbaMat);

        return new ImageData(new Uint8ClampedArray(finalRgbaMat.data), targetW, targetH);
    });
}

/** Main image processing pipeline. */
export async function processImage({
    file,
    maxColors = 32,
    manualScale = null,
    detectMethod = 'auto', // 'auto', 'runs', 'edge'
    downscaleMethod = 'dominant', // 'dominant', 'median', 'mode', 'mean', 'nearest', 'content-adaptive'
    domMeanThreshold = 0.05,
    cleanup = { morph: false, jaggy: false },
    fixedPalette = null,
    alphaThreshold = 128,
    snapGrid = true,
}) {
    if (!file) throw new Error('No file provided.');

    const t0 = performance.now();
    let current = await fileToImageData(file);
    const originalSize = [current.width, current.height];

    if (current.width > 8000 || current.height > 8000 || (current.width * current.height > 10_000_000)) {
        throw new Error(`Image too large: ${current.width}x${current.height}.`);
    }

    // 1. Pre-processing: Binarize alpha
    if (alphaThreshold !== null) {
        current = alphaBinarization(current, alphaThreshold);
    }
    const originalForAdaptiveScale = current;

    // 2. Scale Detection
    let scale = 1;
    if (manualScale) {
        scale = Math.max(1, Array.isArray(manualScale) ? manualScale[0] : manualScale);
        logger.log(`Using manual scale: ${scale}`);
    } else {
        const detectionFn = {
            'runs': runsBasedDetect,
            'edge': edgeAwareDetect,
            'auto': async (img) => {
                const runsScale = runsBasedDetect(img);
                if (runsScale > 1) {
                    logger.log('Auto-detect: "runs" method was successful.');
                    return runsScale;
                }
                logger.log('Auto-detect: "runs" failed, falling back to "edge".');
                return await edgeAwareDetect(img);
            }
        }[detectMethod];

        scale = await detectionFn(originalForAdaptiveScale);
    }
    if (scale <= 1 && downscaleMethod !== 'content-adaptive') {
        logger.log('Scale is 1, skipping downscale and grid snapping.');
    }

    // 2.5. NEW Snap to Grid
    if (snapGrid && scale > 1) {
        logger.log('Snapping to grid via auto-crop...');
        await withCv(async (cv, track) => {
            const srcMat = track(cv.matFromImageData(current));
            const grayMat = track(new cv.Mat());
            cv.cvtColor(srcMat, grayMat, cv.COLOR_RGBA2GRAY);

            // Find optimal offset
            const crop = findOptimalCrop(grayMat, scale, cv);

            // CHANGED: Calculate new size divisible by scale for pixel-perfect result.
            const newWidth = Math.floor((current.width - crop.x) / scale) * scale;
            const newHeight = Math.floor((current.height - crop.y) / scale) * scale;

            if (newWidth < scale || newHeight < scale) {
                logger.warn(`Snapping failed: resulting image size (${newWidth}x${newHeight}) is too small. Skipping snap.`);
            } else {
                const rect = new cv.Rect(crop.x, crop.y, newWidth, newHeight);
                const croppedMat = track(srcMat.roi(rect).clone());

                // Convert back to ImageData
                const canvas = document.createElement('canvas'); // document.createElement is fine for this utility
                canvas.width = croppedMat.cols;
                canvas.height = croppedMat.rows;
                cv.imshow(canvas, croppedMat);
                current = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, croppedMat.cols, croppedMat.rows);

                logger.log(`Image cropped to: ${current.width} x ${current.height} (from offset x:${crop.x}, y:${crop.y})`);
            }
        });
    }

    // 3. Optional Pre-Processing Cleanup
    if (cleanup.morph) {
        current = await morphologicalCleanup(current);
    }

    // 4. Color Quantization (before downscaling)
    const initialColors = countColors(current);
    let colorsUsed = initialColors;
    let quantizeAfter = false;

    if (downscaleMethod === 'content-adaptive') {
        // Не квантизуем до даунскейла
    } else if (maxColors < 256 && initialColors > maxColors) {
        logger.log(`Quantizing from ${initialColors} to a max of ${maxColors} colors.`);
        const quantResult = quantizeImage(current, maxColors, fixedPalette);
        current = quantResult.quantized;
        colorsUsed = quantResult.colorsUsed;
    }

    // 5. Downscaling
    if (scale > 1) {
        logger.log(`Downscaling by ${scale}x using '${downscaleMethod}' method.`);
        if (downscaleMethod === 'dominant') {
            current = downscaleByDominantColor(current, scale, domMeanThreshold);
        } else if (downscaleMethod === 'content-adaptive') {
            const targetW = Math.floor(originalSize[0] / scale);
            const targetH = Math.floor(originalSize[1] / scale);
            current = await contentAdaptiveDownscale(originalForAdaptiveScale, targetW, targetH);
            quantizeAfter = true;
        } else if (['median', 'mode', 'mean', 'nearest'].includes(downscaleMethod)) {
            current = downscaleBlock(current, scale, scale, downscaleMethod, domMeanThreshold);
            quantizeAfter = true;
        } else {
            logger.warn(`Unknown downscale method '${downscaleMethod}', falling back to 'median'.`);
            current = downscaleBlock(current, scale, scale, 'median', domMeanThreshold);
            quantizeAfter = true;
        }
        current = finalizePixels(current);
    }

    // 6. Optional post-downscale quantization
    if (quantizeAfter && maxColors < 256) {
        logger.log('Post-downscale quantization...');
        const quantResult = quantizeImage(current, maxColors, fixedPalette);
        current = quantResult.quantized;
        colorsUsed = quantResult.colorsUsed;
    }

    // 6. Optional Post-Downscale Cleanup
    if (cleanup.jaggy) {
        current = jaggyCleaner(current);
    }

    if (!current?.width || !current?.height) {
        throw new Error("Processing resulted in an empty image.");
    }

    // 7. Encode to PNG
    const png = await encodePng(current);
    const palette = getPaletteFromImage(current);

    logger.log(`Processing complete in ${Math.round(performance.now() - t0)}ms. Final size: ${png.byteLength} bytes.`);

    // Generate processing manifest
    const manifest = {
        original_size: originalSize,
        final_size: [current.width, current.height],
        processing_steps: {
            scale_detection: {
                method: detectMethod,
                detected_scale: scale,
                manual_scale: manualScale
            },
            color_quantization: {
                max_colors: maxColors,
                initial_colors: initialColors,
                final_colors: colorsUsed,
                fixed_palette: fixedPalette ? fixedPalette.length : null
            },
            downscaling: {
                method: downscaleMethod,
                scale_factor: scale,
                dom_mean_threshold: domMeanThreshold,
                applied: scale > 1
            },
            cleanup: {
                morphological: cleanup.morph,
                jaggy: cleanup.jaggy
            },
            alpha_processing: {
                threshold: alphaThreshold,
                binarized: alphaThreshold !== null
            },
            grid_snapping: {
                enabled: snapGrid,
                applied: snapGrid && scale > 1
            }
        },
        processing_time_ms: Math.round(performance.now() - t0),
        timestamp: new Date().toISOString()
    };

    logger.log('Final manifest:', manifest);

    return {
        png,
        imageData: current,
        palette,
        manifest,
    };
}