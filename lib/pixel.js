// pixel.js – A module for advanced pixel art optimization.
// Handles intelligent scale detection, color quantization, downscaling, and PNG encoding.

import * as IQ from 'image-q';
import { cvReady, fileToImageData, morphologicalCleanup, countColors, detectScale, median, mode, mean, logger } from './utils.js';
import SVD from 'svd';

// Get UPNG from global scope (loaded via script tag)
const { UPNG } = window;

/**
 * Ensures OpenCV resources (cv.Mat) are properly released, even if errors occur.
 * @param {Function} fn - The function to execute, receiving (cv, track).
 * @returns {Promise<any>} The result of the wrapped function.
 */
async function withCv(fn) {
    const mats = new Set();
    const cv = await cvReady();
    // The track function adds a Mat to the cleanup set and returns it.
    const track = (mat) => {
        if (mat && typeof mat.delete === 'function') mats.add(mat);
        return mat;
    };

    try {
        return await fn(cv, track);
    } finally {
        for (const mat of mats) {
            if (mat && !mat.isDeleted()) {
                mat.delete();
            }
        }
    }
}


/**
 * Detects the pixel art scale factor using an edge-aware algorithm with OpenCV.
 * @param {ImageData} imgData - The input image data.
 * @param {Array<number>} manualScale - Optional manual scale [hScale, vScale].
 * @returns {Promise<Object>} An object with detected scale information.
 */
async function edgeAwareDetect(imgData, manualScale) {
    if (manualScale) {
        logger.log('Using manual scale:', manualScale);
        const [hScale, vScale] = manualScale;
        return { hScale, vScale, baseScale: Math.max(hScale, vScale) };
    }

    return withCv(async (cv, track) => {
        const gray = track(new cv.Mat());
        cv.cvtColor(track(cv.matFromImageData(imgData)), gray, cv.COLOR_RGBA2GRAY);

        const edges = track(new cv.Mat());
        cv.Canny(gray, edges, 50, 150);

        const contours = track(new cv.MatVector());
        const hierarchy = track(new cv.Mat());
        cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let roiRect;
        if (contours.size() > 0) {
            let maxArea = 0;
            let largestContourIndex = -1;
            for (let i = 0; i < contours.size(); i++) {
                const area = cv.contourArea(contours.get(i));
                if (area > maxArea) {
                    maxArea = area;
                    largestContourIndex = i;
                }
            }

            const largestContour = contours.get(largestContourIndex);
            const rect = cv.boundingRect(largestContour);
            const pad = Math.max(10, Math.min(30, Math.min(rect.width, rect.height) * 0.1));

            roiRect = new cv.Rect(
                Math.max(0, rect.x - pad),
                Math.max(0, rect.y - pad),
                Math.min(imgData.width - (rect.x - pad), rect.width + 2 * pad),
                Math.min(imgData.height - (rect.y - pad), rect.height + 2 * pad)
            );
        } else {
            const border = Math.min(20, Math.min(imgData.width, imgData.height) * 0.05);
            roiRect = new cv.Rect(border, border, imgData.width - 2 * border, imgData.height - 2 * border);
            logger.warn('No contours found, using image center for scale detection.');
        }

        const roiMat = track(gray.roi(roiRect));
        const diffH = track(new cv.Mat());
        const diffV = track(new cv.Mat());
        cv.Sobel(roiMat, diffH, cv.CV_32F, 1, 0, 3);
        cv.Sobel(roiMat, diffV, cv.CV_32F, 0, 1, 3);

        const hSumMat = track(new cv.Mat());
        const vSumMat = track(new cv.Mat());
        cv.reduce(diffH, hSumMat, 0, cv.REDUCE_SUM, cv.CV_32F);
        cv.reduce(diffV, vSumMat, 1, cv.REDUCE_SUM, cv.CV_32F);

        const hSum = hSumMat.data32F;
        const vSum = vSumMat.data32F;

        const hScale = detectScale(Array.from(hSum));
        const vScale = detectScale(Array.from(vSum));
        const safeHScale = Math.max(1, Math.min(hScale, 100));
        const safeVScale = Math.max(1, Math.min(vScale, 100));

        let bestScale = Math.round((safeHScale + safeVScale) / 2) || 1;
        let maxFitScore = -1;

        const candidates = new Set();
        const range = Math.max(2, Math.floor(Math.min(safeHScale, safeVScale) / 3));
        for (let s = Math.max(2, safeHScale - range); s <= safeHScale + range; s++) candidates.add(s);
        for (let s = Math.max(2, safeVScale - range); s <= safeVScale + range; s++) candidates.add(s);
        if (candidates.size === 0) [2, 3, 4].forEach(s => candidates.add(s));

        for (const s of candidates) {
            if (s <= 1) continue;
            const fitScore = (1 - (roiRect.width % s) / s) + (1 - (roiRect.height % s) / s);
            if (fitScore > maxFitScore) {
                maxFitScore = fitScore;
                bestScale = s;
            }
        }

        logger.log('Edge-aware detected scale:', bestScale);
        return { hScale: bestScale, vScale: bestScale, baseScale: bestScale };
    });
}

/**
 * Quantizes image colors using the image-q library.
 * It tries perceptually superior color spaces first (CIEDE2000) and falls back gracefully.
 * @param {ImageData} imgData - The input image data.
 * @param {number} maxColors - The maximum number of colors for the palette.
 * @returns {{quantized: ImageData, colorsUsed: number}}
 */
function quantizeImage(imgData, maxColors) {
    if (typeof IQ === 'undefined') {
        logger.warn('image-q library not found, quantization skipped.');
        return { quantized: imgData, colorsUsed: countColors(imgData) };
    }

    try {
        const inPointContainer = IQ.utils.PointContainer.fromImageData(imgData);

        const strategies = [
            { colorDistanceFormula: 'ciede2000', paletteQuantization: 'neuquant' },
            { colorDistanceFormula: 'cie94-graphic-arts', paletteQuantization: 'wuquant' },
            { colorDistanceFormula: 'euclidean', paletteQuantization: 'wuquant' }
        ];

        let chosenStrategy;
        for (const strategy of strategies) {
            try {
                IQ.buildPaletteSync([inPointContainer], { ...strategy, colors: 2 });
                chosenStrategy = strategy;
                logger.log('Using quantization strategy:', chosenStrategy);
                break;
            } catch (e) { /* Strategy not supported, try next one */ }
        }
        if (!chosenStrategy) throw new Error("No compatible image-q strategy found.");

        const palette = IQ.buildPaletteSync([inPointContainer], { ...chosenStrategy, colors: maxColors });
        const outPointContainer = IQ.applyPaletteSync(inPointContainer, palette, {
            colorDistanceFormula: chosenStrategy.colorDistanceFormula,
            imageQuantization: 'nearest'
        });

        const quantized = new ImageData(new Uint8ClampedArray(outPointContainer.toUint8Array()), imgData.width, imgData.height);

        const data = quantized.data;
        for (let i = 3; i < data.length; i += 4) data[i] = data[i] < 128 ? 0 : 255;

        return { quantized, colorsUsed: palette.getPointContainer().getPointArray().length };
    } catch (error) {
        logger.error('image-q quantization failed, returning original image.', error);
        throw new Error(`Quantization failed: ${error.message}`);
    }
}

/**
 * Downscales an image by sampling pixels within a block.
 * @param {ImageData} imgData - The source image data.
 * @param {number} hScale - Horizontal scale factor.
 * @param {number} vScale - Vertical scale factor.
 * @param {string} [method='median'] - The sampling method ('median', 'mode', 'mean', etc.).
 * @returns {ImageData} The downscaled image data.
 */
function downscaleBlock(imgData, hScale, vScale, method = 'median') {
    const targetW = Math.floor(imgData.width / hScale);
    const targetH = Math.floor(imgData.height / vScale);
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
                    if (d[idx + 3] === 255) { // Only consider opaque pixels for color
                        colorsR.push(d[idx]);
                        colorsG.push(d[idx + 1]);
                        colorsB.push(d[idx + 2]);
                    }
                    colorsA.push(d[idx + 3]); // Always consider alpha for majority alpha
                }
            }

            if (colorsA.length === 0) continue;

            const offset = (ty * targetW + tx) * 4;
            const hasColor = colorsR.length > 0;

            const aggregator = { 'mode': mode, 'mean': mean }[method] || median;
            out[offset] = hasColor ? aggregator(colorsR) : 0;
            out[offset + 1] = hasColor ? aggregator(colorsG) : 0;
            out[offset + 2] = hasColor ? aggregator(colorsB) : 0;
            out[offset + 3] = median(colorsA); // Median is best for alpha to preserve hard edges
        }
    }
    return new ImageData(out, targetW, targetH);
}

// =================================================================================
// START: Modified Content-Adaptive Downscaling Algorithm
// Based on "Content-Adaptive Image Downscaling" by Kopf, Shamir, Peers (SIGGRAPH Asia 2013)
// This version is refactored to handle RGB and Alpha channels separately.
// =================================================================================

/**
 * Helper function to multiply two 2x2 matrices (a * b).
 * @param {Array<Array<number>>} a - First matrix.
 * @param {Array<Array<number>>} b - Second matrix.
 * @returns {Array<Array<number>>} The resulting matrix.
 */
function multiply2x2(a, b) {
    const a00 = a[0][0], a01 = a[0][1], a10 = a[1][0], a11 = a[1][1];
    const b00 = b[0][0], b01 = b[0][1], b10 = b[1][0], b11 = b[1][1];
    return [
        [a00 * b00 + a01 * b10, a00 * b01 + a01 * b11],
        [a10 * b00 + a11 * b10, a10 * b01 + a11 * b11]
    ];
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
 * Downscales an image using a content-adaptive iterative method, now with proper alpha channel support.
 * @param {ImageData} imgData - The source image data (must be RGBA).
 * @param {number} targetW - The target width.
 * @param {number} targetH - The target height.
 * @returns {Promise<ImageData>} The downscaled image data.
 */
async function contentAdaptiveDownscale(imgData, targetW, targetH) {
    if (typeof SVD === 'undefined' || typeof SVD !== 'function') {
        throw new Error("Content-adaptive downscaling requires the svd-js library, but it was not found.");
    }

    logger.log(`Starting content-adaptive downscaling to ${targetW}x${targetH} with alpha support...`);

    return withCv(async (cv, track) => {
        const { width: wi, height: hi } = imgData;
        const wo = targetW, ho = targetH;

        // 1. Load source image and split into RGBA channels
        const srcMat = track(cv.matFromImageData(imgData));
        let channels = track(new cv.MatVector());
        cv.split(srcMat, channels);
        const alpha_channel = track(channels.get(3));

        // Create a 3-channel RGB Mat
        let rgb_channels = track(new cv.MatVector());
        rgb_channels.push_back(channels.get(0));
        rgb_channels.push_back(channels.get(1));
        rgb_channels.push_back(channels.get(2));
        const srcRGB = track(new cv.Mat());
        cv.merge(rgb_channels, srcRGB);

        // 2. Process RGB channels with the content-adaptive algorithm
        // Convert RGB to 32-bit float and then to Lab
        const srcRGB32f = track(new cv.Mat());
        srcRGB.convertTo(srcRGB32f, cv.CV_32F, 1.0 / 255.0);
        const srcLab = track(new cv.Mat());
        cv.cvtColor(srcRGB32f, srcLab, cv.COLOR_RGB2Lab); // Use RGB2Lab now

        // Run the core algorithm on Lab data
        const outLab = _contentAdaptiveCore(srcLab, wo, ho, cv, track);

        // Convert the result back to RGB
        const outRGB_32f = track(new cv.Mat());
        cv.cvtColor(outLab, outRGB_32f, cv.COLOR_Lab2RGB);
        const outRGB_8u = track(new cv.Mat());
        outRGB_32f.convertTo(outRGB_8u, cv.CV_8U, 255.0);

        // 3. Downscale the alpha channel separately using a high-quality standard method
        const outAlpha = track(new cv.Mat());
        const dsize = new cv.Size(wo, ho);
        // cv.INTER_AREA is the best interpolation for downscaling (image shrinking)
        cv.resize(alpha_channel, outAlpha, dsize, 0, 0, cv.INTER_AREA);

        // 4. Merge the downscaled RGB and downscaled Alpha
        let out_rgba_channels = track(new cv.MatVector());
        let out_rgb_split = track(new cv.MatVector());
        cv.split(outRGB_8u, out_rgb_split);
        out_rgba_channels.push_back(out_rgb_split.get(0));
        out_rgba_channels.push_back(out_rgb_split.get(1));
        out_rgba_channels.push_back(out_rgb_split.get(2));
        out_rgba_channels.push_back(outAlpha); // Add the resized alpha channel

        const finalRgbaMat = track(new cv.Mat());
        cv.merge(out_rgba_channels, finalRgbaMat);

        // 5. Create final ImageData
        return new ImageData(new Uint8ClampedArray(finalRgbaMat.data), wo, ho);
    });
}

/**
 * Encodes ImageData to a PNG buffer using UPNG.js with a canvas fallback.
 * @param {ImageData} imgData - The image data to encode.
 * @returns {Promise<Uint8Array>} A Uint8Array containing the PNG file data.
 */
async function encodePng(imgData) {
    if (UPNG && typeof UPNG.encode === 'function') {
        const pngBuf = UPNG.encode([imgData.data.buffer], imgData.width, imgData.height, 0);
        return new Uint8Array(pngBuf);
    }

    logger.warn('UPNG not available or failed, using canvas fallback for PNG encoding.');
    const canvas = document.createElement('canvas');
    canvas.width = imgData.width;
    canvas.height = imgData.height;
    canvas.getContext('2d').putImageData(imgData, 0, 0);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('Canvas toBlob failed to create PNG.');
    return new Uint8Array(await blob.arrayBuffer());
}

/**
 * Extracts the unique colors from an ImageData object.
 * @param {ImageData} imgData The image data.
 * @returns {string[]} An array of hex color strings (e.g., "#rrggbbaa").
 */
function getPaletteFromImage(imgData) {
    const seen = new Set();
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
        seen.add((d[i] << 24) | (d[i + 1] << 16) | (d[i + 2] << 8) | d[i + 3]);
    }

    return Array.from(seen).map(key => {
        const r = (key >>> 24).toString(16).padStart(2, '0');
        const g = ((key >>> 16) & 255).toString(16).padStart(2, '0');
        const b = ((key >>> 8) & 255).toString(16).padStart(2, '0');
        const a = (key & 255).toString(16).padStart(2, '0');
        return `#${r}${g}${b}${a}`;
    });
}

/**
 * Main pixel art processing pipeline.
 */
export async function processImage({
    file,
    maxColors = 32,
    snapGrid = true,
    manualScale,
    downscaleMethod = 'median', // 'median', 'mode', 'mean', or 'content-adaptive'
    cleanup = false,
}) {
    const t0 = performance.now();
    let current;

    try {
        current = await fileToImageData(file);
        logger.log(`Image loaded: ${current.width}x${current.height}`);

        if (current.width > 8000 || current.height > 8000) {
            throw new Error(`Image is too large (${current.width}x${current.height}). Max 8000px.`);
        }

        const originalImageData = current;
        const originalSize = [current.width, current.height];

        // 1. Scale Detection
        const { hScale, vScale, baseScale } = await edgeAwareDetect(current, manualScale);

        if (baseScale <= 1 && downscaleMethod !== 'content-adaptive') { // Allow content-adaptive on 1x for general purpose resize
            logger.warn('Detected scale is 1 or less. Processing is likely not needed for block methods.');
        }

        // 2. (Optional) Cleanup on a copy of the original scaled-up image
        if (cleanup) {
            logger.log('Applying morphological cleanup...');
            current = await morphologicalCleanup(current); // This modifies `current`
        }

        // 3. Downscaling (only if scale > 1 or method is content-adaptive)
        if (baseScale > 1 || downscaleMethod === 'content-adaptive') {
            logger.log(`Downscaling by ${hScale}x${vScale} using '${downscaleMethod}' method.`);

            if (downscaleMethod === 'content-adaptive') {
                const targetW = Math.floor(originalSize[0] / hScale);
                const targetH = Math.floor(originalSize[1] / vScale);
                // The adaptive method should run on the original, not the cleaned up version, for best results
                current = await contentAdaptiveDownscale(originalImageData, targetW, targetH);
            } else {
                current = downscaleBlock(current, hScale, vScale, downscaleMethod);
            }
        }

        // 4. Color Quantization
        let colorsUsed = countColors(current);
        if (maxColors < 256 && colorsUsed > maxColors) {
            logger.log(`Quantizing from ${colorsUsed} to ${maxColors} colors.`);
            const quantResult = quantizeImage(current, maxColors);
            current = quantResult.quantized;
            colorsUsed = quantResult.colorsUsed;
        } else {
            logger.log(`Quantization skipped. Colors used (${colorsUsed}) is within limit (${maxColors}).`);
        }

        if (!current || !current.width || !current.height) {
            throw new Error("Processing resulted in an empty image.");
        }

        // 5. PNG Encoding
        const png = await encodePng(current);

        // 6. Generate Palette & Manifest
        const palette = getPaletteFromImage(current);

        const manifest = {
            original_size: originalSize,
            detected_scale: { horizontal: hScale, vertical: vScale, base: baseScale },
            final_size: [current.width, current.height],
            processing_steps: { snapGrid, cleanup, downscale_method: downscaleMethod, final_max_colors: maxColors },
            colors_used: colorsUsed,
            processing_time_ms: Math.round(performance.now() - t0),
            timestamp: new Date().toISOString()
        };

        logger.log(`Processing complete. Final size: ${png.byteLength} bytes.`);
        return { png, manifest, palette, imageData: current };

    } catch (error) {
        logger.error('Pixel processing pipeline failed:', error);
        throw error;
    }
}