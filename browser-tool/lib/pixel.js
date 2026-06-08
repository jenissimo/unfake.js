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
    dominantOrMean,
    finalizePixels,
    encodePng,
    getPaletteFromImage,
    downscaleBlock,
    quantizeImage,
    detectOptimalColorCount,
} from './utils.js';
import { runOp } from './opencv-client.js';

/**
 * Upper bound for the "Max Colors" control. Selecting this value means "no
 * color reduction" — quantization is skipped entirely. Any lower value caps the
 * palette to that many colors. Kept here (and re-exported via index.js) so the
 * UI slider and the pipeline guards below can never drift apart.
 */
export const MAX_COLORS = 1024;

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
 * Edge-aware scale detection (OpenCV Sobel, single ROI).
 * @param {ImageData} imgData The input image data.
 * @returns {Promise<number>} The detected scale factor.
 */
export async function legacyEdgeAwareDetect(imgData) {
    return runOp('legacyEdgeDetect', { image: imgData });
}

/**
 * Detects pixel art scale using a robust, tiled edge-aware algorithm (OpenCV).
 * @param {ImageData} imgData The input image data.
 * @returns {Promise<number>} The detected scale factor.
 */
export async function edgeAwareDetect(imgData) {
    if (imgData.width * imgData.height > 8_000_000) {
        logger.warn('Image > 8MP, using runs-based detection for performance.');
        return runsBasedDetect(imgData);
    }

    try {
        const bestScale = await runOp('tiledEdgeDetect', { image: imgData });
        logger.log(`Edge-aware detection complete. Best guess: ${bestScale}`);
        return bestScale;
    } catch (error) {
        logger.error('Edge-aware detection failed critically:', error);
        return 1;
    }
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
export function downscaleByDominantColor(imgData, scale, threshold = 0.15) {
    const targetW = Math.floor(imgData.width / scale);
    const targetH = Math.floor(imgData.height / scale);
    const outData = new Uint8ClampedArray(targetW * targetH * 4);
    const srcData = imgData.data;
    const srcWidth = imgData.width;

    for (let ty = 0; ty < targetH; ty++) {
        for (let tx = 0; tx < targetW; tx++) {
            const colorCounts = new Map();
            const alphaValues = [];
            const opaqueColorsForMean = [];

            // Scan scale x scale block once
            for (let dy = 0; dy < scale; dy++) {
                for (let dx = 0; dx < scale; dx++) {
                    const sx = tx * scale + dx;
                    const sy = ty * scale + dy;
                    if (sx >= srcWidth || sy >= imgData.height) continue;
                    const idx = (sy * srcWidth + sx) * 4;

                    const a = srcData[idx + 3];
                    alphaValues.push(a);

                    // Only consider opaque pixels for color determination
                    if (a > 128) {
                        const r = srcData[idx];
                        const g = srcData[idx + 1];
                        const b = srcData[idx + 2];
                        const colorInt = (r << 16) | (g << 8) | b;
                        colorCounts.set(colorInt, (colorCounts.get(colorInt) || 0) + 1);
                        opaqueColorsForMean.push(colorInt);
                    }
                }
            }

            const outIdx = (ty * targetW + tx) * 4;

            if (opaqueColorsForMean.length > 0) {
                let dominantColor = 0;
                let maxCount = 0;

                // Find the dominant color from our counts
                for (const [color, count] of colorCounts.entries()) {
                    if (count > maxCount) {
                        maxCount = count;
                        dominantColor = color;
                    }
                }

                let finalColor;
                // Use dominant color only if its frequency is above the threshold
                if (maxCount / opaqueColorsForMean.length >= threshold) {
                    finalColor = dominantColor;
                } else {
                    // Otherwise, fall back to the mean color of the block
                    const rSum = opaqueColorsForMean.reduce((sum, color) => sum + ((color >> 16) & 0xFF), 0);
                    const gSum = opaqueColorsForMean.reduce((sum, color) => sum + ((color >> 8) & 0xFF), 0);
                    const bSum = opaqueColorsForMean.reduce((sum, color) => sum + (color & 0xFF), 0);
                    const avgR = Math.round(rSum / opaqueColorsForMean.length);
                    const avgG = Math.round(gSum / opaqueColorsForMean.length);
                    const avgB = Math.round(bSum / opaqueColorsForMean.length);
                    finalColor = (avgR << 16) | (avgG << 8) | avgB;
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
 * EXPERIMENTAL: Content-adaptive downscaling.
 * High-quality but computationally expensive. Separates Alpha channel for proper handling.
 * based on https://johanneskopf.de/publications/downscaling/
 * @param {ImageData} imgData - Source image.
 * @param {number} targetW - Target width.
 * @param {number} targetH - Target height.
 * @returns {Promise<ImageData>} Downscaled image.
 */
export async function contentAdaptiveDownscale(imgData, targetW, targetH) {
    logger.warn('Using experimental, CPU-intensive content-adaptive downscaling.');
    return runOp('contentAdaptiveDownscale', { image: imgData, targetW, targetH });
}

/** Main image processing pipeline. */
export async function processImage({
    file,
    maxColors = 32,
    autoColorCount = false,
    manualScale = null,
    detectMethod = 'auto', // 'auto', 'runs', 'edge'
    edgeDetectMethod = 'tiled', // 'tiled' or 'legacy'
    downscaleMethod = 'dominant', // 'dominant', 'median', 'mode', 'mean', 'nearest', 'content-adaptive'
    domMeanThreshold = 0.15, // CHANGED: Increased from 0.05 for more robust dominant color selection.
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
            'edge': edgeDetectMethod === 'legacy' ? legacyEdgeAwareDetect : edgeAwareDetect,
            'auto': async (img) => {
                const runsScale = runsBasedDetect(img);
                if (runsScale > 1) {
                    logger.log('Auto-detect: "runs" method was successful.');
                    return runsScale;
                }
                logger.log('Auto-detect: "runs" failed, falling back to "edge".');
                const edgeFn = edgeDetectMethod === 'legacy' ? legacyEdgeAwareDetect : edgeAwareDetect;
                return await edgeFn(img);
            }
        }[detectMethod];

        scale = await detectionFn(originalForAdaptiveScale);
    }
    if (scale <= 1 && downscaleMethod !== 'content-adaptive') {
        logger.log('Scale is 1, skipping downscale and grid snapping.');
    }

    // 2.5. Snap to Grid (OpenCV Sobel crop alignment)
    if (snapGrid && scale > 1) {
        logger.log('Snapping to grid via auto-crop...');
        const snapped = await runOp('snapToGrid', { image: current, scale });
        if (snapped.width !== current.width || snapped.height !== current.height) {
            current = snapped;
            logger.log(`Image cropped to: ${current.width} x ${current.height}`);
        } else {
            logger.warn('Snapping skipped: resulting image would be too small.');
        }
    }

    // 3. Optional Pre-Processing Cleanup
    if (cleanup.morph) {
        current = await morphologicalCleanup(current);
    }

    // 4. Color Quantization (before downscaling)
    const initialColors = countColors(current);
    let colorsUsed = initialColors;
    let quantizeAfter = false;

    // Auto-detect optimal color count if enabled
    let effectiveMaxColors = maxColors;
    if (autoColorCount && initialColors > 2) {
        logger.log('Auto-detecting optimal color count...');
        try {
            effectiveMaxColors = await detectOptimalColorCount(current, {
                downsampleTo: 64,
                colorQuantizeFactor: 48,
                dominanceThreshold: 0.015,
                maxColors: Math.min(maxColors, 32) // Cap at 32 for auto-detection
            });
            logger.log(`Auto-detected optimal color count: ${effectiveMaxColors}`);
        } catch (error) {
            logger.warn('Auto-color detection failed, using manual setting:', error);
            effectiveMaxColors = maxColors;
        }
    }

    if (downscaleMethod === 'content-adaptive') {
        // Не квантизуем до даунскейла
    } else if (effectiveMaxColors < MAX_COLORS && initialColors > effectiveMaxColors) {
        logger.log(`Quantizing from ${initialColors} to a max of ${effectiveMaxColors} colors.`);
        const quantResult = quantizeImage(current, effectiveMaxColors, fixedPalette);
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
    if (quantizeAfter && effectiveMaxColors < MAX_COLORS) {
        logger.log('Post-downscale quantization...');
        const quantResult = quantizeImage(current, effectiveMaxColors, fixedPalette);
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
                edge_method: manualScale ? null : detectMethod === 'edge' || detectMethod === 'auto' ? edgeDetectMethod : null,
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