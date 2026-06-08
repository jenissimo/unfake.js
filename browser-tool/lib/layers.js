/**
 * @fileoverview Layer extraction for character customization pipeline.
 * Splits an image into two layers:
 * 1. Mask Layer - A solid white silhouette defining the shape
 * 2. Details Layer - Outlines and shadows with brightness-based transparency
 */

import { logger, fileToImageData, encodePng } from './utils.js';

/**
 * Extracts the Mask Layer from an image.
 * Creates a solid white silhouette where any opaque pixel becomes white.
 * @param {ImageData} imgData - The source image data.
 * @param {object} [options] - Extraction options.
 * @param {number} [options.alphaThreshold=1] - Minimum alpha value to consider a pixel visible (1-255).
 * @returns {ImageData} The mask layer image data (white silhouette on transparent).
 */
export function extractMaskLayer(imgData, options = {}) {
    const { alphaThreshold = 1 } = options;
    const { data, width, height } = imgData;
    const out = new Uint8ClampedArray(data.length);

    for (let i = 0; i < data.length; i += 4) {
        const alpha = data[i + 3];
        
        if (alpha >= alphaThreshold) {
            // Visible pixel becomes solid white
            out[i] = 255;     // R
            out[i + 1] = 255; // G
            out[i + 2] = 255; // B
            out[i + 3] = 255; // A (fully opaque)
        } else {
            // Transparent stays transparent
            out[i] = 0;
            out[i + 1] = 0;
            out[i + 2] = 0;
            out[i + 3] = 0;
        }
    }

    return new ImageData(out, width, height);
}

/**
 * Extracts the Details Layer from an image.
 * Converts brightness to transparency: black stays opaque, white becomes invisible.
 * The result is a "ghost" image with black outlines/shadows on transparent background.
 * 
 * Formula: Alpha = 255 - Brightness (where Brightness = (R + G + B) / 3)
 * - Black (0,0,0) → Alpha = 255 (fully opaque)
 * - Gray (128,128,128) → Alpha = 127 (semi-transparent)
 * - White (255,255,255) → Alpha = 0 (invisible)
 * 
 * @param {ImageData} imgData - The source image data.
 * @param {object} [options] - Extraction options.
 * @param {string} [options.brightnessMethod='average'] - Method to calculate brightness: 'average' or 'luminance'.
 * @param {number} [options.detailsBoost=1.0] - Multiplier for details intensity (0.5-2.0). Higher = more visible details.
 * @returns {ImageData} The details layer image data (black with variable transparency).
 */
export function extractDetailsLayer(imgData, options = {}) {
    const { 
        brightnessMethod = 'average',
        detailsBoost = 1.0 
    } = options;
    
    const { data, width, height } = imgData;
    const out = new Uint8ClampedArray(data.length);

    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const alpha = data[i + 3];

        if (alpha === 0) {
            // Already transparent, keep it that way
            out[i] = 0;
            out[i + 1] = 0;
            out[i + 2] = 0;
            out[i + 3] = 0;
        } else {
            // Calculate brightness based on method
            let brightness;
            if (brightnessMethod === 'luminance') {
                // Proper luminance calculation (ITU-R BT.601)
                brightness = 0.299 * r + 0.587 * g + 0.114 * b;
            } else {
                // Simple average
                brightness = (r + g + b) / 3;
            }
            
            // Invert brightness to get alpha: darker = more opaque
            let newAlpha = 255 - brightness;
            
            // Apply details boost (clamped to 0-255)
            newAlpha = Math.min(255, Math.max(0, Math.round(newAlpha * detailsBoost)));
            
            // Also consider original alpha
            const finalAlpha = Math.round((newAlpha * alpha) / 255);

            // Output is pure black with variable transparency
            out[i] = 0;     // R
            out[i + 1] = 0; // G
            out[i + 2] = 0; // B
            out[i + 3] = finalAlpha;
        }
    }

    return new ImageData(out, width, height);
}

/**
 * Main layer extraction pipeline.
 * Splits an input image into mask and details layers for character customization.
 * @param {object} options - Processing options.
 * @param {File} [options.file] - The input image file.
 * @param {ImageData} [options.imageData] - Direct ImageData input (alternative to file, used when chaining with pixel art mode).
 * @param {number} [options.alphaThreshold=1] - Minimum alpha value to consider a pixel visible (1-255).
 * @param {string} [options.brightnessMethod='average'] - Method to calculate brightness: 'average' or 'luminance'.
 * @param {number} [options.detailsBoost=1.0] - Multiplier for details intensity (0.5-2.0).
 * @returns {Promise<{mask: {png: Uint8Array, imageData: ImageData}, details: {png: Uint8Array, imageData: ImageData}, manifest: object}>}
 */
export async function extractLayers({ 
    file, 
    imageData: inputImageData,
    alphaThreshold = 1,
    brightnessMethod = 'average',
    detailsBoost = 1.0
}) {
    if (!file && !inputImageData) throw new Error('No file or imageData provided.');

    const t0 = performance.now();
    logger.log('Starting layer extraction...');

    // 1. Load image (from file or use provided ImageData)
    let imgData;
    if (inputImageData) {
        imgData = inputImageData;
        logger.log(`Using provided ImageData: ${imgData.width}x${imgData.height}`);
    } else {
        imgData = await fileToImageData(file);
        logger.log(`Loaded image: ${imgData.width}x${imgData.height}`);
    }
    const originalSize = [imgData.width, imgData.height];

    // 2. Extract mask layer
    logger.log('Extracting mask layer...');
    const maskData = extractMaskLayer(imgData, { alphaThreshold });
    const maskPng = await encodePng(maskData);
    logger.log(`Mask layer created: ${maskPng.byteLength} bytes`);

    // 3. Extract details layer
    logger.log('Extracting details layer...');
    const detailsData = extractDetailsLayer(imgData, { brightnessMethod, detailsBoost });
    const detailsPng = await encodePng(detailsData);
    logger.log(`Details layer created: ${detailsPng.byteLength} bytes`);

    const t1 = performance.now();
    
    // 4. Build manifest
    const manifest = {
        original_size: originalSize,
        processing_time_ms: Math.round(t1 - t0),
        timestamp: new Date().toISOString(),
        options: {
            alphaThreshold,
            brightnessMethod,
            detailsBoost,
            source: inputImageData ? 'imageData' : 'file'
        },
        layers: {
            mask: {
                description: 'White silhouette on transparent background',
                size_bytes: maskPng.byteLength,
            },
            details: {
                description: 'Black outlines/shadows with brightness-to-alpha conversion',
                size_bytes: detailsPng.byteLength,
            }
        }
    };

    logger.log(`Layer extraction complete in ${Math.round(t1 - t0)}ms`);

    return {
        mask: {
            png: maskPng,
            imageData: maskData,
        },
        details: {
            png: detailsPng,
            imageData: detailsData,
        },
        manifest,
    };
}
