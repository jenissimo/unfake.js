// vector.js – Vector processing module
// Handles image vectorization, SVG generation, and color tracing

import * as IQ from 'image-q';
import { potrace, init } from 'potrace-wasm';
import { cvReady, fileToImageData, logger } from './utils.js';

/**
 * Automatically detect optimal color count in image
 */
async function detectOptimalColorCount(imgData, { 
    downsampleTo = 64,
    colorQuantizeFactor = 48,
    dominanceThreshold = 0.015
} = {}) {
    const cv = await cvReady();
    let src = cv.matFromImageData(imgData);
    
    // 1. Downsample image for overall structure analysis
    const aspectRatio = src.rows / src.cols;
    const targetWidth = downsampleTo;
    const targetHeight = Math.round(targetWidth * aspectRatio);
    const dsize = new cv.Size(targetWidth, targetHeight);
    let smallMat = new cv.Mat();
    cv.resize(src, smallMat, dsize, 0, 0, cv.INTER_AREA);
    
    // 2. Apply blur to eliminate gradients and noise
    cv.medianBlur(smallMat, smallMat, 5);
    
    // 3. Additional Gaussian blur for smoothing
    const kernel = new cv.Size(5, 5);
    cv.GaussianBlur(smallMat, smallMat, kernel, 1, 1);

    const colorCounts = new Map();
    const totalPixels = smallMat.rows * smallMat.cols;
    const d = smallMat.data;

    // 4. Collect color statistics with aggressive quantization
    for (let i = 0; i < d.length; i += 4) {
        const alpha = d[i + 3];
        if (alpha < 200) continue; // Ignore transparent/semi-transparent pixels

        const r = Math.round(d[i] / colorQuantizeFactor) * colorQuantizeFactor;
        const g = Math.round(d[i + 1] / colorQuantizeFactor) * colorQuantizeFactor;
        const b = Math.round(d[i + 2] / colorQuantizeFactor) * colorQuantizeFactor;

        const colorKey = `${r},${g},${b}`;
        colorCounts.set(colorKey, (colorCounts.get(colorKey) || 0) + 1);
    }
    
    // 5. Analyze dominant colors
    const minPixelsForDominance = Math.max(3, Math.round(totalPixels * dominanceThreshold));
    
    console.log(`Total pixels: ${totalPixels}, min pixels for dominance: ${minPixelsForDominance}`);
    console.log(`Found ${colorCounts.size} unique quantized colors`);
    
    // Sort colors by frequency of use
    const sortedColors = Array.from(colorCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .filter(([color, count]) => count >= minPixelsForDominance);
    
    console.log('Top 10 colors by frequency:');
    sortedColors.slice(0, 10).forEach(([color, count], index) => {
        const percentage = ((count / totalPixels) * 100).toFixed(1);
        console.log(`  ${index + 1}. rgb(${color}) - ${count} pixels (${percentage}%)`);
    });
    
    // 6. Apply additional filtering
    let significantColors = sortedColors.length;
    
    if (significantColors > 32) {
        const strictThreshold = Math.max(minPixelsForDominance, Math.round(totalPixels * 0.02));
        significantColors = sortedColors.filter(([, count]) => count >= strictThreshold).length;
        console.log(`After strict filtering (2% threshold): ${significantColors} colors`);
    }
    
    src.delete();
    smallMat.delete();

    // 7. Return reasonable number of colors
    const result = Math.max(2, Math.min(significantColors, 32));
    console.log(`Final auto-detected color count: ${result}`);
    return result;
}

/**
 * Quantize image colors using image-q library or fallback
 */
function quantizeImage(imgData, max) {
  if (typeof IQ === 'undefined') {
    logger.warn('image-q library not loaded, using fallback quantization');
    return fallbackQuantize(imgData, max);
  }
  
  try {
    const inPointContainer = IQ.utils.PointContainer.fromImageData(imgData);
    
    const palette = IQ.buildPaletteSync([inPointContainer], {
      colorDistanceFormula: 'euclidean',
      paletteQuantization: 'wuquant',
      colors: max
    });
    
    const outPointContainer = IQ.applyPaletteSync(inPointContainer, palette, {
      colorDistanceFormula: 'euclidean',
      imageQuantization: 'nearest'
    });
    
    const quantizedData = outPointContainer.toUint8Array();
    
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
 * Fallback quantization without external dependencies
 */
function fallbackQuantize(imgData, maxColors) {
  const { data, width, height } = imgData;
  const colorMap = new Map();
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    
    const quantizeFactor = Math.max(1, Math.floor(256 / maxColors));
    const qr = Math.floor(r / quantizeFactor) * quantizeFactor;
    const qg = Math.floor(g / quantizeFactor) * quantizeFactor;
    const qb = Math.floor(b / quantizeFactor) * quantizeFactor;
    const qa = Math.floor(a / quantizeFactor) * quantizeFactor;
    
    const key = `${qr},${qg},${qb},${qa}`;
    colorMap.set(key, { r: qr, g: qg, b: qb, a: qa });
  }
  
  const colors = Array.from(colorMap.values());
  if (colors.length > maxColors) {
    colors.splice(maxColors);
  }
  
  const result = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    
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

/**
 * Extract unique colors from quantized ImageData
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
 * Create binary mask (black on white) for one color
 */
function createMask(imgData, targetColor) {
    console.log(`Creating mask for color: rgb(${targetColor.r}, ${targetColor.g}, ${targetColor.b})`);
    const mask = new ImageData(imgData.width, imgData.height);
    const d = imgData.data;
    const m = mask.data;
    let blackPixels = 0;
    
    for (let i = 0; i < d.length; i += 4) {
        const r = d[i], g = d[i+1], b = d[i+2], a = d[i+3];
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
 * Trace one color to SVG path
 */
async function traceColor(imgData, color, options) {
    console.log(`traceColor called for: rgb(${color.r}, ${color.g}, ${color.b})`);
    try {
        // Initialize potrace if not already done
        if (typeof potrace === 'undefined') {
            console.warn('potrace-wasm not loaded, using fallback');
            return { paths: createFallbackPath(imgData, color), viewBox: null };
        }
        
        console.log('Creating canvas for potrace...');
        const canvas = typeof OffscreenCanvas !== 'undefined' 
            ? new OffscreenCanvas(imgData.width, imgData.height) 
            : document.createElement('canvas');

        if (canvas instanceof HTMLCanvasElement) {
            canvas.width = imgData.width;
            canvas.height = imgData.height;
        }

        const ctx = canvas.getContext('2d');
        
        const mask = createMask(imgData, color);
        ctx.putImageData(mask, 0, 0);
        
        console.log('Canvas created, calling potrace-wasm...');
        const result = await potrace(canvas, {
            turdsize: options.turdsize,
            alphamax: options.alphamax,
            opticurve: options.opticurve ? 1 : 0,
        });
        
        console.log('Potrace result type:', typeof result);
        console.log('Potrace result preview:', typeof result === 'string' ? result.substring(0, 200) : result);
        
        if (typeof result === 'string' && result.includes('<path')) {
            console.log('Parsing SVG string from potrace...');
            
            const parser = new DOMParser();
            const svgDoc = parser.parseFromString(result, 'image/svg+xml');
            
            const svgElement = svgDoc.querySelector('svg');
            let viewBox = svgElement ? svgElement.getAttribute('viewBox') : null;
            const width = svgElement ? svgElement.getAttribute('width') : null;
            const height = svgElement ? svgElement.getAttribute('height') : null;
            
            console.log('SVG attributes from potrace:', { viewBox, width, height });
            
            if (!viewBox && width && height) {
                viewBox = `0 0 ${width.replace('pt', '')} ${height.replace('pt', '')}`;
                console.log('Generated viewBox from width/height:', viewBox);
            }
            
            const pathElements = svgDoc.querySelectorAll('path');
             
            if (pathElements.length > 0 && viewBox) {
                const samplePath = pathElements[0].getAttribute('d');
                const coordMatch = samplePath?.match(/M(\d+)\s+(\d+)/);
                if (coordMatch) {
                    const maxCoord = Math.max(parseInt(coordMatch[1]), parseInt(coordMatch[2]));
                    const viewBoxParts = viewBox.split(' ').map(parseFloat);
                    const viewBoxMax = Math.max(viewBoxParts[2], viewBoxParts[3]);
                    
                    console.log('Coordinate analysis:', { maxCoord, viewBoxMax, ratio: maxCoord / viewBoxMax });
                    
                    if (maxCoord > viewBoxMax * 2) {
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
                        
                        const newViewBox = `0 0 ${Math.ceil(maxX * 1.1)} ${Math.ceil(maxY * 1.1)}`;
                        console.log('Recalculated viewBox based on coordinates:', newViewBox);
                        viewBox = newViewBox;
                    }
                }
            }
             
            console.log('Final viewBox for this color:', viewBox);
            console.log('Found path elements:', pathElements.length);
            
            if (pathElements.length > 0) {
                const pathStrings = [];
                pathElements.forEach(pathEl => {
                    const d = pathEl.getAttribute('d');
                    if (d && d.trim().length > 0) {
                        pathStrings.push(d.trim());
                    }
                });
                
                console.log('Valid paths found:', pathStrings.length);
                
                if (pathStrings.length > 0) {
                    const validPathStrings = pathStrings.filter(d => {
                        const startsWithGiantSquare = d.startsWith('M0 5120') || 
                                                     d.startsWith('M0 2560') || 
                                                     d.startsWith('M0 1280') ||
                                                     d.includes('l0 -5120') ||
                                                     d.includes('l0 -2560') ||
                                                     d.includes('l0 -1280');
                        
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
                        const rgbaColor = color.a !== undefined && color.a !== 255 
                            ? `rgba(${color.r}, ${color.g}, ${color.b}, ${(color.a / 255).toFixed(3)})` 
                            : `rgb(${color.r}, ${color.g}, ${color.b})`;
                        
                        console.log(`Using color: ${rgbaColor} for path generation`);
                        
                        const strokeColor = rgbaColor;
                        const strokeWidth = Math.max(1, Math.ceil(Math.min(imgData.width, imgData.height) / 200));
                        
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
        return { paths: createFallbackPath(imgData, color), viewBox: null };
    }
}

/**
 * Create simple fallback path for color (minimal rasterization)
 */
function createFallbackPath(imgData, color) {
    logger.warn(`Creating fallback path for color: rgb(${color.r}, ${color.g}, ${color.b})`);
    
    const paths = [];
    const { data, width, height } = imgData;
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const r = data[idx], g = data[idx + 1], b = data[idx + 2], a = data[idx + 3];
            
            if (r === color.r && g === color.g && b === color.b && a === color.a) {
                paths.push(`M${x},${y}h1v1h-1z`);
            }
        }
    }
    
    if (paths.length === 0) {
        logger.warn('No pixels found for color, returning empty path');
        return '';
    }
    
    const rgbaColor = color.a !== undefined && color.a !== 255 
        ? `rgba(${color.r}, ${color.g}, ${color.b}, ${(color.a / 255).toFixed(3)})` 
        : `rgb(${color.r}, ${color.g}, ${color.b})`;
        
    const strokeColor = rgbaColor;
    const strokeWidth = Math.max(1, Math.ceil(Math.min(width, height) / 200));
        
    console.log(`Creating fallback path with color: ${rgbaColor}, stroke-width: ${strokeWidth}`);
    const fallbackPath = `<path d="${paths.join('')}" fill="${rgbaColor}" stroke="${strokeColor}" stroke-width="${strokeWidth}" stroke-linejoin="round" stroke-linecap="round" shape-rendering="geometricPrecision" />`;
    logger.log(`Fallback path created with ${paths.length} pixel rectangles`);
    return fallbackPath;
}

/**
 * Main vectorization function
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
    morphology: true,
    morphologyKernel: 5
  }
}) {
    logger.log('vectorizeImage called with:', { file: file.name, numColors, preProcess });
    const t0 = performance.now();

    // Initialize potrace-wasm
    try {
        if (typeof init === 'function') {
            await init();
            console.log('potrace-wasm initialized successfully');
        }
    } catch (error) {
        console.warn('Failed to initialize potrace-wasm:', error);
    }

    const cv = await cvReady();
    const imgData = await fileToImageData(file);
    const originalSize = [imgData.width, imgData.height];
    
    // Determine number of colors
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

    // --- PRE-PROCESSING ---
    if (preProcess.enabled) {
        console.log(`Applying pre-processing filter: ${preProcess.filter}`);
        const processedMat = new cv.Mat();
        if (preProcess.filter === 'bilateral') {
            const d = preProcess.value;
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

    const preProcessedImageData = new ImageData(
        new Uint8ClampedArray(src.data),
        src.cols,
        src.rows
    );
    src.delete();

    // Color quantization
    console.log(`Quantizing image to ${finalNumColors} colors...`);
    const { quantized, colorsUsed } = quantizeImage(preProcessedImageData, finalNumColors);
    console.log(`Quantization complete. Colors used: ${colorsUsed}`);

    // Tracing
    console.log('Tracing color layers...');
    const { data, width, height } = quantized;

    let palette = getPaletteFromQuantized(quantized);
    console.log('Palette colors found:', palette.length);
    console.log('Palette preview:', palette.slice(0, 5));
    
    console.log('Full palette details:');
    palette.forEach((color, index) => {
        const rgbaString = color.a !== undefined && color.a !== 255 
            ? `rgba(${color.r}, ${color.g}, ${color.b}, ${(color.a / 255).toFixed(3)})` 
            : `rgb(${color.r}, ${color.g}, ${color.b})`;
        console.log(`  Color ${index}: ${rgbaString}`);
    });

    // Sort layers by area
    console.log('Sorting palette by pixel count for proper layering...');
    const pixelCounts = new Map(palette.map(c => [JSON.stringify(c), 0]));
    
    for (let i = 0; i < quantized.data.length; i += 4) {
        const r = quantized.data[i], g = quantized.data[i+1], b = quantized.data[i+2], a = quantized.data[i+3];
        if (a === 0) continue;
        const key = JSON.stringify({ r, g, b, a });
        if (pixelCounts.has(key)) {
            pixelCounts.set(key, pixelCounts.get(key) + 1);
        }
    }
    
    // Determine background color
    console.log('Detecting background color...');
    let backgroundColor = null;
    let maxCount = -1;
    
    for (const [colorStr, count] of pixelCounts.entries()) {
        const color = JSON.parse(colorStr);
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
    
    const paletteForTracing = palette.filter(color => {
        if (!backgroundColor) return true;
        
        return !(color.r === backgroundColor.r && 
                 color.g === backgroundColor.g && 
                 color.b === backgroundColor.b && 
                 color.a === backgroundColor.a);
    });
    
    console.log(`Palette filtered: ${palette.length} → ${paletteForTracing.length} colors (background excluded)`);

    const svgPaths = [];

    for (const color of paletteForTracing) {
        if (color.a === 0) continue;

        const rgbaString = color.a !== undefined && color.a !== 255 
            ? `rgba(${color.r}, ${color.g}, ${color.b}, ${(color.a / 255).toFixed(3)})` 
            : `rgb(${color.r}, ${color.g}, ${color.b})`;
        console.log(`Tracing color: ${rgbaString}`);
        
        try {
            const adaptiveTurdSize = Math.max(turdSize, Math.floor(Math.min(width, height) / 30));
            const adaptiveAlphaMax = Math.min(alphaMax, 0.5);
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

    // Assemble final SVG
    console.log('Assembling final SVG...');

    let finalViewBox = `0 0 ${width} ${height}`;
    if (svgPaths.length > 0 && svgPaths[0].viewBox) {
        finalViewBox = svgPaths[0].viewBox;
        console.log('Using viewBox from potrace:', finalViewBox);
    } else {
        console.log('Using original viewBox:', finalViewBox);
    }

    const viewBoxParts = finalViewBox.split(' ');
    const viewBoxHeight = parseFloat(viewBoxParts[3]) || height;
    const svgHeader = `<svg viewBox="${finalViewBox}" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%; max-width: 100%; max-height: 100%;" shape-rendering="geometricPrecision"><g transform="scale(1,-1) translate(0,-${viewBoxHeight})">`;
    
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