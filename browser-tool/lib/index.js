// index.js – Main entry point for unfake.js library
// Exports all public functions from pixel and vector modules

import {
  processImage,
  MAX_COLORS,
} from './pixel.js';
import { vectorizeImage } from './vector.js';
import { extractLayers, extractMaskLayer, extractDetailsLayer } from './layers.js';
import { 
  fileToImageData, 
  morphologicalCleanup,
  countColors,
  detectScale,
  median,
  mode,
  mean,
  logger,
  dominantOrMean,
  finalizePixels,
  encodePng,
  getPaletteFromImage,
  downscaleBlock
} from './utils.js';

// Re-export all functions
export {
  processImage,
  vectorizeImage,
  MAX_COLORS,
  extractLayers,
  extractMaskLayer,
  extractDetailsLayer,
};
export { 
  fileToImageData, 
  morphologicalCleanup,
  countColors,
  detectScale,
  median,
  mode,
  mean,
  logger,
  dominantOrMean,
  finalizePixels,
  encodePng,
  getPaletteFromImage,
  downscaleBlock
};

// Default export with all functions
export default {
  processImage,
  vectorizeImage,
  MAX_COLORS,
  extractLayers,
  extractMaskLayer,
  extractDetailsLayer,
  utils: {
    fileToImageData, 
    morphologicalCleanup,
    countColors,
    detectScale,
    median,
    mode,
    mean,
    logger,
    dominantOrMean,
    finalizePixels,
    encodePng,
    getPaletteFromImage,
    downscaleBlock
  }
}; 