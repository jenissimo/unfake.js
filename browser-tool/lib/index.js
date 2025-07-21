// index.js – Main entry point for unfake.js library
// Exports all public functions from pixel and vector modules

import { 
  processImage,
} from './pixel.js';
import { vectorizeImage } from './vector.js';
import { 
  cvReady, 
  fileToImageData, 
  morphologicalCleanup,
  countColors,
  detectScale,
  findOptimalCrop,
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
};
export { 
  cvReady, 
  fileToImageData, 
  morphologicalCleanup,
  countColors,
  detectScale,
  findOptimalCrop,
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
  utils: {
    cvReady, 
    fileToImageData, 
    morphologicalCleanup,
    countColors,
    detectScale,
    findOptimalCrop,
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