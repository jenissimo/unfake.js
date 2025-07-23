// unfake.js Browser Tool
// Main application file

// Enable debug logging
window.DEBUG_PIXEL_PROCESSOR = true;

// Import dependencies
import unfake from '../lib/index.js';
import { Pane } from 'tweakpane';
import Magnifier from './magnifier.js';
import { cvReady } from '../lib/utils.js';

// Global state
let appState = {
    mode: 'pixel', // 'pixel' or 'vector'
    originalImage: null,
    processedImage: {
        pixel: null,
        vector: null
    },
    isProcessing: false,
    opencvReady: false,
    magnifier: null,
    originalPalette: {
        pixel: null,
        vector: null
    } // Added for color replacement consistency
};

// DOM elements
let elements = {};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

// Main initialization
function initializeApp() {
    console.log('Initializing unfake.js browser tool...');
    
    // Cache DOM elements
    cacheElements();
    
    // Initialize UI components
    initializeUI();
    
    // Asynchronously wait for OpenCV to be ready
    cvReady().then(() => {
        console.log('OpenCV is ready.');
        appState.opencvReady = true;
        updateUI();
    }).catch(err => {
        console.error('Failed to load OpenCV:', err);
        alert('Could not initialize OpenCV. Please try refreshing the page.');
        // Optionally, disable processing features
        elements.processBtn.disabled = true;
        elements.processBtn.textContent = 'OpenCV Failed';
    });
    
    // Initialize magnifier
    initializeMagnifier();
    
    // Bind event listeners
    bindEvents();
    
    // Set initial UI state from hash
    if (!window.location.hash) {
        window.location.hash = 'pixel';
    } else {
        setModeFromHash();
    }
    
    // Update initial UI state
    updateUI();
    updateMainTitle();
}

// Set application mode based on URL hash
function setModeFromHash() {
    const newMode = window.location.hash.substring(1) || 'pixel';
    if (newMode === appState.mode && document.querySelector(`.mode-btn[data-mode="${newMode}"]`).classList.contains('active')) {
        return; // Already in the correct state
    }

    if (!['pixel', 'vector'].includes(newMode)) {
        appState.mode = 'pixel';
        window.location.hash = 'pixel'; // Correct invalid hash
    } else {
        appState.mode = newMode;
    }
    
    console.log(`Mode set to: ${appState.mode}`);

    // Update active button
    elements.modeBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === appState.mode);
    });

    // Update main title
    updateMainTitle();

    // Update UI
    updateProcessButtonText();
    updateSettingsVisibility();

    // Check if result for this mode is cached
    const cachedResult = appState.processedImage[appState.mode];
    if (cachedResult) {
        console.log(`Displaying cached result for ${appState.mode}`);
        displayResult(cachedResult);
        updateUI();
        // Обновляем хинт под оригиналом при смене режима
        updateOriginalImageHint();
        return;
    }

    // If there's an image, but no result for this mode, clear the display
    if (appState.originalImage) {
        clearResultArea();
        // Обновляем хинт под оригиналом при смене режима
        updateOriginalImageHint();
    }
}

// Cache DOM elements for performance
function cacheElements() {
    elements = {
        // Mode switching
        modeBtns: document.querySelectorAll('.mode-btn'),
        themeToggle: document.getElementById('theme-toggle'),
        magnifierToggle: document.getElementById('magnifier-toggle'),
        
        // Upload area
        uploadArea: document.getElementById('upload-area'),
        fileInput: document.getElementById('file-input'),
        imageDisplay: document.getElementById('image-display'),
        originalImage: document.getElementById('original-image'),
        resetBtn: document.getElementById('reset-btn'),
        
        // Process button
        processBtn: document.getElementById('process-btn'),
        btnText: document.querySelector('.btn-text'),
        btnLoading: document.querySelector('.btn-loading'),
        
        // Result area
        resultArea: document.getElementById('result-area'),
        actionButtons: document.querySelector('.action-buttons'),
        copyBtn: document.getElementById('copy-btn'),
        downloadBtn: document.getElementById('download-btn'),
        
        // Settings
        tweakpaneContainer: document.getElementById('tweakpane-container'),
        
        // Result settings
        resultTweakpaneContainer: document.getElementById('result-tweakpane-container')
    };
}

// Initialize UI components
function initializeUI() {
    // Initialize Tweakpane
    initializeTweakpane();
    
    // Initialize Result Tweakpane
    initializeResultTweakpane();
    
    // Initialize theme
    initializeTheme();
}

// Initialize Tweakpane settings panel
function initializeTweakpane() {
    // Create separate Tweakpane instances for each mode
    const pixelPane = new Pane({
        container: elements.tweakpaneContainer
    });
    
    const vectorPane = new Pane({
        container: elements.tweakpaneContainer
    });
    
    // Settings object
    const settings = {
        // Pixel Art settings
        maxColors: 16,
        snapGrid: true,
        gridDetectionAlgorithm: 'auto-tiled', // 'auto-tiled', 'auto-legacy'
        downscaleMethod: 'dominant',
        domMeanThreshold: 0.05,
        cleanupMorph: true,
        cleanupJaggy: true,
        autoPixelSize: true,
        pixelSize: 4,
        alphaThreshold: 128,
        enableAlphaBinarization: true,

        // Vector settings
        preProcess: true,
        preProcessFilter: 'bilateral',
        preProcessValue: 30,
        preProcessMorphology: false,
        
        quantize: true,
        quantizeAuto: true,

        postProcess: true,
        postProcessFilter: 'median', // Switched to median for sharper edges
        postProcessValue: 3,     // Small value is effective for median

        // ImageTracer.js specific settings
        numberofcolors: 16,
        ltres: 2,           // Relaxed tolerance to encourage smoother lines
        qtres: 2,           // Relaxed tolerance to encourage smoother curves
        pathomit: 30,
        rightangleenhance: true,
        strokewidth: 1.5,   // Use a small stroke to fill any potential gaps
        blurradius: 0,      // Disabled; we use our own post-process smoothing
        blurdelta: 20,
        linefilter: true,   // Enabled for smoother lines
        roundcoords: 2,     // Increased precision to reduce gaps
        viewbox: true,
        
        // Magnifier settings
        magnifierZoom: 4,
        magnifierSize: 200,
        magnifierShowCrosshair: true,
        magnifierCrispPixels: true,
        magnifierDebug: false,
        
        // Palette (will be populated dynamically)
        palette: []
    };
    
    // --- Pixel Art Pane ---
    const mainPixelFolder = pixelPane.addFolder({
        title: 'Main Settings',
        expanded: true
    });

    mainPixelFolder.addBinding(settings, 'maxColors', {
        label: 'Max Colors',
        min: 2,
        max: 256,
        step: 1
    });
    
    mainPixelFolder.addBinding(settings, 'snapGrid', {
        label: 'Snap to Grid'
    });

    mainPixelFolder.addBinding(settings, 'gridDetectionAlgorithm', {
        label: 'Grid Detection',
        options: {
            'Complex Grid': 'auto-tiled',
            'Simple Grid': 'auto-legacy',
        }
    });

    const advancedPixelFolder = pixelPane.addFolder({
        title: 'Advanced Settings',
        expanded: false,
    });
    
    // --- Downscaling Sub-folder ---
    const downscalingFolder = advancedPixelFolder.addFolder({ title: 'Downscaling Method' });

    downscalingFolder.addBinding(settings, 'downscaleMethod', {
        label: 'Method',
        options: {
            'dominant': 'dominant',
            'median': 'median',
            'mode': 'mode', 
            'mean': 'mean',
            'nearest': 'nearest',
            'content-adaptive (experimental)': 'content-adaptive',
        }
    });
    
    const domMeanThresholdBinding = downscalingFolder.addBinding(settings, 'domMeanThreshold', {
        label: 'Dominant Threshold',
        min: 0.01,
        max: 0.5,
        step: 0.01
    });
    
    domMeanThresholdBinding.hidden = !['dominant', 'median', 'mode', 'mean'].includes(settings.downscaleMethod);
    downscalingFolder.children[0].on('change', (ev) => {
        domMeanThresholdBinding.hidden = !['dominant', 'median', 'mode', 'mean'].includes(ev.value);
    });
    
    // --- Manual Pixel Size Sub-folder ---
    const pixelSizeFolder = advancedPixelFolder.addFolder({ title: 'Manual Pixel Size' });
    const autoPixelSizeBinding = pixelSizeFolder.addBinding(settings, 'autoPixelSize', {
        label: 'Auto-detect size'
    });
    
    const pixelSizeBinding = pixelSizeFolder.addBinding(settings, 'pixelSize', {
        label: 'Pixel Size',
        min: 1,
        max: 32,
        step: 1
    });
    
    pixelSizeBinding.hidden = settings.autoPixelSize;
    autoPixelSizeBinding.on('change', (ev) => {
        pixelSizeBinding.hidden = ev.value;
    });
    
    // --- Cleanup Sub-folder ---
    const cleanupFolder = advancedPixelFolder.addFolder({ title: 'Image Cleanup' });
    cleanupFolder.addBinding(settings, 'cleanupMorph', {
        label: 'Morphological Cleanup'
    });
    
    cleanupFolder.addBinding(settings, 'cleanupJaggy', {
        label: 'Jaggy Cleanup'
    });
    
    const enableAlphaBinarizationBinding = cleanupFolder.addBinding(settings, 'enableAlphaBinarization', {
        label: 'Enable Alpha Binarization'
    });
    
    const alphaThresholdBinding = cleanupFolder.addBinding(settings, 'alphaThreshold', {
        label: 'Alpha Threshold',
        min: 0,
        max: 255,
        step: 1
    });

    alphaThresholdBinding.hidden = !settings.enableAlphaBinarization;
    enableAlphaBinarizationBinding.on('change', (ev) => {
        alphaThresholdBinding.hidden = !ev.value;
        if (!ev.value) {
            settings.alphaThreshold = 0;
        } else if (settings.alphaThreshold === 0) {
            settings.alphaThreshold = 128;
        }
    });


    // Vector folder
    const vectorFolder = vectorPane.addFolder({
        title: 'Vector Settings',
        expanded: true
    });

    // --- Pre-processing Folder ---
    const preProcessFolder = vectorFolder.addFolder({ title: 'Noise Reduction', expanded: true });
    
    const preProcessBinding = preProcessFolder.addBinding(settings, 'preProcess', { label: 'Enable' });
    const preProcessFilterBinding = preProcessFolder.addBinding(settings, 'preProcessFilter', { label: 'Filter', options: { Bilateral: 'bilateral', Median: 'median' }});
    const preProcessValueBinding = preProcessFolder.addBinding(settings, 'preProcessValue', { label: 'Filter Strength', min: 1, max: 50, step: 1 });
    const preProcessMorphologyBinding = preProcessFolder.addBinding(settings, 'preProcessMorphology', { label: 'Fill Gaps' });
    
    function updatePreProcessVisibility() {
        const enabled = settings.preProcess;
        preProcessFilterBinding.hidden = !enabled;
        preProcessValueBinding.hidden = !enabled;
        preProcessMorphologyBinding.hidden = !enabled;
    }
    preProcessBinding.on('change', updatePreProcessVisibility);
    
    // --- Quantization Folder ---
    const quantizeFolder = vectorFolder.addFolder({ title: 'Color Palette', expanded: true });
    
    const quantizeBinding = quantizeFolder.addBinding(settings, 'quantize', { label: 'Enable' });
    const quantizeAutoBinding = quantizeFolder.addBinding(settings, 'quantizeAuto', { label: 'Auto-detect Colors' });
    
    const numberOfColorsBinding = quantizeFolder.addBinding(settings, 'numberofcolors', {
        label: 'Manual Colors',
        min: 2,
        max: 64,
        step: 1
    });

    function updateQuantizeVisibility() {
        const quantizeEnabled = settings.quantize;
        quantizeAutoBinding.hidden = !quantizeEnabled;
        numberOfColorsBinding.hidden = !quantizeEnabled || settings.quantizeAuto;
    }
    
    quantizeBinding.on('change', updateQuantizeVisibility);
    quantizeAutoBinding.on('change', updateQuantizeVisibility);

    // --- Post-processing Folder (Smoothing) ---
    const postProcessFolder = vectorFolder.addFolder({ title: 'Post-Quantization Smoothing', expanded: true });
    
    const postProcessBinding = postProcessFolder.addBinding(settings, 'postProcess', { label: 'Enable' });
    const postProcessFilterBinding = postProcessFolder.addBinding(settings, 'postProcessFilter', { label: 'Filter', options: { Median: 'median', Gaussian: 'gaussian' }});
    const postProcessValueBinding = postProcessFolder.addBinding(settings, 'postProcessValue', { label: 'Smoothing Level', min: 1, max: 11, step: 2 });
    
    function updatePostProcessVisibility() {
        const enabled = settings.postProcess;
        postProcessFilterBinding.hidden = !enabled;
        postProcessValueBinding.hidden = !enabled;
    }
    postProcessBinding.on('change', updatePostProcessVisibility);
    
    // --- Tracer Settings Folder (Advanced) ---
    const tracerFolder = vectorFolder.addFolder({ title: 'Tracer Settings (Advanced)', expanded: false });

    tracerFolder.addBinding(settings, 'ltres', {
        label: 'Line Threshold',
        min: 0.01,
        max: 10,
        step: 0.01
    });

    tracerFolder.addBinding(settings, 'qtres', {
        label: 'Curve Threshold',
        min: 0.01,
        max: 10,
        step: 0.01
    });

    tracerFolder.addBinding(settings, 'pathomit', {
        label: 'Path Omit',
        min: 0,
        max: 50,
        step: 1
    });

    tracerFolder.addBinding(settings, 'rightangleenhance', {
        label: 'Right Angle Enhance'
    });

    tracerFolder.addBinding(settings, 'strokewidth', {
        label: 'Stroke Width',
        min: 0,
        max: 10,
        step: 0.1
    });

    tracerFolder.addBinding(settings, 'blurradius', {
        label: 'Blur Radius',
        min: 0,
        max: 32,
        step: 1
    });

    tracerFolder.addBinding(settings, 'blurdelta', {
        label: 'Blur Delta',
        min: 0,
        max: 256,
        step: 1
    });

    tracerFolder.addBinding(settings, 'linefilter', {
        label: 'Line Filter'
    });

    tracerFolder.addBinding(settings, 'roundcoords', {
        label: 'Round Coords',
        min: 0,
        max: 3,
        step: 1
    });

    tracerFolder.addBinding(settings, 'viewbox', {
        label: 'SVG ViewBox'
    });
    
    // Initialize visibility
    updatePreProcessVisibility();
    updateQuantizeVisibility();
    updatePostProcessVisibility();
    
    // Magnifier folder (shared between modes)
    const magnifierFolder = pixelPane.addFolder({
        title: 'Magnifier Settings',
        expanded: false
    });
    
    const vectorMagnifierFolder = vectorPane.addFolder({
        title: 'Magnifier Settings',
        expanded: false
    });
    
    magnifierFolder.addBinding(settings, 'magnifierZoom', {
        label: 'Zoom Level',
        min: 1,
        max: 16,
        step: 1
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.setZoomLevel(ev.value);
        }
    });
    
    magnifierFolder.addBinding(settings, 'magnifierSize', {
        label: 'Size',
        min: 100,
        max: 400,
        step: 10
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.setSize(ev.value);
        }
    });
    
    magnifierFolder.addBinding(settings, 'magnifierShowCrosshair', {
        label: 'Show Crosshair'
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.options.showCrosshair = ev.value;
            if (appState.magnifier.isActive) {
                appState.magnifier.updateMagnification();
            }
        }
    });
    
    magnifierFolder.addBinding(settings, 'magnifierCrispPixels', {
        label: 'Crisp Pixels'
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.options.crispPixels = ev.value;
            if (appState.magnifier.isActive) {
                appState.magnifier.updateMagnification();
            }
        }
    });
    
    magnifierFolder.addBinding(settings, 'magnifierDebug', {
        label: 'Debug Mode'
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.options.debug = ev.value;
        }
    });
    
    // Add same magnifier settings to vector pane
    vectorMagnifierFolder.addBinding(settings, 'magnifierZoom', {
        label: 'Zoom Level',
        min: 1,
        max: 16,
        step: 1
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.setZoomLevel(ev.value);
        }
    });
    
    vectorMagnifierFolder.addBinding(settings, 'magnifierSize', {
        label: 'Size',
        min: 100,
        max: 400,
        step: 10
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.setSize(ev.value);
        }
    });
    
    vectorMagnifierFolder.addBinding(settings, 'magnifierShowCrosshair', {
        label: 'Show Crosshair'
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.options.showCrosshair = ev.value;
            if (appState.magnifier.isActive) {
                appState.magnifier.updateMagnification();
            }
        }
    });
    
    vectorMagnifierFolder.addBinding(settings, 'magnifierCrispPixels', {
        label: 'Crisp Pixels'
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.options.crispPixels = ev.value;
            if (appState.magnifier.isActive) {
                appState.magnifier.updateMagnification();
            }
        }
    });
    
    vectorMagnifierFolder.addBinding(settings, 'magnifierDebug', {
        label: 'Debug Mode'
    }).on('change', (ev) => {
        if (appState.magnifier) {
            appState.magnifier.options.debug = ev.value;
        }
    });
    
    // Store settings and pane references globally
    window.appSettings = settings;
    window.appPanes = {
        pixel: pixelPane,
        vector: vectorPane
    };
    
    // Update pane visibility based on mode
    updateSettingsVisibility();
}

// Helper functions for color conversion
function hexToRgba(hex) {
    if (!hex) return null;
    let localHex = hex.toLowerCase();
    if (localHex.startsWith('#')) {
        localHex = localHex.slice(1);
    }
    
    if (localHex.length === 3) {
        localHex = localHex[0] + localHex[0] + localHex[1] + localHex[1] + localHex[2] + localHex[2];
    }

    if (localHex.length === 6) {
        localHex += 'ff';
    }

    if (localHex.length !== 8) {
        return null; // Invalid hex code
    }

    const bigint = parseInt(localHex, 16);
    return {
        r: (bigint >> 24) & 255,
        g: (bigint >> 16) & 255,
        b: (bigint >> 8) & 255,
        a: bigint & 255
    };
}

function rgbaToHex(color) {
    const toHex = (c) => Math.round(c).toString(16).padStart(2, '0');
    return `#${toHex(color.r)}${toHex(color.g)}${toHex(color.b)}${toHex(color.a || 255)}`;
}

function rgbToHex(color) {
    const toHex = (c) => {
        const hex = Math.round(c).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    };
    return `#${toHex(color.r)}${toHex(color.g)}${toHex(color.b)}`;
}

function handlePaletteChange(newPaletteObj) {
    const processedImage = appState.processedImage[appState.mode];
    if (!processedImage || !processedImage.palette) return;

    // Use the original palette for creating the color map
    const originalPalette = appState.originalPalette[appState.mode] || processedImage.palette;

    // Build a map of old hex -> new hex, comparing normalized values
    const hexColorMap = new Map();
    originalPalette.forEach((oldColor, index) => {
        const key = `color${index + 1}`;
        const newColor = newPaletteObj[key];
        if (oldColor && newColor && oldColor.toLowerCase() !== newColor.toLowerCase()) {
            hexColorMap.set(oldColor.toLowerCase(), newColor.toLowerCase());
        }
    });

    if (appState.mode === 'pixel') {
        const canvas = processedImage.canvas;
        const originalImageData = processedImage.originalImageData;
        if (!canvas || !originalImageData) {
            console.error('Canvas or original image data not available for palette change.');
            return;
        }

        try {
            const ctx = canvas.getContext('2d', { willReadFrequently: true });

            // This map translates any color from the original image to its final replacement color
            const finalColorMap = new Map();

            // Pre-populate the finalColorMap with replacements
            originalPalette.forEach((hex) => {
                const oldHex = hex.toLowerCase();
                const newHex = hexColorMap.get(oldHex);
                if (newHex) {
                    finalColorMap.set(oldHex, hexToRgba(newHex));
                }
            });

            // We work from the original data every time to apply the full new palette
            const originalData = originalImageData.data;
            const newData = new Uint8ClampedArray(originalData.length);
            
            for (let i = 0; i < originalData.length; i += 4) {
                const r = originalData[i];
                const g = originalData[i + 1];
                const b = originalData[i + 2];
                const a = originalData[i + 3];

                if (a === 0) { // Preserve fully transparent pixels
                    newData[i] = 0; newData[i + 1] = 0; newData[i + 2] = 0; newData[i + 3] = 0;
                    continue;
                }
                
                const currentHex = rgbaToHex({ r, g, b, a }).toLowerCase();
                
                let finalRgba = finalColorMap.get(currentHex);

                if (finalRgba === undefined) {
                    // This color wasn't in the original palette but is in the image.
                    // This can happen with anti-aliasing. We keep it as is.
                    finalColorMap.set(currentHex, { r, g, b, a });
                    finalRgba = { r, g, b, a };
                }
                
                newData[i] = finalRgba.r;
                newData[i + 1] = finalRgba.g;
                newData[i + 2] = finalRgba.b;
                newData[i + 3] = finalRgba.a;
            }
            
            const newImageData = new ImageData(newData, originalImageData.width, originalImageData.height);
            ctx.putImageData(newImageData, 0, 0);

        } catch (error) {
            console.error('Error updating pixel art palette on canvas:', error);
        }

    } else { // vector
        if (!processedImage.originalSvg || !appState.originalPalette) {
            console.error('Original SVG or palette not available for palette change.');
            return;
        }
        
        let currentSvg = processedImage.originalSvg;
        const newPalette = Object.values(newPaletteObj);

        appState.originalPalette[appState.mode].forEach((oldHex, index) => {
            const newHex = newPalette[index];
            
            if (oldHex && newHex) {
                const oldRgba = hexToRgba(oldHex);
                const newRgba = hexToRgba(newHex);

                if (oldRgba && newRgba) {
                    const newRgbString = `rgb(${newRgba.r},${newRgba.g},${newRgba.b})`;
                    const oldRgbRegexPart = `rgb\\(\\s*${oldRgba.r}\\s*,\\s*${oldRgba.g}\\s*,\\s*${oldRgba.b}\\s*\\)`;
                    const oldColorRegexPart = `(${oldHex.replace('#', '#?')}|${oldRgbRegexPart})`;
                    const finalRegex = new RegExp(`(fill|stroke)="${oldColorRegexPart}"`, 'gi');
                    
                    currentSvg = currentSvg.replace(finalRegex, `$1="${newRgbString}"`);
                }
            }
        });
        
        processedImage.svg = currentSvg;
        elements.resultArea.querySelector('.result-svg').innerHTML = currentSvg;
    }

    // Update the master palette state for the next change
    processedImage.palette = Object.values(newPaletteObj);
    appState.originalPalette[appState.mode] = [...processedImage.palette];
}


// Initialize Result Tweakpane for palette editing
function initializeResultTweakpane() {
    // Create Result Tweakpane instance
    const resultPane = new Pane({
        container: elements.resultTweakpaneContainer,
        title: 'Result Settings'
    });
    
    // Result settings object
    const resultSettings = {
        palette: [],
        exportFormat: 'png'
    };
    
    // Palette folder
    const paletteFolder = resultPane.addFolder({
        title: 'Palette',
        expanded: true
    });
    
    // Store result settings and pane globally
    window.resultSettings = resultSettings;
    window.resultPane = resultPane;
    window.paletteFolder = paletteFolder;
    
    // Initially hide the result pane
    resultPane.element.style.display = 'none';
}

// Update settings visibility based on current mode
function updateSettingsVisibility() {
    if (!window.appPanes) return;
    
    const pixelPane = window.appPanes.pixel;
    const vectorPane = window.appPanes.vector;
    
    if (appState.mode === 'pixel') {
        pixelPane.element.style.display = 'block';
        vectorPane.element.style.display = 'none';
    } else {
        pixelPane.element.style.display = 'none';
        vectorPane.element.style.display = 'block';
    }
    
    // Magnifier settings are always visible (shared between modes)
    // The folder is added to both panes, so it will be visible in the active pane
}

// Initialize magnifier
function initializeMagnifier() {
    appState.magnifier = new Magnifier({
        zoomLevel: 4,
        size: 200,
        showCrosshair: true,
        crosshairColor: '#ff0000',
        crispPixels: true,
        debug: false
    });
    handleMagnifierToggle();
    console.log('Magnifier initialized');
}

// Initialize theme system
function initializeTheme() {
    const savedTheme = localStorage.getItem('unfake-theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

// Update theme icon
function updateThemeIcon(theme) {
    const icon = elements.themeToggle;
    icon.textContent = theme === 'dark' ? '☀️' : '🌙';
}

// Bind all event listeners
function bindEvents() {
    // Mode switching
    elements.modeBtns.forEach(btn => {
        btn.addEventListener('click', handleModeSwitch);
    });
    
    // Theme toggle
    elements.themeToggle.addEventListener('click', handleThemeToggle);
    
    // Magnifier toggle
    elements.magnifierToggle.addEventListener('click', handleMagnifierToggle);
    
    // File upload
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('drop', handleFileDrop);
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Reset button
    elements.resetBtn.addEventListener('click', handleReset);
    
    // Process button
    elements.processBtn.addEventListener('click', handleProcess);
    
    // Action buttons
    elements.copyBtn.addEventListener('click', handleCopy);
    elements.downloadBtn.addEventListener('click', handleDownload);
    
    // Paste from clipboard
    document.addEventListener('paste', handlePaste);

    // Add hashchange listener for mode switching
    window.addEventListener('hashchange', setModeFromHash);
}

// Handle mode switching
function handleModeSwitch(e) {
    const newMode = e.target.dataset.mode;
    if (newMode !== appState.mode) {
        window.location.hash = newMode;
    }
}

// Handle theme toggle
function handleThemeToggle() {
    const currentTheme = document.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.body.setAttribute('data-theme', newTheme);
    localStorage.setItem('unfake-theme', newTheme);
    updateThemeIcon(newTheme);
}

// Handle magnifier toggle
function handleMagnifierToggle() {
    if (!appState.magnifier) return;
    
    const isActive = appState.magnifier.isActive;
    
    if (isActive) {
        appState.magnifier.deactivate();
        elements.magnifierToggle.classList.remove('active');
        elements.magnifierToggle.textContent = '🔍';
        elements.magnifierToggle.title = 'Toggle magnifier';
    } else {
        // Check if we have any image to magnify
        const originalImage = elements.originalImage;
        const resultImage = elements.resultArea.querySelector('img');
        
        if (!originalImage && !resultImage) {
            alert('Please upload an image first');
            return;
        }
        
        // Activate magnifier - it will automatically find images under cursor
        appState.magnifier.activate();
        elements.magnifierToggle.classList.add('active');
        elements.magnifierToggle.textContent = '🔍';
        elements.magnifierToggle.title = 'Magnifier active - hover over images';
    }
}



// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('drag-over');
}

// Handle file drop
function handleFileDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file select
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle paste from clipboard
function handlePaste(e) {
    const items = e.clipboardData.items;
    for (let item of items) {
        if (item.type.indexOf('image') !== -1) {
            const file = item.getAsFile();
            handleFile(file);
            break;
        }
    }
}



// Handle file processing
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file (PNG, JPG, JPEG, WebP)');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
        alert('File size must be less than 10MB');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            appState.originalImage = img;
            appState.originalFile = file; // Store the original file for processing
            displayOriginalImage(img);
            updateUI();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Display original image
function displayOriginalImage(img) {
    elements.originalImage.src = img.src;
    elements.uploadArea.style.display = 'none';
    elements.imageDisplay.style.display = 'block';
    
    // Add size information
    const sizeInfo = document.createElement('div');
    sizeInfo.className = 'image-size-info';
    sizeInfo.textContent = `${img.naturalWidth} × ${img.naturalHeight} pixels`;
    
    // Remove existing size info if any
    const existingInfo = elements.imageDisplay.querySelector('.image-size-info');
    if (existingInfo) {
        existingInfo.remove();
    }
    elements.imageDisplay.appendChild(sizeInfo);

    // Remove existing hint if any
    const existingHint = elements.imageDisplay.querySelector('.image-hint');
    if (existingHint) {
        existingHint.remove();
    }
    // Add hint if needed
    const hint = getOriginalImageHint(img.naturalWidth, img.naturalHeight, appState.mode);
    if (hint) {
        const hintDiv = document.createElement('div');
        hintDiv.className = 'image-hint';
        hintDiv.textContent = hint;
        elements.imageDisplay.appendChild(hintDiv);
    }
}

// Возвращает текст хинта для оригинального изображения (или null)
function getOriginalImageHint(width, height, mode) {
    if (mode === 'pixel' && width < 128 && height < 128) {
        return 'The image is very small. Are you sure this is not a mistake? Pixel art downscaling works best with larger images.';
    }
    if (mode === 'vector' && (width < 1024 || height < 1024)) {
        return 'The image is quite small for vectorization. For better results, consider lowering Filter Strength, Smoothing Level, Stroke Width, and Path Omit in the settings.';
    }
    return null;
}

// Обновляет хинт под оригинальным изображением (например, при смене режима)
function updateOriginalImageHint() {
    const img = appState.originalImage;
    if (!img) return;
    // Remove existing hint if any
    const existingHint = elements.imageDisplay.querySelector('.image-hint');
    if (existingHint) {
        existingHint.remove();
    }
    const hint = getOriginalImageHint(img.naturalWidth, img.naturalHeight, appState.mode);
    if (hint) {
        const hintDiv = document.createElement('div');
        hintDiv.className = 'image-hint';
        hintDiv.textContent = hint;
        elements.imageDisplay.appendChild(hintDiv);
    }
}

// Handle reset
function handleReset() {
    appState.originalImage = null;
    appState.originalFile = null;
    appState.processedImage = { pixel: null, vector: null };
    
    elements.uploadArea.style.display = 'flex';
    elements.imageDisplay.style.display = 'none';
    elements.processBtn.style.display = 'none';
    elements.actionButtons.style.display = 'none';
    
    // Clear result area
    clearResultArea();
    
    // Clear original image size info
    const originalSizeInfo = elements.imageDisplay.querySelector('.image-size-info');
    if (originalSizeInfo) {
        originalSizeInfo.remove();
    }
    
    // Hide result Tweakpane
    if (window.resultPane) {
        window.resultPane.element.style.display = 'none';
    }
    
    // Deactivate magnifier if active
    if (appState.magnifier && appState.magnifier.isActive) {
        appState.magnifier.deactivate();
        elements.magnifierToggle.classList.remove('active');
        elements.magnifierToggle.textContent = '🔍';
        elements.magnifierToggle.title = 'Toggle magnifier';
    }
    
    updateUI();
}

// Handle process button click
async function handleProcess() {
    if (!appState.originalImage) {
        alert('Please upload an image first');
        return;
    }
    
    if (!appState.opencvReady) {
        alert('OpenCV is not ready yet. Please wait a moment and try again.');
        return;
    }
    
    setProcessingState(true);
    
    try {
        console.log('Starting image processing...');
        const result = await processImage();
        console.log('Processing completed successfully');

        // Store result in the correct mode-specific slot
        appState.processedImage[appState.mode] = result;

        displayResult(result);
        updateUI();
    } catch (error) {
        console.error('Processing error:', error);
        
        // More specific error messages
        let errorMessage = 'Error processing image: ';
        const errorStr = String(error.message || error);
        
        if (errorStr.includes('OpenCV')) {
            errorMessage += 'OpenCV library error. Please refresh the page and try again.';
        } else if (errorStr.includes('timeout')) {
            errorMessage += 'Processing timed out. Try with a smaller image.';
        } else if (errorStr.includes('memory')) {
            errorMessage += 'Not enough memory. Try with a smaller image.';
        } else if (errorStr.includes('1928192')) {
            errorMessage += 'Image processing failed. Try with different settings or a smaller image.';
        } else {
            errorMessage += errorStr;
        }
        
        alert(errorMessage);
    } finally {
        setProcessingState(false);
    }
}

// Process image using unfake.js
async function processImage() {
    const settings = window.appSettings;
    try {
        if (appState.mode === 'pixel') {
            console.log('Processing pixel art with settings:', settings);
            const manualScale = settings.autoPixelSize ? null : [settings.pixelSize, settings.pixelSize];
            
            const [detectMethod, edgeDetectMethod] = settings.gridDetectionAlgorithm.split('-');

            console.log('Manual scale value:', manualScale);
            if (manualScale) {
                console.log(`Using manual pixel size: ${settings.pixelSize}×${settings.pixelSize}px`);
            } else {
                console.log('Using automatic scale detection');
            }
            if (['dominant', 'median', 'mode', 'mean'].includes(settings.downscaleMethod)) {
                console.log(`Using ${settings.downscaleMethod} threshold: ${settings.domMeanThreshold} (${(settings.domMeanThreshold * 100).toFixed(1)}%)`);
            }
            return await unfake.processImage({
                file: appState.originalFile,
                maxColors: settings.maxColors,
                snapGrid: settings.snapGrid,
                detectMethod: detectMethod,
                edgeDetectMethod: edgeDetectMethod || 'tiled',
                downscaleMethod: settings.downscaleMethod,
                domMeanThreshold: settings.domMeanThreshold,
                manualScale: manualScale,
                cleanup: {
                    morph: settings.cleanupMorph,
                    jaggy: settings.cleanupJaggy
                },
                alphaThreshold: settings.enableAlphaBinarization ? settings.alphaThreshold : null
            });
        } else {
            console.log('Processing vector with settings:', settings);
            
            const preProcessOptions = {
                enabled: settings.preProcess,
                filter: settings.preProcessFilter,
                value: settings.preProcessValue,
                morphology: settings.preProcessMorphology,
            };

            const quantizeOptions = {
                enabled: settings.quantize,
                maxColors: settings.quantizeAuto ? 'auto' : settings.numberofcolors,
            };

            const postProcessOptions = {
                enabled: settings.postProcess,
                filter: settings.postProcessFilter,
                value: settings.postProcessValue,
            };

            // Pass all imagetracer options
            return await unfake.vectorizeImage({
                file: appState.originalFile,
                preProcess: preProcessOptions,
                quantize: quantizeOptions,
                postProcess: postProcessOptions,
                numberofcolors: settings.numberofcolors,
                ltres: settings.ltres,
                qtres: settings.qtres,
                pathomit: settings.pathomit,
                rightangleenhance: settings.rightangleenhance,
                strokewidth: settings.strokewidth,
                blurradius: settings.blurradius,
                blurdelta: settings.blurdelta,
                linefilter: settings.linefilter,
                roundcoords: settings.roundcoords,
                viewbox: settings.viewbox
            });
        }
    } catch (error) {
        console.error('Error in processImage:', error);
        // Ensure error is a proper Error object
        if (typeof error === 'number') {
            throw new Error(`Processing failed with code: ${error}`);
        } else if (typeof error === 'string') {
            throw new Error(error);
        } else {
            throw error;
        }
    }
}

// Display result
function displayResult(result) {
    if (appState.mode === 'pixel') {
        displayPixelArtResult(result);
    } else {
        displayVectorResult(result);
    }
    

}

// Display pixel art result
function displayPixelArtResult(result) {
    if (typeof UPNG === 'undefined') {
        console.error('UPNG.js is not available.');
        alert('A required library (UPNG.js) is missing. Please reload the page.');
        return;
    }

    console.log('Displaying pixel art result:', result);
    console.log('PNG buffer size:', result.png.byteLength);
    console.log('Palette size:', result.palette.length);
    
    try {
        // Decode PNG and get raw RGBA pixel data to avoid color space issues with drawImage
        const png = UPNG.decode(result.png);
        const rgba8 = UPNG.toRGBA8(png)[0];
        const rgba8Clamped = new Uint8ClampedArray(rgba8);
        const imageData = new ImageData(rgba8Clamped, png.width, png.height);

        // Get the visual size of the original image in the interface
        const originalImgElement = elements.originalImage;
        const visualOriginalWidth = originalImgElement.offsetWidth;
        const visualOriginalHeight = originalImgElement.offsetHeight;
        
        // Calculate scale factor to match the visual size of the original
        const scaleFactor = Math.max(visualOriginalWidth / png.width, visualOriginalHeight / png.height);
        
        // Get actual pixel size from manifest or settings
        let pixelSize = 1;
        let pixelSizeInfo = '';
        
        console.log('Manifest processing_steps:', result.manifest?.processing_steps);
        
        if (result.manifest && result.manifest.processing_steps && result.manifest.processing_steps.scale_detection) {
            const scaleDetection = result.manifest.processing_steps.scale_detection;
            pixelSize = scaleDetection.detected_scale || 1;
            console.log('Using pixel size from manifest:', pixelSize);
            
            if (scaleDetection.manual_scale) {
                pixelSizeInfo = `Manual pixel size: ${pixelSize}×${pixelSize}px`;
            } else {
                pixelSizeInfo = `Detected pixel size: ${pixelSize}×${pixelSize}px`;
            }
        } else if (window.appSettings && !window.appSettings.autoPixelSize) {
            pixelSize = window.appSettings.pixelSize;
            pixelSizeInfo = `Manual pixel size: ${pixelSize}×${pixelSize}px`;
        } else {
            pixelSizeInfo = `Pixel size: ${pixelSize}×${pixelSize}px`;
        }
        
        // Add additional info about processing
        let processingInfo = '';
        if (result.manifest && result.manifest.processing_steps) {
            const steps = result.manifest.processing_steps;
            const info = [];
            
            // Scale detection info
            if (steps.scale_detection) {
                const scaleInfo = steps.scale_detection;
                let methodInfo = `Detection: ${scaleInfo.method}`;
                if ((scaleInfo.method === 'edge' || scaleInfo.method === 'auto') && !scaleInfo.manual_scale) {
                    // Reformat the display string to be cleaner
                    const edgeMethodDisplay = (scaleInfo.edge_method === 'legacy') ? 'Simple' : 'Complex';
                    methodInfo += ` (${edgeMethodDisplay})`;
                }
                info.push(methodInfo);
            }
            
            // Downscaling info
            if (steps.downscaling) {
                const downscaleInfo = steps.downscaling;
                if (downscaleInfo.method) {
                    let methodInfo = `Method: ${downscaleInfo.method}`;
                    if (['dominant', 'median', 'mode', 'mean'].includes(downscaleInfo.method) && downscaleInfo.dom_mean_threshold) {
                        methodInfo += ` (${(downscaleInfo.dom_mean_threshold * 100).toFixed(0)}%)`;
                    }
                    info.push(methodInfo);
                }
            }
            
            // Color quantization info
            if (steps.color_quantization) {
                const colorInfo = steps.color_quantization;
                if (colorInfo.final_colors && colorInfo.max_colors) {
                    info.push(`Colors: ${colorInfo.final_colors}/${colorInfo.max_colors}`);
                }
            }
            
            processingInfo = info.join(' • ');
        }
        
        const canvas = document.createElement('canvas');
        canvas.width = png.width;
        canvas.height = png.height;
        canvas.style.cssText = `width: ${png.width * scaleFactor}px; height: ${png.height * scaleFactor}px; image-rendering: pixelated; image-rendering: -moz-crisp-edges; image-rendering: crisp-edges;`;
        
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.putImageData(imageData, 0, 0);

        const resultHtml = `
            <div class="result-image"></div>
            <div class="image-size-info">
                ${png.width} × ${png.height} pixels (${pixelSizeInfo})
            </div>
            ${processingInfo ? `<div class="processing-info">${processingInfo}</div>` : ''}
        `;
        
        elements.resultArea.innerHTML = resultHtml;
        elements.resultArea.querySelector('.result-image').appendChild(canvas);

        const newProcessedImageState = { 
            ...result,
            canvas: canvas,
            originalImageData: imageData
        };
        delete newProcessedImageState.png;

        appState.processedImage[appState.mode] = newProcessedImageState;
    
    } catch (error) {
        console.error('Error displaying pixel art result:', error);
        alert('Failed to display the processed pixel art image.');
    }
    
    elements.downloadBtn.textContent = 'Download PNG';
    elements.actionButtons.style.display = 'flex';
    
    // Ensure palette is in hex format for consistency
    if (appState.processedImage[appState.mode] && appState.processedImage[appState.mode].palette) {
        appState.processedImage[appState.mode].palette = appState.processedImage[appState.mode].palette.map(p => {
            if (typeof p === 'string') return p;
            return rgbaToHex({r: p.R, g: p.G, b: p.B, a: p.A});
        });
    }
    
    // Update palette in Result Tweakpane
    updatePaletteInTweakpane(result.palette);
    
    console.log('Result displayed successfully');
}



// Update palette in Result Tweakpane
function updatePaletteInTweakpane(palette) {
    if (!window.paletteFolder || !window.resultSettings || !palette) return;
    
    const paletteFolder = window.paletteFolder;

    // Clear existing palette bindings and listeners
    paletteFolder.children.forEach(child => paletteFolder.remove(child));
    paletteFolder.off('change', window.handlePaletteFolderChange);

    const paletteHex = palette.map(p => {
        if (typeof p === 'string') return p; // Already hex
        if (p.a !== undefined && p.a < 255) {
             return rgbaToHex({r: p.R, g: p.G, b: p.B, a: p.A});
        }
        return rgbToHex({r: p.R, g: p.G, b: p.B});
    });

    // Store the initial palette для текущего режима
    appState.originalPalette[appState.mode] = [...paletteHex];
    
    // Create palette object with color properties
    const paletteObj = {};
    paletteHex.forEach((color, index) => {
        paletteObj[`color${index + 1}`] = color;
    });
    
    // Add color pickers for each color in palette
    Object.keys(paletteObj).forEach(key => {
        paletteFolder.addBinding(paletteObj, key, { label: key.replace('color', 'Color ') });
    });
    
    // Store palette object globally
    window.paletteObj = paletteObj;
    
    // Add a single listener to the folder
    window.handlePaletteFolderChange = (ev) => {
        // The event value from a folder is the complete object
        handlePaletteChange(paletteObj);
    };
    paletteFolder.on('change', window.handlePaletteFolderChange);

    // Show the result pane
    window.resultPane.element.style.display = 'block';
}

// Display vector result
function displayVectorResult(result) {
    elements.resultArea.innerHTML = `
        <div class="result-svg">
            ${result.svg}
        </div>
    `;
    
    elements.downloadBtn.textContent = 'Download SVG';
    elements.actionButtons.style.display = 'flex';
    
    // Store SVG for download
    appState.processedImage[appState.mode] = result;
    
    // Store original SVG for palette changes
    if (appState.processedImage[appState.mode].svg) {
        appState.processedImage[appState.mode].originalSvg = result.svg;
    }
    
    const paletteForTweakpane = result.palette ? result.palette.map(c => ({ R: c.r, G: c.g, B: c.b })) : [];

    // Show palette editor if palette is available
    if (paletteForTweakpane.length > 0) {
        // We also need to store the palette in the same format as the pixel art mode
        const hexPalette = paletteForTweakpane.map(p => rgbToHex({r: p.R, g: p.G, b: p.B}));
        appState.processedImage[appState.mode].palette = hexPalette;

        updatePaletteInTweakpane(paletteForTweakpane);
    } else if (window.resultPane) {
        window.resultPane.element.style.display = 'none';
    }
}

// Handle copy to clipboard
async function handleCopy() {
    if (!appState.processedImage[appState.mode]) return;
    
    try {
        if (appState.mode === 'pixel') {
            await copyPixelArt();
        } else {
            await copyVector();
        }
        
        // Show success feedback
        const originalText = elements.copyBtn.textContent;
        elements.copyBtn.textContent = 'Copied!';
        elements.copyBtn.disabled = true;
        
        setTimeout(() => {
            elements.copyBtn.textContent = originalText;
            elements.copyBtn.disabled = false;
        }, 2000);
        
    } catch (error) {
        console.error('Copy error:', error);
        alert('Failed to copy to clipboard: ' + error.message);
    }
}

// Copy pixel art to clipboard
async function copyPixelArt() {
    const canvas = appState.processedImage[appState.mode].canvas;
    if (!canvas) {
        throw new Error('No image data available');
    }

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    
    // Copy to clipboard using Clipboard API
    if (navigator.clipboard && navigator.clipboard.write) {
        await navigator.clipboard.write([
            new ClipboardItem({
                'image/png': blob
            })
        ]);
    } else {
        throw new Error('Clipboard API not supported');
    }
}

// Copy vector to clipboard
async function copyVector() {
    if (!appState.processedImage[appState.mode].svg) {
        throw new Error('No SVG data available');
    }
    
    // Copy SVG text to clipboard
    if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(appState.processedImage[appState.mode].svg);
    } else {
        throw new Error('Clipboard API not supported');
    }
}

// Handle download
function handleDownload() {
    if (!appState.processedImage[appState.mode]) return;
    
    if (appState.mode === 'pixel') {
        downloadPixelArt();
    } else {
        downloadVector();
    }
}

// Download pixel art as PNG
function downloadPixelArt() {
    const canvas = appState.processedImage[appState.mode].canvas;
    if (!canvas) {
        alert('No image canvas available to download.');
        return;
    }

    const link = document.createElement('a');
    link.download = 'unfake-pixel-art.png';
    
    canvas.toBlob((blob) => {
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
    }, 'image/png');
}

// Download vector as SVG
function downloadVector() {
    const svgBlob = new Blob([appState.processedImage[appState.mode].svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(svgBlob);
    
    const link = document.createElement('a');
    link.download = 'unfake-vector.svg';
    link.href = url;
    link.click();
    
    URL.revokeObjectURL(url);
}

// Set processing state
function setProcessingState(processing) {
    appState.isProcessing = processing;
    
    if (processing) {
        elements.btnText.style.display = 'none';
        elements.btnLoading.style.display = 'inline';
        elements.processBtn.disabled = true;
    } else {
        elements.btnText.style.display = 'inline';
        elements.btnLoading.style.display = 'none';
        elements.processBtn.disabled = false;
    }
}

// Update process button text
function updateProcessButtonText() {
    const modeText = appState.mode === 'pixel' ? 'Pixel Art' : 'Vector';
    elements.btnText.textContent = `✨ Process`;
}

// Update UI based on current state
function updateUI() {
    // Show/hide process button
    if (appState.originalImage && appState.opencvReady) {
        elements.processBtn.style.display = 'block';
    } else {
        elements.processBtn.style.display = 'none';
    }
    
    // Update process button text
    updateProcessButtonText();
    
    // Show/hide action buttons
    if (appState.processedImage[appState.mode]) {
        elements.actionButtons.style.display = 'flex';
    } else {
        elements.actionButtons.style.display = 'none';
    }
    
    // Show OpenCV status
    if (appState.originalImage && !appState.opencvReady) {
        // Show loading indicator
        elements.resultArea.innerHTML = `
            <div class="placeholder">
                <div class="placeholder-icon">⏳</div>
                <p>Loading OpenCV...</p>
                <p class="upload-hint">Please wait while the image processing library loads</p>
            </div>
        `;
    }
}

function clearResultArea() {
    elements.resultArea.innerHTML = `
        <div class="placeholder">
            <div class="placeholder-icon">✨</div>
            <p>Upload an image and click "Process"</p>
        </div>
    `;
    elements.actionButtons.style.display = 'none';
    if (window.resultPane) {
        window.resultPane.element.style.display = 'none';
    }
}

// Make functions globally available
window.appState = appState;
window.updateUI = updateUI; 

function updateMainTitle() {
    const titleEl = document.getElementById('main-title');
    if (!titleEl) return;
    if (appState.mode === 'pixel') {
        titleEl.textContent = 'unfake.js - Pixel Art Mode';
    } else {
        titleEl.textContent = 'unfake.js - Vectorize Mode';
    }
} 