// unfake.js Browser Tool
// Main application file

// Enable debug logging
window.DEBUG_PIXEL_PROCESSOR = true;

// Import dependencies
import unfake from '../../lib/index.js';
import { Pane } from 'tweakpane';

// Global state
let appState = {
    mode: 'pixel', // 'pixel' or 'vector'
    originalImage: null,
    processedImage: null,
    isProcessing: false,
    opencvReady: false
};

// DOM elements
let elements = {};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

// Initialize OpenCV (this will be called by the script tag)
window.onOpenCvReady = function() {
    console.log('OpenCV.js is ready');
    if (window.appState) {
        window.appState.opencvReady = true;
        if (window.updateUI) window.updateUI();
    }
};

// Check OpenCV status periodically
function checkOpenCVStatus() {
    if (typeof cv !== 'undefined' && cv.getBuildInformation) {
        console.log('OpenCV is ready');
        appState.opencvReady = true;
        updateUI();
        return true;
    }
    return false;
}

// Start checking OpenCV status
setInterval(checkOpenCVStatus, 1000);

// Main initialization
function initializeApp() {
    console.log('Initializing unfake.js browser tool...');
    
    // Cache DOM elements
    cacheElements();
    
    // Initialize UI components
    initializeUI();
    
    // Bind event listeners
    bindEvents();
    
    // Update initial UI state
    updateUI();
}

// Cache DOM elements for performance
function cacheElements() {
    elements = {
        // Mode switching
        modeBtns: document.querySelectorAll('.mode-btn'),
        themeToggle: document.getElementById('theme-toggle'),
        
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
        downscaleMethod: 'median',
        autoPixelSize: true,
        pixelSize: 4,
        alphaThreshold: 128,
        enableAlphaBinarization: true,
        
        // Vector settings
        filterStrength: 31,
        fillHoles: true,
        kernelSize: 3,
        turdSize: 'auto',
        turdSizeManual: 10,
        optimizeCurves: true,
        
        // Palette (will be populated dynamically)
        palette: []
    };
    
    // Pixel Art folder
    const pixelFolder = pixelPane.addFolder({
        title: 'Pixel Art Settings',
        expanded: true
    });
    
    pixelFolder.addBinding(settings, 'maxColors', {
        label: 'Max Colors',
        min: 2,
        max: 64,
        step: 1
    });
    
    pixelFolder.addBinding(settings, 'snapGrid', {
        label: 'Snap to Grid'
    });
    
    pixelFolder.addBinding(settings, 'downscaleMethod', {
        label: 'Downscale Method',
        options: {
            'nearest': 'nearest',
            'median': 'median',
            'mode': 'mode', 
            'mean': 'mean',
            'contrast': 'contrast',
            'content-adaptive (slow)': 'content-adaptive',
        }
    });
    
    // Auto pixel size checkbox
    const autoPixelSizeBinding = pixelFolder.addBinding(settings, 'autoPixelSize', {
        label: 'Auto pixel size'
    });
    
    // Manual pixel size slider (initially hidden)
    const pixelSizeBinding = pixelFolder.addBinding(settings, 'pixelSize', {
        label: 'Pixel Size',
        min: 1,
        max: 16,
        step: 1
    });
    
    // Hide/show pixel size slider based on auto checkbox
    pixelSizeBinding.hidden = settings.autoPixelSize;
    
    // Update visibility when auto checkbox changes
    autoPixelSizeBinding.on('change', (ev) => {
        pixelSizeBinding.hidden = ev.value;
    });
    
    // Alpha threshold slider (initially hidden)
    const alphaThresholdBinding = pixelFolder.addBinding(settings, 'alphaThreshold', {
        label: 'Alpha Threshold',
        min: 0,
        max: 255,
        step: 1
    });
    
    // Enable alpha binarization checkbox
    const enableAlphaBinarizationBinding = pixelFolder.addBinding(settings, 'enableAlphaBinarization', {
        label: 'Enable Alpha Binarization'
    });
    
    // Hide/show alpha threshold slider based on checkbox
    alphaThresholdBinding.hidden = !settings.enableAlphaBinarization;
    
    // Update visibility when checkbox changes
    enableAlphaBinarizationBinding.on('change', (ev) => {
        alphaThresholdBinding.hidden = !ev.value;
        if (!ev.value) {
            settings.alphaThreshold = 0; // Disable binarization
        } else if (settings.alphaThreshold === 0) {
            settings.alphaThreshold = 128; // Enable with default value
        }
    });
    
    // Vector folder
    const vectorFolder = vectorPane.addFolder({
        title: 'Vector Settings',
        expanded: true
    });
    
    vectorFolder.addBinding(settings, 'filterStrength', {
        label: 'Filter Strength',
        min: 5,
        max: 100,
        step: 1
    });
    
    vectorFolder.addBinding(settings, 'fillHoles', {
        label: 'Fill Holes'
    });
    
    vectorFolder.addBinding(settings, 'kernelSize', {
        label: 'Kernel Size',
        min: 3,
        max: 15,
        step: 2
    });
    
    vectorFolder.addBinding(settings, 'turdSize', {
        label: 'Turd Size',
        options: {
            'auto': 'auto',
            'manual': 'manual'
        }
    });
    
    vectorFolder.addBinding(settings, 'turdSizeManual', {
        label: 'Manual Turd Size',
        min: 1,
        max: 50,
        step: 1
    });
    
    vectorFolder.addBinding(settings, 'optimizeCurves', {
        label: 'Optimize Curves'
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
}

// Handle mode switching
function handleModeSwitch(e) {
    const newMode = e.target.dataset.mode;
    if (newMode === appState.mode) return;
    
    // Update active button
    elements.modeBtns.forEach(btn => btn.classList.remove('active'));
    e.target.classList.add('active');
    
    // Update mode
    appState.mode = newMode;
    
    // Update UI
    updateProcessButtonText();
    updateSettingsVisibility();
    
    // Reprocess if we have an image
    if (appState.originalImage) {
        handleProcess();
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
}

// Handle reset
function handleReset() {
    appState.originalImage = null;
    appState.originalFile = null;
    appState.processedImage = null;
    
    elements.uploadArea.style.display = 'block';
    elements.imageDisplay.style.display = 'none';
    elements.processBtn.style.display = 'none';
    elements.actionButtons.style.display = 'none';
    
    // Clear result area
    elements.resultArea.innerHTML = `
        <div class="placeholder">
            <div class="placeholder-icon">✨</div>
            <p>Upload an image and click "Process"</p>
        </div>
    `;
    
    // Clear original image size info
    const originalSizeInfo = elements.imageDisplay.querySelector('.image-size-info');
    if (originalSizeInfo) {
        originalSizeInfo.remove();
    }
    
    // Hide result Tweakpane
    if (window.resultPane) {
        window.resultPane.element.style.display = 'none';
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
        displayResult(result);
        updateUI();
    } catch (error) {
        console.error('Processing error:', error);
        
        // More specific error messages
        let errorMessage = 'Error processing image: ';
        if (error.message.includes('OpenCV')) {
            errorMessage += 'OpenCV library error. Please refresh the page and try again.';
        } else if (error.message.includes('timeout')) {
            errorMessage += 'Processing timed out. Try with a smaller image.';
        } else if (error.message.includes('memory')) {
            errorMessage += 'Not enough memory. Try with a smaller image.';
        } else {
            errorMessage += error.message;
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
            console.log('Manual scale value:', manualScale);
            return await unfake.processImage({
                file: appState.originalFile,
                maxColors: settings.maxColors,
                snapGrid: settings.snapGrid,
                downscaleMethod: settings.downscaleMethod,
                manualScale: manualScale,
                alphaThreshold: settings.alphaThreshold,
                cleanup: true,
                cleanupStrength: 10
            });
        } else {
            console.log('Processing vector with settings:', settings);
            return await unfake.vectorizeImage({
                file: appState.originalFile,
                numColors: 'auto',
                turdSize: settings.turdSize === 'auto' ? 'auto' : settings.turdSizeManual,
                opticurve: settings.optimizeCurves,
                preProcess: {
                    enabled: true,
                    filter: 'bilateral',
                    value: settings.filterStrength,
                    morphology: settings.fillHoles,
                    morphologyKernel: settings.kernelSize
                }
            });
        }
    } catch (error) {
        console.error('Error in processImage:', error);
        throw error;
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
    console.log('Displaying pixel art result:', result);
    console.log('PNG buffer size:', result.png.byteLength);
    console.log('Palette size:', result.palette.length);
    
    // Convert PNG buffer to data URL for display
    const blob = new Blob([result.png], { type: 'image/png' });
    const dataUrl = URL.createObjectURL(blob);
    console.log('Created data URL:', dataUrl.substring(0, 50) + '...');
    
    // Load image to get dimensions for display
    const img = new Image();
    img.onload = () => {
        // Get the visual size of the original image in the interface
        const originalImgElement = elements.originalImage;
        const visualOriginalWidth = originalImgElement.offsetWidth;
        const visualOriginalHeight = originalImgElement.offsetHeight;
        
        // Calculate scale factor to match the visual size of the original
        const scaleFactor = Math.max(visualOriginalWidth / img.width, visualOriginalHeight / img.height);
        
        // Calculate pixel size from scale detection
        const pixelSize = Math.round(visualOriginalWidth / img.width);
        
        const html = `
            <div class="result-image">
                <img src="${dataUrl}" alt="Pixel art result" 
                     style="width: ${img.width * scaleFactor}px; height: ${img.height * scaleFactor}px; image-rendering: pixelated; image-rendering: -moz-crisp-edges; image-rendering: crisp-edges;"
                     onload="console.log('Original image loaded successfully')" 
                     onerror="console.error('Original image failed to load')">
            </div>
            <div class="image-size-info">
                ${img.width} × ${img.height} pixels (Pixel size: ${pixelSize}×${pixelSize}px)
            </div>
        `;
        
        console.log('Setting innerHTML with original image (CSS scaled)');
        elements.resultArea.innerHTML = html;
    };
    img.src = dataUrl;
    
    elements.downloadBtn.textContent = 'Download PNG';
    elements.actionButtons.style.display = 'flex';
    
    // Store original PNG buffer and data URL for download/copy
    appState.processedImage = { 
        ...result, 
        dataUrl,
        originalPngBuffer: result.png // Store original buffer for copying
    };
    
    // Update palette in Result Tweakpane
    updatePaletteInTweakpane(result.palette);
    
    console.log('Result displayed successfully');
}



// Update palette in Result Tweakpane
function updatePaletteInTweakpane(palette) {
    if (!window.paletteFolder || !window.resultSettings) return;
    
    // Clear existing palette bindings
    window.paletteFolder.children.forEach(child => {
        window.paletteFolder.remove(child);
    });
    
    // Create palette object with color properties
    const paletteObj = {};
    palette.forEach((color, index) => {
        paletteObj[`color${index + 1}`] = color;
    });
    
    // Add color pickers for each color in palette
    palette.forEach((color, index) => {
        const colorKey = `color${index + 1}`;
        console.log('Color:', color);
        paletteObj[colorKey] = color;
        
        window.paletteFolder.addBinding(paletteObj, colorKey, {
            label: `Color ${index + 1}`
        });
    });
    
    // Store palette object globally
    window.paletteObj = paletteObj;
    
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
    appState.processedImage = result;
}

// Handle copy to clipboard
async function handleCopy() {
    if (!appState.processedImage) return;
    
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
    if (!appState.processedImage.originalPngBuffer) {
        throw new Error('No image data available');
    }
    
    // Create blob from original PNG buffer (original size, not CSS scaled)
    const blob = new Blob([appState.processedImage.originalPngBuffer], { type: 'image/png' });
    
    // Copy to clipboard using Clipboard API
    if (navigator.clipboard && navigator.clipboard.write) {
        await navigator.clipboard.write([
            new ClipboardItem({
                'image/png': blob
            })
        ]);
    } else {
        // Fallback: try to copy the displayed image element
        const imgElement = elements.resultArea.querySelector('img');
        if (imgElement) {
            try {
                // Create canvas with original dimensions
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                // Get original dimensions from the image
                const originalImg = new Image();
                originalImg.crossOrigin = 'anonymous';
                
                await new Promise((resolve, reject) => {
                    originalImg.onload = resolve;
                    originalImg.onerror = reject;
                    originalImg.src = appState.processedImage.dataUrl;
                });
                
                canvas.width = originalImg.width;
                canvas.height = originalImg.height;
                ctx.drawImage(originalImg, 0, 0);
                
                canvas.toBlob(async (blob) => {
                    await navigator.clipboard.write([
                        new ClipboardItem({
                            'image/png': blob
                        })
                    ]);
                }, 'image/png');
                
            } catch (error) {
                throw new Error('Failed to copy image: ' + error.message);
            }
        } else {
            throw new Error('No image element found');
        }
    }
}

// Copy vector to clipboard
async function copyVector() {
    if (!appState.processedImage.svg) {
        throw new Error('No SVG data available');
    }
    
    // Copy SVG text to clipboard
    if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(appState.processedImage.svg);
    } else {
        throw new Error('Clipboard API not supported');
    }
}

// Handle download
function handleDownload() {
    if (!appState.processedImage) return;
    
    if (appState.mode === 'pixel') {
        downloadPixelArt();
    } else {
        downloadVector();
    }
}

// Download pixel art as PNG
function downloadPixelArt() {
    const link = document.createElement('a');
    link.download = 'unfake-pixel-art.png';
    
    // Use original PNG buffer for download (not upscaled)
    if (appState.processedImage.originalPngBuffer) {
        const blob = new Blob([appState.processedImage.originalPngBuffer], { type: 'image/png' });
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
    } else {
        // Fallback to dataUrl if original buffer not available
        link.href = appState.processedImage.dataUrl;
        link.click();
    }
}

// Download vector as SVG
function downloadVector() {
    const svgBlob = new Blob([appState.processedImage.svg], { type: 'image/svg+xml' });
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
    elements.btnText.textContent = `Process → ${modeText}`;
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
    if (appState.processedImage) {
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

// Make functions globally available
window.appState = appState;
window.updateUI = updateUI; 