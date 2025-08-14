// magnifier.js – Magnifier tool for detailed image comparison
// Provides zoom functionality for pixel-perfect analysis

class Magnifier {
    constructor(options = {}) {
        this.options = {
            zoomLevel: 4,
            size: 200,
            borderWidth: 2,
            borderColor: '#333',
            backgroundColor: '#fff',
            showCrosshair: true,
            crosshairColor: '#ff0000',
            crosshairWidth: 1,
            crispPixels: true,
            debug: false,
            ...options
        };
        
        this.isActive = false;
        this.element = null;
        this.crosshair = null;
        this.targetImage = null;
        this.mouseX = 0;
        this.mouseY = 0;
        
        this.svgImageCache = new WeakMap();
        this.imgDataURLCache = new WeakMap();
        this.isDestroyed = false;
        this.rafScheduled = false;
        this.isUpdating = false;
        
        this.init();
    }
    
    init() {
        // Create magnifier container
        this.element = document.createElement('div');
        this.element.className = 'magnifier';
        this.element.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            width: ${this.options.size}px;
            height: ${this.options.size}px;
            border: ${this.options.borderWidth}px solid ${this.options.borderColor};
            border-radius: 50%;
            background: ${this.options.backgroundColor};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            pointer-events: none;
            z-index: 1000;
            display: none;
            overflow: hidden;
            background-repeat: no-repeat;
        `;
        
        // Create crosshair
        if (this.options.showCrosshair) {
            this.crosshair = document.createElement('div');
            this.crosshair.className = 'magnifier-crosshair';
            this.crosshair.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                width: ${this.options.size}px;
                height: ${this.options.size}px;
                pointer-events: none;
                transform: translate(-50%, -50%);
            `;
            this.element.appendChild(this.crosshair);

            const style = document.createElement('style');
            style.innerHTML = `
                .magnifier-crosshair::before, .magnifier-crosshair::after {
                    content: '';
                    position: absolute;
                    background-color: ${this.options.crosshairColor};
                    z-index: 1;
                }
                .magnifier-crosshair::before {
                    top: 50%; left: 0; width: 100%; height: ${this.options.crosshairWidth}px;
                    transform: translateY(-50%);
                }
                .magnifier-crosshair::after {
                    left: 50%; top: 0; height: 100%; width: ${this.options.crosshairWidth}px;
                    transform: translateX(-50%);
                }
            `;
            document.head.appendChild(style);
        }
        
        // Add to document
        document.body.appendChild(this.element);
    }
    
    // Activate magnifier
    activate() {
        this.isActive = true;
        this.element.style.display = 'block';
        
        // Add mouse move listener
        this.boundMouseMove = this.handleMouseMove.bind(this);
        document.addEventListener('mousemove', this.boundMouseMove);
        
        console.log('Magnifier activated');
    }
    
    // Deactivate magnifier
    deactivate() {
        this.isActive = false;
        this.element.style.display = 'none';
        
        // Remove event listeners
        if (this.boundMouseMove) {
            document.removeEventListener('mousemove', this.boundMouseMove);
            this.boundMouseMove = null;
        }
        
        console.log('Magnifier deactivated');
    }
    
    // Handle mouse movement
    handleMouseMove(e) {
        if (!this.isActive || this.isUpdating) return;

        this.mouseX = e.clientX;
        this.mouseY = e.clientY;

        // Use rAF to avoid performance issues
        if (!this.rafScheduled) {
            this.rafScheduled = true;
            window.requestAnimationFrame(() => {
                this.updateOnMouseMove();
                this.rafScheduled = false;
            });
        }
    }

    updateOnMouseMove() {
        // Find image under cursor
        const imageUnderCursor = this.findImageUnderCursor(this.mouseX, this.mouseY);
        
        if (!imageUnderCursor) {
            // No image under cursor, hide magnifier
            this.element.style.display = 'none';
            return;
        }
        
        // Update target image if changed
        if (this.targetImage !== imageUnderCursor) {
            this.targetImage = imageUnderCursor;
            if (this.options.debug) {
                console.log('Magnifier switched to new image');
            }
        }
        
        // Show magnifier if it was hidden
        this.element.style.display = 'block';
        
        this.updateMagnification();
    }
    
    // Find image element under cursor
    findImageUnderCursor(x, y) {
        // Get all images and SVGs in the app
        const originalImage = document.getElementById('original-image');
        const resultImages = document.querySelectorAll('#result-area img, #result-area svg, #result-area canvas');
        
        // Check original image first
        if (originalImage && this.isPointInElement(x, y, originalImage)) {
            return originalImage;
        }
        
        // Check result images
        for (const img of resultImages) {
            if (this.isPointInElement(x, y, img)) {
                return img;
            }
        }
        
        return null;
    }
    
    // Check if point is inside element
    isPointInElement(x, y, element) {
        const rect = element.getBoundingClientRect();
        return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
    }
    
    // Update crosshair visibility
    updateCrosshair() {
        if (this.crosshair) {
            this.crosshair.style.display = this.options.showCrosshair ? 'block' : 'none';
        }
    }

    // Update magnified content
    updateMagnification() {
        if (!this.targetImage || this.isDestroyed || this.isUpdating) return;

        const setBackgroundPosition = () => {
            const targetRect = this.targetImage.getBoundingClientRect();
            const zoom = this.options.zoomLevel;

            const cursorX_rel = this.mouseX - targetRect.left;
            const cursorY_rel = this.mouseY - targetRect.top;

            const bgPosX = -(cursorX_rel * zoom - this.options.size / 2);
            const bgPosY = -(cursorY_rel * zoom - this.options.size / 2);

            this.element.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;
            this.updatePosition();
        };

        // If data URL is already cached, just update the position
        if (this.imgDataURLCache.has(this.targetImage) || this.svgImageCache.has(this.targetImage)) {
            setBackgroundPosition();
            return;
        }

        // --- Data URL generation (only runs once per image) ---
        this.isUpdating = true;
        this.element.style.cursor = 'wait';

        const tagName = this.targetImage.tagName.toLowerCase();
        const isSVG = tagName === 'svg';
        const zoom = this.options.zoomLevel;
        const targetRect = this.targetImage.getBoundingClientRect();

        const setBackground = (imageElement, srcOverride = null) => {
            if (!imageElement && !srcOverride) {
                this.isUpdating = false;
                this.element.style.cursor = 'default';
                return;
            }

            const bgSizeX = targetRect.width * zoom;
            const bgSizeY = targetRect.height * zoom;
            this.element.style.backgroundSize = `${bgSizeX}px ${bgSizeY}px`;
            
            this.element.style.imageRendering = this.options.crispPixels ? 'pixelated' : 'auto';
            
            const targetSrc = srcOverride || imageElement.src;
            if (targetSrc && this.element.style.backgroundImage !== `url("${targetSrc}")`) {
                this.element.style.backgroundImage = `url("${targetSrc}")`;
            }
            
            setBackgroundPosition();
            this.isUpdating = false;
            this.element.style.cursor = 'default';
        };

        if (isSVG) {
            let cachedImg = new Image();
            cachedImg.loading = true;
            this.svgImageCache.set(this.targetImage, cachedImg);
            
            const svgString = new XMLSerializer().serializeToString(this.targetImage);
            const svgDataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgString)}`;
            
            cachedImg.onload = () => {
                setBackground(cachedImg);
            }
            cachedImg.src = svgDataUrl;

        } else { // Handles <img> and <canvas>
            const isImg = tagName === 'img';
            const isReady = isImg ? (this.targetImage.complete && this.targetImage.naturalWidth > 0) : true;

            if (isReady) {
                const canvas = document.createElement('canvas');
                const sourceWidth = isImg ? this.targetImage.naturalWidth : this.targetImage.width;
                const sourceHeight = isImg ? this.targetImage.naturalHeight : this.targetImage.height;

                if (sourceWidth === 0 || sourceHeight === 0) {
                    this.isUpdating = false;
                    this.element.style.cursor = 'default';
                    return;
                }

                canvas.width = sourceWidth;
                canvas.height = sourceHeight;
                
                const ctx = canvas.getContext('2d');
                ctx.drawImage(this.targetImage, 0, 0);
                
                try {
                    const dataURL = canvas.toDataURL();
                    this.imgDataURLCache.set(this.targetImage, dataURL);
                    setBackground(null, dataURL);
                } catch (e) {
                    console.error("Magnifier: Could not generate data URL, falling back to src.", e);
                    if (isImg) {
                        setBackground(this.targetImage); // Fallback for tainted <img>
                    }
                    this.isUpdating = false;
                    this.element.style.cursor = 'default';
                }
            } else if (isImg) {
                this.targetImage.onload = () => {
                    if(!this.isDestroyed) {
                        this.isUpdating = false;
                        this.updateMagnification();
                    }
                };
            }
        }
    }
    
    // Clear cache for a specific image element
    clearCacheForElement(element) {
        if (!element) return;
        if (this.svgImageCache && this.svgImageCache.has(element)) {
            this.svgImageCache.delete(element);
            if (this.options.debug) console.log('Magnifier: Cleared SVG cache for element');
        }
        if (this.imgDataURLCache && this.imgDataURLCache.has(element)) {
            this.imgDataURLCache.delete(element);
            if (this.options.debug) console.log('Magnifier: Cleared DataURL cache for element');
        }
    }

    // Draw crosshair - This is now handled by CSS
    
    // Update magnifier position
    updatePosition() {
        const size = this.options.size;
        const offset = 20;
        
        // Position magnifier near mouse but keep it in viewport
        let left = this.mouseX + offset;
        let top = this.mouseY + offset;
        
        // Adjust if magnifier would go off screen
        if (left + size > window.innerWidth) {
            left = this.mouseX - size - offset;
        }
        if (top + size > window.innerHeight) {
            top = this.mouseY - size - offset;
        }
        
        this.element.style.left = `${left}px`;
        this.element.style.top = `${top}px`;
    }
    
    // Set zoom level
    setZoomLevel(level) {
        this.options.zoomLevel = Math.max(1, Math.min(16, level));
        if (this.isActive) {
            this.updateMagnification();
        }
    }
    
    // Set size
    setSize(size) {
        this.options.size = size;
        this.element.style.width = `${size}px`;
        this.element.style.height = `${size}px`;
        if (this.crosshair) {
            this.crosshair.style.width = `${size}px`;
            this.crosshair.style.height = `${size}px`;
        }
    }
    
    // Toggle crosshair
    toggleCrosshair() {
        this.options.showCrosshair = !this.options.showCrosshair;
        if (this.crosshair) {
            this.crosshair.style.display = this.options.showCrosshair ? 'block' : 'none';
        }
    }
    
    // Destroy magnifier
    destroy() {
        this.isDestroyed = true;
        this.deactivate();
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
        this.element = null;
        this.crosshair = null;
        this.targetImage = null;
        this.svgImageCache = null;
        this.imgDataURLCache = null;
    }
}

// Export for use in other modules
export default Magnifier; 