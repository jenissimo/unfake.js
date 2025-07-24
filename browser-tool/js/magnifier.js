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
        if (!this.isActive) return;
        
        // Find image under cursor
        const imageUnderCursor = this.findImageUnderCursor(e.clientX, e.clientY);
        
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
        
        this.mouseX = e.clientX;
        this.mouseY = e.clientY;
        
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
    
    // Update magnified content
    updateMagnification() {
        if (!this.targetImage || this.isDestroyed) return;

        const targetRect = this.targetImage.getBoundingClientRect();
        const tagName = this.targetImage.tagName.toLowerCase();
        const isSVG = tagName === 'svg';
        const zoom = this.options.zoomLevel;

        // Check if mouse is within the target element's bounds
        if (!this.isPointInElement(this.mouseX, this.mouseY, this.targetImage)) {
            this.element.style.display = 'none';
            return;
        }

        this.element.style.display = 'block';

        const setBackground = (imageElement, srcOverride = null) => {
            if (!imageElement && !srcOverride) return;

            // For IMG elements, naturalWidth is on the element itself.
            // For SVGs converted to Image, it's on the new Image object.
            const naturalWidth = imageElement?.naturalWidth;
            const naturalHeight = imageElement?.naturalHeight;

            if (!srcOverride && (!naturalWidth || !naturalHeight)) return;

            const bgSizeX = targetRect.width * zoom;
            const bgSizeY = targetRect.height * zoom;
            this.element.style.backgroundSize = `${bgSizeX}px ${bgSizeY}px`;
            
            this.element.style.imageRendering = this.options.crispPixels ? 'pixelated' : 'auto';

            const cursorX_rel = this.mouseX - targetRect.left;
            const cursorY_rel = this.mouseY - targetRect.top;
            
            const bgPosX = -(cursorX_rel * zoom - this.options.size / 2);
            const bgPosY = -(cursorY_rel * zoom - this.options.size / 2);

            this.element.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;
            
            const targetSrc = srcOverride || imageElement.src;
            if (targetSrc && this.element.style.backgroundImage !== `url("${targetSrc}")`) {
                this.element.style.backgroundImage = `url("${targetSrc}")`;
            }
        };

        if (isSVG) {
            let cachedImg = this.svgImageCache.get(this.targetImage);
            if (cachedImg?.complete && cachedImg.naturalWidth > 0) {
                setBackground(cachedImg);
            } else if (!cachedImg || !cachedImg.loading) { // Check custom flag
                const svgString = new XMLSerializer().serializeToString(this.targetImage);
                const svgDataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgString)}`;
                
                cachedImg = cachedImg || new Image();
                cachedImg.loading = true; // Set flag
                this.svgImageCache.set(this.targetImage, cachedImg);
                
                const currentTarget = this.targetImage;
                cachedImg.onload = () => {
                    cachedImg.loading = false;
                    if (!this.isDestroyed && this.targetImage === currentTarget) {
                        this.updateMagnification();
                    }
                }
                cachedImg.src = svgDataUrl;
            }
        } else { // Handles <img> and <canvas>
            let dataURL = this.imgDataURLCache.get(this.targetImage);
            if (dataURL) {
                setBackground(null, dataURL);
            } else {
                const isImg = tagName === 'img';
                const isReady = isImg ? (this.targetImage.complete && this.targetImage.naturalWidth > 0) : true; // Canvas is always "ready"

                if (isReady) {
                    const canvas = document.createElement('canvas');
                    const sourceWidth = isImg ? this.targetImage.naturalWidth : this.targetImage.width;
                    const sourceHeight = isImg ? this.targetImage.naturalHeight : this.targetImage.height;

                    if (sourceWidth === 0 || sourceHeight === 0) return;

                    canvas.width = sourceWidth;
                    canvas.height = sourceHeight;
                    
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(this.targetImage, 0, 0);
                    
                    try {
                        dataURL = canvas.toDataURL();
                        this.imgDataURLCache.set(this.targetImage, dataURL);
                        setBackground(null, dataURL);
                    } catch (e) {
                        console.error("Magnifier: Could not generate data URL, falling back to src.", e);
                        if (isImg) {
                            setBackground(this.targetImage); // Fallback for tainted <img>
                        }
                    }
                } else if (isImg) {
                    // Image is not loaded yet, use src and wait for onload.
                    this.targetImage.onload = () => {
                        if(!this.isDestroyed) {
                            this.updateMagnification();
                        }
                    };
                    setBackground(this.targetImage);
                }
            }
        }
        
        this.updatePosition();
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