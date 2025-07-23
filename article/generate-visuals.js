import { fileToImageData } from '../browser-tool/lib/utils.js';
import { downscaleByDominantColor } from '../browser-tool/lib/pixel.js';
import { promises as fs } from 'fs';
import { createCanvas, loadImage } from 'canvas';

async function main() {
    console.log('Starting visualization generation...');

    const imagePath = '../demo-pixel.png';
    const outputPath = './images';

    // 1. Load the source image
    const imageBuffer = await fs.readFile(imagePath);
    const image = await loadImage(imageBuffer);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height);

    console.log(`Loaded image: ${image.width}x${image.height}`);

    const scale = 8; // The size of our blocks for the visualization

    // Visualization 1: Grid overlay
    await generateGridVisualization(image, scale, `${outputPath}/vis_01_grid.png`);


    // TODO: Add more visualization generation steps here

    console.log('Visualization generation complete.');
}

async function generateGridVisualization(image, scale, outputImagePath) {
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);

    ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)';
    ctx.lineWidth = 1;

    for (let x = 0; x <= image.width; x += scale) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, image.height);
        ctx.stroke();
    }

    for (let y = 0; y <= image.height; y += scale) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(image.width, y);
        ctx.stroke();
    }

    const buffer = canvas.toBuffer('image/png');
    await fs.writeFile(outputImagePath, buffer);
    console.log(`Generated: ${outputImagePath}`);
}

main().catch(console.error); 