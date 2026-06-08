#!/usr/bin/env node
/**
 * One-shot copy of browser-tool CDN deps into browser-tool/vendor/.
 * Run from repo root: node scripts/vendor-copy.mjs
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const vendorDir = join(root, 'browser-tool', 'vendor');

const files = {
  'image-q.js': 'https://cdn.jsdelivr.net/npm/image-q@4.0.0/dist/esm/image-q.mjs',
  'tweakpane.min.js': 'https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js',
  'svd.js': 'https://cdn.jsdelivr.net/npm/svd-js@1.1.1/build-es/svd.js',
  'pako.min.js': 'https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js',
  'UPNG.js': 'https://cdn.jsdelivr.net/npm/upng-js@2.1.0/UPNG.js',
  'imagetracer_v1.2.6.js': 'https://cdn.jsdelivr.net/npm/imagetracerjs@1.2.6/imagetracer_v1.2.6.js',
  'opencv.js': 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.12.0-release.1/dist/opencv.js',
};

await mkdir(vendorDir, { recursive: true });

for (const [name, url] of Object.entries(files)) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  const body = await res.text();
  await writeFile(join(vendorDir, name), body, 'utf8');
  console.log(`wrote ${name} (${body.length} bytes)`);
}

console.log('Done — commit browser-tool/vendor/ for itch.io deploys.');
