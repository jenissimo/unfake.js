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
  'tweakpane.min.js': 'https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js',
  'imagetracer_v1.2.6.js': 'https://cdn.jsdelivr.net/npm/imagetracerjs@1.2.6/imagetracer_v1.2.6.js',
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
