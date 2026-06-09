/** Lazy-load classic (non-module) CDN scripts so the HTML shell paints immediately. */

const loaded = new Set();

export function loadClassicScript(src) {
  if (loaded.has(src)) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src;
    el.async = true;
    el.onload = () => { loaded.add(src); resolve(); };
    el.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(el);
  });
}

function vendorScript(file) {
  return new URL(`../vendor/${file}`, import.meta.url).href;
}

export async function ensureImageTracer() {
  if (typeof window !== 'undefined' && window.ImageTracer) return;
  await loadClassicScript(vendorScript('imagetracer_v1.2.6.js'));
}

