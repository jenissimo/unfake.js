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

export async function ensurePakoAndUPNG() {
  if (typeof window !== 'undefined' && window.UPNG?.encode) return;
  await loadClassicScript(vendorScript('pako.min.js'));
  await loadClassicScript(vendorScript('UPNG.js'));
}

export async function ensureImageTracer() {
  if (typeof window !== 'undefined' && window.ImageTracer) return;
  await ensurePakoAndUPNG();
  await loadClassicScript(vendorScript('imagetracer_v1.2.6.js'));
}

/** OpenCV.js is a classic UMD/Emscripten bundle (~10 MB) — must NOT be loaded via import(). */
const OPENCV_CDN =
  'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.12.0-release.1/dist/opencv.js';

let cvInitPromise = null;
let opencvWasmReady = false;

/** Let the status overlay paint before the main-thread compile freeze. */
function yieldToBrowser() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(resolve, 50)));
  });
}

function hookCvReadyCallback(cv, onReady) {
  if (!cv || typeof onReady !== 'function') return;
  const prev = cv.onRuntimeInitialized;
  cv.onRuntimeInitialized = () => {
    if (typeof prev === 'function') prev();
    onReady();
  };
}

function markOpenCvReady() {
  opencvWasmReady = true;
}

function isCvRuntimeReady() {
  return opencvWasmReady;
}

/** One-shot sanity check after WASM reports ready — never call in a poll loop. */
function verifyCvRuntime() {
  const cv = window.cv;
  if (!cv?.Mat || typeof cv.getBuildInformation !== 'function') {
    throw new Error('OpenCV module loaded but cv object is invalid.');
  }
  const m = new cv.Mat(1, 1, cv.CV_8UC1);
  m.delete();
}

function waitForCvRuntime(timeoutMs) {
  return new Promise((resolve, reject) => {
    if (isCvRuntimeReady()) {
      resolve(window.cv);
      return;
    }

    let settled = false;
    const done = (ok, err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearInterval(poll);
      if (ok) resolve(window.cv);
      else reject(err);
    };

    const onReady = () => {
      try {
        markOpenCvReady();
        verifyCvRuntime();
        done(true);
      } catch (e) {
        done(false, e);
      }
    };

    const timer = setTimeout(
      () => done(false, new Error(`OpenCV failed to initialize within ${timeoutMs / 1000} seconds.`)),
      timeoutMs,
    );

    if (window.cv) hookCvReadyCallback(window.cv, onReady);

    // Lightweight backup poll — do NOT allocate Mat here (that freezes the tab).
    const poll = setInterval(() => {
      if (isCvRuntimeReady()) onReady();
    }, 500);
  });
}

function loadOpenCVScript(src, timeoutMs) {
  if (document.querySelector(`script[src="${src}"]`)) {
    return waitForCvRuntime(timeoutMs);
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    const done = (ok, err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearInterval(poll);
      if (ok) resolve(window.cv);
      else reject(err);
    };

    const onReady = () => {
      try {
        markOpenCvReady();
        verifyCvRuntime();
        done(true);
      } catch (e) {
        done(false, e);
      }
    };

    const timer = setTimeout(
      () => done(false, new Error(`OpenCV failed to initialize within ${timeoutMs / 1000} seconds.`)),
      timeoutMs,
    );

    const prevModule = window.Module;
    window.Module = {
      ...(prevModule || {}),
      onRuntimeInitialized() {
        if (typeof prevModule?.onRuntimeInitialized === 'function') {
          prevModule.onRuntimeInitialized();
        }
        onReady();
      },
    };

    const poll = setInterval(() => {
      if (isCvRuntimeReady()) onReady();
    }, 500);

    const el = document.createElement('script');
    el.src = src;
    el.async = true;
    el.onload = () => {
      loaded.add(src);
      if (window.cv) hookCvReadyCallback(window.cv, onReady);
    };
    el.onerror = () => done(false, new Error(`Failed to load script: ${src}`));
    document.head.appendChild(el);
  });
}

/**
 * Lazy-load OpenCV (classic script → window.cv).
 * @param {{ timeoutMs?: number }} [opts]
 * @returns {Promise<object>}
 */
export async function ensureOpenCV({ timeoutMs = 180_000 } = {}) {
  if (typeof window !== 'undefined' && isCvRuntimeReady()) {
    return window.cv;
  }

  if (!cvInitPromise) {
    const vendorUrl = vendorScript('opencv.js');
    cvInitPromise = (async () => {
      await yieldToBrowser();

      let lastErr;
      for (const src of [vendorUrl, OPENCV_CDN]) {
        try {
          const cv = await loadOpenCVScript(src, timeoutMs);
          return cv;
        } catch (err) {
          lastErr = err;
          opencvWasmReady = false;
          if (src === vendorUrl) continue;
          throw err;
        }
      }
      throw lastErr || new Error('OpenCV failed to load.');
    })().catch((err) => {
      cvInitPromise = null;
      opencvWasmReady = false;
      loaded.delete(vendorUrl);
      loaded.delete(OPENCV_CDN);
      throw err;
    });
  }

  return cvInitPromise;
}
