/**
 * Main-thread facade for OpenCV running in a dedicated Web Worker.
 * Keeps WASM compile off the UI thread.
 */

import { ensureOpenCV as ensureOpenCVMainThread } from './scripts.js';

const OPENCV_CDN =
  'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.12.0-release.1/dist/opencv.js';

let worker = null;
let workerReady = null;
let rpcId = 0;
const pending = new Map();
let useMainThreadFallback = false;

function workerUrl(file) {
  return new URL(file, import.meta.url).href;
}

export function imageDataToPayload(img) {
  if (!img) return null;
  const copy = new Uint8ClampedArray(img.data);
  return { width: img.width, height: img.height, buffer: copy.buffer };
}

export function payloadToImageData(payload) {
  if (!payload) return null;
  const expected = payload.width * payload.height * 4;
  const bytes = new Uint8ClampedArray(payload.buffer, 0, expected);
  return new ImageData(bytes, payload.width, payload.height);
}

function serializePayload(payload) {
  if (!payload) return payload;
  const out = { ...payload };
  if (payload.image instanceof ImageData) {
    out.image = imageDataToPayload(payload.image);
  }
  return out;
}

function collectTransfers(serialized) {
  const transfers = [];
  if (serialized?.image?.buffer) {
    transfers.push(serialized.image.buffer);
  }
  return transfers;
}

function handleWorkerMessage(e) {
  const msg = e.data;
  if (msg.type === 'ready') {
    workerReady?.resolve();
    workerReady = null;
    return;
  }
  if (msg.type === 'init-error') {
    workerReady?.reject(new Error(msg.error || 'OpenCV worker init failed'));
    workerReady = null;
    return;
  }
  if (msg.type === 'rpc-result') {
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.ok) {
      entry.resolve(msg.result);
    } else {
      entry.reject(new Error(msg.error || 'OpenCV worker RPC failed'));
    }
  }
}

function spawnWorker() {
  const w = new Worker(workerUrl('../workers/opencv-worker.js'));
  w.onmessage = handleWorkerMessage;
  w.onerror = (err) => {
    workerReady?.reject(err.error || new Error('OpenCV worker error'));
    workerReady = null;
  };
  return w;
}

async function initWorker() {
  const vendorOpencv = workerUrl('../vendor/opencv.js');
  let opencvUrl = OPENCV_CDN;
  try {
    const head = await fetch(vendorOpencv, { method: 'HEAD' });
    if (head.ok) opencvUrl = vendorOpencv;
  } catch {
    /* CDN fallback */
  }

  worker.postMessage({
    type: 'init',
    svdUrl: workerUrl('../vendor/svd-worker.js'),
    opencvUrl,
    opsUrl: workerUrl('../workers/opencv-ops.js'),
  });
}

/**
 * Ensure OpenCV worker is ready (or main-thread fallback).
 * @returns {Promise<void>}
 */
export async function ensureOpenCVWorker() {
  if (useMainThreadFallback) {
    await ensureOpenCVMainThread();
    return;
  }

  if (workerReady?.promise) return workerReady.promise;

  if (typeof Worker === 'undefined') {
    console.warn('[unfake.js] Web Workers unavailable — using main-thread OpenCV (UI may freeze).');
    useMainThreadFallback = true;
    await ensureOpenCVMainThread();
    return;
  }

  let readyResolve;
  let readyReject;
  const readyPromise = new Promise((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
  });
  const timer = setTimeout(
    () => readyReject(new Error('OpenCV worker failed to initialize within 180 seconds.')),
    180_000,
  );
  workerReady = {
    promise: readyPromise,
    resolve: () => { clearTimeout(timer); readyResolve(); },
    reject: (err) => { clearTimeout(timer); readyReject(err); },
  };

  worker = spawnWorker();

  try {
    await initWorker();
    await workerReady.promise;
  } catch (err) {
    console.warn('[unfake.js] OpenCV worker failed, falling back to main thread:', err);
    worker?.terminate();
    worker = null;
    workerReady = null;
    useMainThreadFallback = true;
    await ensureOpenCVMainThread();
  }
}

/**
 * Run a named OpenCV operation in the worker.
 * @param {string} op
 * @param {object} payload
 * @returns {Promise<ImageData|number>}
 */
export async function runOp(op, payload = {}) {
  await ensureOpenCVWorker();

  if (useMainThreadFallback) {
    throw new Error(`OpenCV worker unavailable; operation "${op}" requires worker mode.`);
  }

  const id = ++rpcId;
  const serialized = serializePayload(payload);
  const transfers = collectTransfers(serialized);

  const result = await new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: 'rpc', id, op, payload: serialized }, transfers);
  });

  if (result?.type === 'image') {
    return payloadToImageData(result);
  }
  if (result?.type === 'number') {
    return result.value;
  }
  return result;
}

/** Alias for cvReady() compatibility. */
export async function cvReady() {
  await ensureOpenCVWorker();
  return { worker: !useMainThreadFallback };
}
