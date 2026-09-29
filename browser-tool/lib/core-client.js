/**
 * Main-thread facade for unfake-core WASM (detect / snap / downscale).
 * OpenCV is not required for these ops.
 */

import { payloadToImageData, imageDataToPayload } from './core-codec.js';

let worker = null;
let workerReady = null;
let rpcId = 0;
const pending = new Map();
let coreAvailable = null;

function workerUrl(file) {
  return new URL(file, import.meta.url).href;
}

function handleWorkerMessage(e) {
  const msg = e.data;
  if (msg.type === 'ready') {
    workerReady?.resolve();
    workerReady = null;
    return;
  }
  if (msg.type === 'init-error') {
    workerReady?.reject(new Error(msg.error || 'unfake-core worker init failed'));
    workerReady = null;
    return;
  }
  if (msg.type === 'rpc-result') {
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.ok) entry.resolve(msg.result);
    else entry.reject(new Error(msg.error || 'unfake-core RPC failed'));
  }
}

function spawnWorker() {
  const w = new Worker(workerUrl('../workers/unfake-core-worker.js'), { type: 'module' });
  w.onmessage = handleWorkerMessage;
  w.onerror = (err) => {
    workerReady?.reject(err.error || new Error('unfake-core worker error'));
    workerReady = null;
  };
  return w;
}

/**
 * @returns {Promise<boolean>} true if WASM core is usable
 */
export async function ensureCoreWorker() {
  if (coreAvailable === true) return true;
  if (coreAvailable === false) return false;

  if (typeof Worker === 'undefined') {
    coreAvailable = false;
    return false;
  }

  if (workerReady?.promise) {
    try {
      await workerReady.promise;
      coreAvailable = true;
      return true;
    } catch {
      coreAvailable = false;
      return false;
    }
  }

  let readyResolve;
  let readyReject;
  const readyPromise = new Promise((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
  });
  const timer = setTimeout(
    () => readyReject(new Error('unfake-core worker init timeout')),
    30_000,
  );
  workerReady = {
    promise: readyPromise,
    resolve: () => { clearTimeout(timer); readyResolve(); },
    reject: (err) => { clearTimeout(timer); readyReject(err); },
  };

  worker = spawnWorker();
  worker.postMessage({
    type: 'init',
    wasmUrl: '../vendor/unfake-core/unfake_wasm_bg.wasm',
  });

  try {
    await workerReady.promise;
    coreAvailable = true;
    return true;
  } catch (err) {
    console.warn('[unfake.js] unfake-core WASM unavailable, using JS fallback:', err);
    worker?.terminate();
    worker = null;
    workerReady = null;
    coreAvailable = false;
    return false;
  }
}

/** @returns {Promise<ImageData|number|{x:number,y:number}|object|null>} */
export async function runCoreOp(op, payload = {}) {
  const ok = await ensureCoreWorker();
  if (!ok) throw new Error('unfake-core unavailable');

  const id = ++rpcId;
  const serialized = { ...payload };
  const transfers = [];
  if (payload.image instanceof ImageData) {
    const copy = new Uint8ClampedArray(payload.image.data);
    serialized.image = {
      width: payload.image.width,
      height: payload.image.height,
      buffer: copy.buffer,
    };
    transfers.push(copy.buffer);
  }

  const result = await new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: 'rpc', id, op, payload: serialized }, transfers);
  });

  if (result?.type === 'image') return payloadToImageData(result);
  if (result?.type === 'number') return result.value;
  if (result?.type === 'offset') return { x: result.x, y: result.y };
  if (result?.type === 'grid') return result.grid;
  if (result?.type === 'bounds') return result.bounds;
  if (result?.type === 'bytes') return new Uint8Array(result.buffer);
  if (result?.type === 'quantize') {
    return {
      quantized: payloadToImageData({ width: result.width, height: result.height, buffer: result.buffer }),
      palette: result.palette,
      colorsUsed: result.colorsUsed,
    };
  }
  return result;
}

export async function coreReady() {
  return ensureCoreWorker();
}

/** Downscale method id for WASM (matches unfake-wasm). */
export const CORE_DS = {
  nearest: 0,
  median: 1,
  mode: 2,
  dominant: 3,
  qvote: 4,
  mean: 5,
};
