/* global importScripts, self */

let cv = null;
let ready = false;
let initStarted = false;

function postReady() {
  if (ready) return;
  ready = true;
  self.postMessage({ type: 'ready' });
}

function waitForCvRuntime() {
  if (cv?.Mat && typeof cv.getBuildInformation === 'function') {
    try {
      const m = new cv.Mat(1, 1, cv.CV_8UC1);
      m.delete();
      postReady();
      return;
    } catch {
      /* WASM still loading */
    }
  }
  const prev = cv.onRuntimeInitialized;
  cv.onRuntimeInitialized = () => {
    if (typeof prev === 'function') prev();
    postReady();
  };
}

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'init') {
    if (initStarted) return;
    initStarted = true;
    try {
      const prevModule = self.Module;
      self.Module = {
        ...(prevModule || {}),
        onRuntimeInitialized() {
          if (typeof prevModule?.onRuntimeInitialized === 'function') {
            prevModule.onRuntimeInitialized();
          }
          postReady();
        },
      };

      importScripts(msg.svdUrl);
      importScripts(msg.opencvUrl);
      importScripts(msg.opsUrl);

      cv = self.cv;
      if (!cv) throw new Error('OpenCV script loaded but self.cv is missing.');
      waitForCvRuntime();
    } catch (err) {
      self.postMessage({ type: 'init-error', error: String(err.message || err) });
    }
    return;
  }

  if (msg.type === 'rpc') {
    if (!ready || !cv || !self.openCvOps?.[msg.op]) {
      self.postMessage({
        type: 'rpc-result',
        id: msg.id,
        ok: false,
        error: ready ? `Unknown OpenCV op: ${msg.op}` : 'OpenCV worker not ready',
      });
      return;
    }

    try {
      const result = self.openCvOps[msg.op](msg.payload, cv);
      const transfers = [];
      let out = result;

      if (result?.type === 'image') {
        transfers.push(result.buffer);
      }

      self.postMessage({ type: 'rpc-result', id: msg.id, ok: true, result: out }, transfers);
    } catch (err) {
      self.postMessage({
        type: 'rpc-result',
        id: msg.id,
        ok: false,
        error: String(err.message || err),
      });
    }
  }
};
