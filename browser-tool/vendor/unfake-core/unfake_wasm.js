/**
 * Content-adaptive (perceptual) downscale to target dims. Packed image out.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {number} target_w
 * @param {number} target_h
 * @returns {Uint8Array}
 */
export function content_adaptive_packed(rgba, width, height, target_w, target_h) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.content_adaptive_packed(ptr0, len0, width, height, target_w, target_h);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {number} scale
 * @returns {Uint32Array}
 */
export function crop_offset(rgba, width, height, scale) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.crop_offset(ptr0, len0, width, height, scale);
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @returns {number}
 */
export function detect_auto(rgba, width, height) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.detect_auto(ptr0, len0, width, height);
    return ret >>> 0;
}

/**
 * Auto color-count heuristic (returns clamped count).
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {number} downsample_to
 * @param {number} quantize
 * @param {number} dominance
 * @param {number} max_colors
 * @returns {number}
 */
export function detect_color_count(rgba, width, height, downsample_to, quantize, dominance, max_colors) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.detect_color_count(ptr0, len0, width, height, downsample_to, quantize, dominance, max_colors);
    return ret >>> 0;
}

/**
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @returns {number}
 */
export function detect_legacy(rgba, width, height) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.detect_legacy(ptr0, len0, width, height);
    return ret >>> 0;
}

/**
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @returns {number}
 */
export function detect_runs(rgba, width, height) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.detect_runs(ptr0, len0, width, height);
    return ret >>> 0;
}

/**
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @returns {number}
 */
export function detect_tiled(rgba, width, height) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.detect_tiled(ptr0, len0, width, height);
    return ret >>> 0;
}

/**
 * method: 0=nearest 1=median 2=mode 3=dominant 4=qvote 5=mean
 * align_grid: 1 = crop to grid before downscale (CLI); 0 = image already snapped (browser pipeline)
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {number} scale
 * @param {number} method
 * @param {number} dom_threshold
 * @param {number} align_grid
 * @returns {Uint8Array}
 */
export function downscale_rgba_packed(rgba, width, height, scale, method, dom_threshold, align_grid) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.downscale_rgba_packed(ptr0, len0, width, height, scale, method, dom_threshold, align_grid);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Encode RGBA8 to PNG bytes.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @returns {Uint8Array}
 */
export function encode_png_bytes(rgba, width, height) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.encode_png_bytes(ptr0, len0, width, height);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Morphological open+close cleanup. Packed image out.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @returns {Uint8Array}
 */
export function morph_cleanup_packed(rgba, width, height) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.morph_cleanup_packed(ptr0, len0, width, height);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Quantize to <=max_colors (imagequant) or remap to a fixed palette.
 * Packed out: u32 w, u32 h, u32 palette_len, palette RGBA bytes, image RGBA bytes.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {number} max_colors
 * @param {Uint8Array} fixed
 * @returns {Uint8Array}
 */
export function quantize_packed(rgba, width, height, max_colors, fixed) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(fixed, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.quantize_packed(ptr0, len0, width, height, max_colors, ptr1, len1);
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Packed: u32 LE width, u32 LE height, RGBA bytes.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {number} scale
 * @returns {Uint8Array}
 */
export function snap_rgba_packed(rgba, width, height, scale) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.snap_rgba_packed(ptr0, len0, width, height, scale);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Composite over solid background where alpha != 0. Packed image out.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {number} r
 * @param {number} g
 * @param {number} b
 * @param {number} a
 * @returns {Uint8Array}
 */
export function vector_fill_bg_packed(rgba, width, height, r, g, b, a) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vector_fill_bg_packed(ptr0, len0, width, height, r, g, b, a);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Vector post-process. filter: 0=none 2=median 3=gaussian. Packed image out.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {number} filter
 * @param {number} value
 * @returns {Uint8Array}
 */
export function vector_postprocess_packed(rgba, width, height, filter, value) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vector_postprocess_packed(ptr0, len0, width, height, filter, value);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Vector pre-process. filter: 0=none 1=bilateral 2=median. Packed image out.
 * @param {Uint8Array} rgba
 * @param {number} width
 * @param {number} height
 * @param {number} filter
 * @param {number} value
 * @param {number} morphology
 * @param {number} morph_kernel
 * @returns {Uint8Array}
 */
export function vector_preprocess_packed(rgba, width, height, filter, value, morphology, morph_kernel) {
    const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vector_preprocess_packed(ptr0, len0, width, height, filter, value, morphology, morph_kernel);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./unfake_wasm_bg.js": import0,
    };
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('unfake_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
