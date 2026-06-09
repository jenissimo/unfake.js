//! WASM exports for browser worker (Phase 2).

use wasm_bindgen::prelude::*;
use unfake_core::{
    auto_tiled_detect, content_adaptive_downscale, detect_optimal_color_count, downscale,
    encode_png, find_optimal_crop_offset, morph_open_close, runs_based_detect, single_region_edge_detect,
    quantize, snap_to_grid, tiled_edge_detect, vector_fill_background, vector_postprocess,
    vector_preprocess, ColorCountOpts, DownscaleMethod, DownscaleOptions,
};

fn pack_image(data: Vec<u8>, width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + data.len());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&data);
    out
}

#[wasm_bindgen]
pub fn detect_tiled(rgba: &[u8], width: u32, height: u32) -> u32 {
    tiled_edge_detect(rgba, width, height)
}

#[wasm_bindgen]
pub fn detect_legacy(rgba: &[u8], width: u32, height: u32) -> u32 {
    single_region_edge_detect(rgba, width, height)
}

#[wasm_bindgen]
pub fn detect_runs(rgba: &[u8], width: u32, height: u32) -> u32 {
    runs_based_detect(rgba, width, height)
}

#[wasm_bindgen]
pub fn detect_auto(rgba: &[u8], width: u32, height: u32) -> u32 {
    auto_tiled_detect(rgba, width, height)
}

#[wasm_bindgen]
pub fn crop_offset(rgba: &[u8], width: u32, height: u32, scale: u32) -> Vec<u32> {
    let o = find_optimal_crop_offset(rgba, width, height, scale);
    vec![o.x, o.y]
}

/// Encode RGBA8 to PNG bytes.
#[wasm_bindgen]
pub fn encode_png_bytes(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    encode_png(rgba, width, height)
}

/// Quantize to <=max_colors (imagequant) or remap to a fixed palette.
/// Packed out: u32 w, u32 h, u32 palette_len, palette RGBA bytes, image RGBA bytes.
#[wasm_bindgen]
pub fn quantize_packed(rgba: &[u8], width: u32, height: u32, max_colors: u32, fixed: &[u8]) -> Vec<u8> {
    let fp: Vec<[u8; 4]> = fixed.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    let r = quantize(rgba, width, height, max_colors, &fp);
    let mut out = Vec::with_capacity(12 + r.palette.len() * 4 + r.image.data.len());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&(r.palette.len() as u32).to_le_bytes());
    for c in &r.palette {
        out.extend_from_slice(c);
    }
    out.extend_from_slice(&r.image.data);
    out
}

/// Vector pre-process. filter: 0=none 1=bilateral 2=median. Packed image out.
#[wasm_bindgen]
pub fn vector_preprocess_packed(
    rgba: &[u8],
    width: u32,
    height: u32,
    filter: u8,
    value: u32,
    morphology: u8,
    morph_kernel: u32,
) -> Vec<u8> {
    let img = vector_preprocess(rgba, width, height, filter, value, morphology != 0, morph_kernel);
    pack_image(img.data, img.width, img.height)
}

/// Composite over solid background where alpha != 0. Packed image out.
#[wasm_bindgen]
pub fn vector_fill_bg_packed(rgba: &[u8], width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let img = vector_fill_background(rgba, width, height, [r, g, b, a]);
    pack_image(img.data, img.width, img.height)
}

/// Vector post-process. filter: 0=none 2=median 3=gaussian. Packed image out.
#[wasm_bindgen]
pub fn vector_postprocess_packed(rgba: &[u8], width: u32, height: u32, filter: u8, value: u32) -> Vec<u8> {
    let img = vector_postprocess(rgba, width, height, filter, value);
    pack_image(img.data, img.width, img.height)
}

/// Content-adaptive (perceptual) downscale to target dims. Packed image out.
#[wasm_bindgen]
pub fn content_adaptive_packed(
    rgba: &[u8],
    width: u32,
    height: u32,
    target_w: u32,
    target_h: u32,
) -> Vec<u8> {
    let img = content_adaptive_downscale(rgba, width, height, target_w, target_h);
    pack_image(img.data, img.width, img.height)
}

/// Morphological open+close cleanup. Packed image out.
#[wasm_bindgen]
pub fn morph_cleanup_packed(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let img = morph_open_close(rgba, width, height);
    pack_image(img.data, img.width, img.height)
}

/// Auto color-count heuristic (returns clamped count).
#[wasm_bindgen]
pub fn detect_color_count(
    rgba: &[u8],
    width: u32,
    height: u32,
    downsample_to: u32,
    quantize: u32,
    dominance: f64,
    max_colors: u32,
) -> u32 {
    detect_optimal_color_count(
        rgba,
        width,
        height,
        &ColorCountOpts {
            downsample_to,
            quantize,
            dominance,
            max_colors,
        },
    )
}

/// Packed: u32 LE width, u32 LE height, RGBA bytes.
#[wasm_bindgen]
pub fn snap_rgba_packed(rgba: &[u8], width: u32, height: u32, scale: u32) -> Vec<u8> {
    let img = snap_to_grid(rgba, width, height, scale);
    pack_image(img.data, img.width, img.height)
}

/// method: 0=nearest 1=median 2=mode 3=dominant 4=qvote 5=mean
/// align_grid: 1 = crop to grid before downscale (CLI); 0 = image already snapped (browser pipeline)
#[wasm_bindgen]
pub fn downscale_rgba_packed(
    rgba: &[u8],
    width: u32,
    height: u32,
    scale: u32,
    method: u8,
    dom_threshold: f64,
    align_grid: u8,
) -> Vec<u8> {
    let m = match method {
        0 => DownscaleMethod::Nearest,
        1 => DownscaleMethod::Median,
        2 => DownscaleMethod::Mode,
        4 => DownscaleMethod::Qvote,
        5 => DownscaleMethod::Mean,
        _ => DownscaleMethod::Dominant,
    };
    let out = downscale(
        rgba,
        width,
        height,
        scale,
        &DownscaleOptions {
            method: m,
            dom_threshold,
            align_grid: align_grid != 0,
        },
    );
    pack_image(out.data, out.width, out.height)
}
