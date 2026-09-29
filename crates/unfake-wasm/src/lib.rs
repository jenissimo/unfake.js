//! WASM exports for browser worker (Phase 2).

use wasm_bindgen::prelude::*;
use unfake_core::{
    auto_tiled_detect, content_adaptive_downscale, detect_optimal_color_count, downscale,
    downscale_bounds, downscale_grid, elastic_bounds, encode_png, find_optimal_crop_offset,
    fractional_grid_detect, grid_for_scale, morph_open_close, runs_based_detect,
    single_region_edge_detect, quantize, snap_to_grid, tiled_edge_detect, vector_fill_background,
    vector_postprocess, vector_preprocess, AxisGrid, CellBounds, ColorCountOpts, DownscaleMethod,
    DownscaleOptions, Grid,
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

/// Fractional (non-integer) cell grid, or empty when the integer detectors
/// apply. Out: [period_x, period_y, offset_x, offset_y, cells_x, cells_y].
#[wasm_bindgen]
pub fn detect_grid(rgba: &[u8], width: u32, height: u32) -> Vec<f64> {
    fractional_grid_detect(rgba, width, height)
        .map(|g| grid_vec(&g, width, height))
        .unwrap_or_default()
}

/// Grid for a manual (possibly fractional) cell size; layout as `detect_grid`.
#[wasm_bindgen]
pub fn grid_for_scale_vec(rgba: &[u8], width: u32, height: u32, scale: f64) -> Vec<f64> {
    grid_vec(&grid_for_scale(rgba, width, height, scale), width, height)
}

/// Boundaries fitted to hand-drawn cells around a grid (from `detect_grid`, or
/// an integer scale with its crop offset), or empty when the grid already fits.
/// Out: [adapted_x, adapted_y, len_x, x bounds…, y bounds…].
#[wasm_bindgen]
pub fn elastic_bounds_vec(
    rgba: &[u8],
    width: u32,
    height: u32,
    period_x: f64,
    period_y: f64,
    offset_x: f64,
    offset_y: f64,
) -> Vec<f64> {
    let grid = grid_from(period_x, period_y, offset_x, offset_y);
    match elastic_bounds(rgba, width, height, &grid) {
        Some((b, adapted)) => {
            let mut out = vec![adapted[0] as u8 as f64, adapted[1] as u8 as f64, b.x.len() as f64];
            out.extend_from_slice(&b.x);
            out.extend_from_slice(&b.y);
            out
        }
        None => Vec::new(),
    }
}

fn grid_from(period_x: f64, period_y: f64, offset_x: f64, offset_y: f64) -> Grid {
    let axis = |period, offset| AxisGrid {
        period,
        offset,
        coherence: 0.0,
        prominence: 0.0,
    };
    Grid {
        x: axis(period_x, offset_x),
        y: axis(period_y, offset_y),
    }
}

fn grid_vec(g: &Grid, width: u32, height: u32) -> Vec<f64> {
    let (cw, ch) = g.cells(width, height);
    vec![g.x.period, g.y.period, g.x.offset, g.y.offset, cw as f64, ch as f64]
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
    let out = downscale(
        rgba,
        width,
        height,
        scale,
        &DownscaleOptions {
            method: downscale_method(method),
            dom_threshold,
            align_grid: align_grid != 0,
        },
    );
    pack_image(out.data, out.width, out.height)
}

/// Downscale on a fractional grid (from `detect_grid` / `grid_for_scale_vec`).
/// method ids as `downscale_rgba_packed`. Packed image out.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn downscale_grid_packed(
    rgba: &[u8],
    width: u32,
    height: u32,
    period_x: f64,
    period_y: f64,
    offset_x: f64,
    offset_y: f64,
    method: u8,
    dom_threshold: f64,
) -> Vec<u8> {
    let out = downscale_grid(
        rgba,
        width,
        height,
        &grid_from(period_x, period_y, offset_x, offset_y),
        &DownscaleOptions {
            method: downscale_method(method),
            dom_threshold,
            align_grid: false,
        },
    );
    pack_image(out.data, out.width, out.height)
}

/// Downscale on explicit cell boundaries (from `elastic_bounds_vec`).
/// method ids as `downscale_rgba_packed`. Packed image out.
#[wasm_bindgen]
pub fn downscale_bounds_packed(
    rgba: &[u8],
    width: u32,
    height: u32,
    bounds_x: &[f64],
    bounds_y: &[f64],
    method: u8,
    dom_threshold: f64,
) -> Vec<u8> {
    let bounds = CellBounds {
        x: bounds_x.to_vec(),
        y: bounds_y.to_vec(),
    };
    let out = downscale_bounds(
        rgba,
        width,
        height,
        &bounds,
        &DownscaleOptions {
            method: downscale_method(method),
            dom_threshold,
            align_grid: false,
        },
    );
    pack_image(out.data, out.width, out.height)
}

fn downscale_method(id: u8) -> DownscaleMethod {
    match id {
        0 => DownscaleMethod::Nearest,
        1 => DownscaleMethod::Median,
        2 => DownscaleMethod::Mode,
        4 => DownscaleMethod::Qvote,
        5 => DownscaleMethod::Mean,
        _ => DownscaleMethod::Dominant,
    }
}
