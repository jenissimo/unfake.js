//! Block downscale methods (parity with `browser-tool/lib/utils.js` + `pixel.js`).

use crate::detect::{CellBounds, Grid};
use crate::image::RgbaImage;
use crate::math::{mean_u8, median_u8, mode_u8};
use crate::snap::find_optimal_crop_offset;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownscaleMethod {
    Nearest,
    Median,
    Mode,
    Mean,
    /// `downscaleByDominantColor` — packed RGB dominant + mean fallback, binary alpha.
    Dominant,
    /// Quantize to 24 colors (median-cut, no dither) then mode vote per block.
    Qvote,
}

pub struct DownscaleOptions {
    pub method: DownscaleMethod,
    /// Used by `Dominant` (default 0.15 in prod).
    pub dom_threshold: f64,
    /// Crop to grid before downscale (uses snap offset).
    pub align_grid: bool,
}

impl Default for DownscaleOptions {
    fn default() -> Self {
        Self {
            method: DownscaleMethod::Dominant,
            dom_threshold: 0.15,
            align_grid: true,
        }
    }
}

pub fn downscale(
    rgba: &[u8],
    width: u32,
    height: u32,
    scale: u32,
    opts: &DownscaleOptions,
) -> RgbaImage {
    if scale <= 1 {
        return RgbaImage {
            data: rgba.to_vec(),
            width,
            height,
        };
    }

    let src = if opts.align_grid {
        let off = find_optimal_crop_offset(rgba, width, height, scale);
        let cw = ((width.saturating_sub(off.x)) / scale) * scale;
        let ch = ((height.saturating_sub(off.y)) / scale) * scale;
        if cw < scale || ch < scale {
            return RgbaImage {
                data: rgba.to_vec(),
                width,
                height,
            };
        }
        RgbaImage::from_raw(rgba.to_vec(), width, height)
            .and_then(|img| img.crop(off.x, off.y, cw, ch))
            .unwrap()
    } else {
        RgbaImage {
            data: rgba.to_vec(),
            width,
            height,
        }
    };

    let xs = uniform_spans(src.width / scale, scale);
    let ys = uniform_spans(src.height / scale, scale);
    reduce_cells(&src, &xs, &ys, opts)
}

/// Downscale on a (possibly fractional) grid: one output pixel per cell whose
/// center is inside the image, reduced over the cell's inner region so the
/// resampling blur at cell borders doesn't vote. `opts.align_grid` is ignored —
/// the grid carries its own phase.
pub fn downscale_grid(
    rgba: &[u8],
    width: u32,
    height: u32,
    grid: &Grid,
    opts: &DownscaleOptions,
) -> RgbaImage {
    downscale_bounds(rgba, width, height, &grid.bounds(width, height), opts)
}

/// Downscale on explicit cell boundaries (a regular grid's, or an elastic fit
/// from `elastic_bounds`); same inner-region sampling as `downscale_grid`.
pub fn downscale_bounds(
    rgba: &[u8],
    width: u32,
    height: u32,
    bounds: &CellBounds,
    opts: &DownscaleOptions,
) -> RgbaImage {
    let src = RgbaImage {
        data: rgba.to_vec(),
        width,
        height,
    };
    let xs = bound_spans(&bounds.x, width);
    let ys = bound_spans(&bounds.y, height);
    reduce_cells(&src, &xs, &ys, opts)
}

/// Source pixel range `[start, end)` reduced into one output pixel; `center`
/// is what `Nearest` samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    center: usize,
}

fn uniform_spans(cells: u32, scale: u32) -> Vec<Span> {
    let s = scale as usize;
    (0..cells as usize)
        .map(|t| Span {
            start: t * s,
            end: t * s + s,
            center: t * s + s / 2,
        })
        .collect()
}

/// Inner region of each cell between consecutive boundaries: pixels whose
/// centers are at least a quarter cell from the borders (at least the center
/// pixel).
fn bound_spans(bounds: &[f64], len: u32) -> Vec<Span> {
    let n = len as i64;
    bounds
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let p = b - a;
            let margin = (0.25 * p).min(((p - 1.0) / 2.0).max(0.0));
            let center = (((a + b) / 2.0).floor() as i64).clamp(0, n - 1);
            let lo = ((a + margin - 0.5).ceil() as i64).max(0);
            let hi = ((b - margin - 0.5).floor() as i64).min(n - 1);
            let (start, end) = if hi >= lo { (lo, hi + 1) } else { (center, center + 1) };
            Span {
                start: start as usize,
                end: end as usize,
                center: center as usize,
            }
        })
        .collect()
}

fn reduce_cells(img: &RgbaImage, xs: &[Span], ys: &[Span], opts: &DownscaleOptions) -> RgbaImage {
    if xs.is_empty() || ys.is_empty() {
        return RgbaImage {
            data: vec![0, 0, 0, 0],
            width: 1,
            height: 1,
        };
    }
    match opts.method {
        DownscaleMethod::Dominant => downscale_dominant(img, xs, ys, opts.dom_threshold),
        DownscaleMethod::Qvote => downscale_qvote(img, xs, ys),
        _ => downscale_block(img, xs, ys, opts.method),
    }
}

fn downscale_block(img: &RgbaImage, xs: &[Span], ys: &[Span], method: DownscaleMethod) -> RgbaImage {
    let target_w = xs.len();
    let target_h = ys.len();
    let mut out = vec![0u8; target_w * target_h * 4];
    let d = &img.data;
    let sw = img.width as usize;

    for (ty, row) in ys.iter().enumerate() {
        for (tx, col) in xs.iter().enumerate() {
            let offset = (ty * target_w + tx) * 4;

            if method == DownscaleMethod::Nearest {
                let idx = (row.center * sw + col.center) * 4;
                out[offset..offset + 4].copy_from_slice(&d[idx..idx + 4]);
                continue;
            }

            let mut rs = Vec::new();
            let mut gs = Vec::new();
            let mut bs = Vec::new();
            let mut alphas = Vec::new();

            for sy in row.start..row.end {
                for sx in col.start..col.end {
                    let idx = (sy * sw + sx) * 4;
                    alphas.push(d[idx + 3]);
                    if d[idx + 3] > 128 {
                        rs.push(d[idx]);
                        gs.push(d[idx + 1]);
                        bs.push(d[idx + 2]);
                    }
                }
            }

            let has_color = !rs.is_empty();
            let agg = |ch: &[u8]| -> u8 {
                match method {
                    DownscaleMethod::Mode => mode_u8(ch),
                    DownscaleMethod::Mean => mean_u8(ch),
                    _ => median_u8(ch),
                }
            };

            out[offset] = if has_color { agg(&rs) } else { 0 };
            out[offset + 1] = if has_color { agg(&gs) } else { 0 };
            out[offset + 2] = if has_color { agg(&bs) } else { 0 };
            out[offset + 3] = median_u8(&alphas);
        }
    }

    RgbaImage {
        data: out,
        width: target_w as u32,
        height: target_h as u32,
    }
}

fn downscale_dominant(img: &RgbaImage, xs: &[Span], ys: &[Span], threshold: f64) -> RgbaImage {
    let target_w = xs.len();
    let target_h = ys.len();
    let mut out = vec![0u8; target_w * target_h * 4];
    let d = &img.data;
    let sw = img.width as usize;

    for (ty, row) in ys.iter().enumerate() {
        for (tx, col) in xs.iter().enumerate() {
            use std::collections::HashMap;
            let mut color_counts: HashMap<u32, u32> = HashMap::new();
            let mut alphas = Vec::new();
            let mut opaque_packed = Vec::new();

            for sy in row.start..row.end {
                for sx in col.start..col.end {
                    let idx = (sy * sw + sx) * 4;
                    let a = d[idx + 3];
                    alphas.push(a);
                    if a > 128 {
                        let packed =
                            ((d[idx] as u32) << 16) | ((d[idx + 1] as u32) << 8) | (d[idx + 2] as u32);
                        *color_counts.entry(packed).or_insert(0) += 1;
                        opaque_packed.push(packed);
                    }
                }
            }

            let out_idx = (ty * target_w + tx) * 4;
            if opaque_packed.is_empty() {
                continue;
            }

            let (dominant, max_count) = color_counts
                .iter()
                .max_by_key(|(_, c)| *c)
                .map(|(k, c)| (*k, *c))
                .unwrap();

            let final_color = if max_count as f64 / opaque_packed.len() as f64 >= threshold {
                dominant
            } else {
                let r_sum: u32 = opaque_packed.iter().map(|c| (c >> 16) & 0xff).sum();
                let g_sum: u32 = opaque_packed.iter().map(|c| (c >> 8) & 0xff).sum();
                let b_sum: u32 = opaque_packed.iter().map(|c| c & 0xff).sum();
                let n = opaque_packed.len() as u32;
                let avg_r = (r_sum / n) as u8;
                let avg_g = (g_sum / n) as u8;
                let avg_b = (b_sum / n) as u8;
                ((avg_r as u32) << 16) | ((avg_g as u32) << 8) | (avg_b as u32)
            };

            out[out_idx] = ((final_color >> 16) & 0xff) as u8;
            out[out_idx + 1] = ((final_color >> 8) & 0xff) as u8;
            out[out_idx + 2] = (final_color & 0xff) as u8;
            out[out_idx + 3] = if median_u8(&alphas) > 128 { 255 } else { 0 };
        }
    }

    RgbaImage {
        data: out,
        width: target_w as u32,
        height: target_h as u32,
    }
}

fn downscale_qvote(img: &RgbaImage, xs: &[Span], ys: &[Span]) -> RgbaImage {
    let quantized = quantize_rgba(&img.data, img.width, img.height, 24);
    let qimg = RgbaImage {
        data: quantized,
        width: img.width,
        height: img.height,
    };
    downscale_block(&qimg, xs, ys, DownscaleMethod::Mode)
}

fn quantize_rgba(rgba: &[u8], width: u32, height: u32, max_colors: u32) -> Vec<u8> {
    use imagequant::{new, RGBA};

    let mut liq = new();
    let _ = liq.set_max_colors(max_colors);
    let _ = liq.set_speed(1);

    let pixels: Vec<RGBA> = rgba
        .chunks_exact(4)
        .map(|c| RGBA {
            r: c[0],
            g: c[1],
            b: c[2],
            a: c[3],
        })
        .collect();

    let mut img = match liq.new_image(pixels, width as usize, height as usize, 0.0) {
        Ok(i) => i,
        Err(_) => return rgba.to_vec(),
    };

    let mut res = match liq.quantize(&mut img) {
        Ok(r) => r,
        Err(_) => return rgba.to_vec(),
    };
    res.set_dithering_level(0.0).ok();

    let (palette, indices) = match res.remapped(&mut img) {
        Ok(r) => r,
        Err(_) => return rgba.to_vec(),
    };

    let mut out = vec![0u8; rgba.len()];
    for (i, &idx) in indices.iter().enumerate() {
        let c = palette[idx as usize];
        let o = i * 4;
        out[o] = c.r;
        out[o + 1] = c.g;
        out[o + 2] = c.b;
        out[o + 3] = rgba[o + 3];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker_4x(w: u32, h: u32) -> RgbaImage {
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let c = if (x / 4 + y / 4) % 2 == 0 { 255 } else { 0 };
                data[i] = c;
                data[i + 1] = c;
                data[i + 2] = c;
                data[i + 3] = 255;
            }
        }
        RgbaImage { data, width: w, height: h }
    }

    #[test]
    fn downscale_4x_nearest_halves_dimensions() {
        let src = checker_4x(32, 32);
        let out = downscale(
            &src.data,
            src.width,
            src.height,
            4,
            &DownscaleOptions {
                method: DownscaleMethod::Nearest,
                dom_threshold: 0.15,
                align_grid: false,
            },
        );
        assert_eq!(out.width, 8);
        assert_eq!(out.height, 8);
    }
}
