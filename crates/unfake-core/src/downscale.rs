//! Block downscale methods (parity with `browser-tool/lib/utils.js` + `pixel.js`).

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

    match opts.method {
        DownscaleMethod::Dominant => downscale_dominant(&src, scale, opts.dom_threshold),
        DownscaleMethod::Qvote => downscale_qvote(&src, scale),
        _ => downscale_block(&src, scale, opts.method),
    }
}

fn downscale_block(img: &RgbaImage, scale: u32, method: DownscaleMethod) -> RgbaImage {
    let target_w = img.width / scale;
    let target_h = img.height / scale;
    if target_w == 0 || target_h == 0 {
        return RgbaImage {
            data: vec![0, 0, 0, 0],
            width: 1,
            height: 1,
        };
    }

    let mut out = vec![0u8; (target_w * target_h * 4) as usize];
    let d = &img.data;
    let sw = img.width as usize;

    for ty in 0..target_h {
        for tx in 0..target_w {
            let offset = ((ty * target_w + tx) * 4) as usize;

            if method == DownscaleMethod::Nearest {
                let sx = (tx * scale + scale / 2) as usize;
                let sy = (ty * scale + scale / 2) as usize;
                if sx < img.width as usize && sy < img.height as usize {
                    let idx = (sy * sw + sx) * 4;
                    out[offset..offset + 4].copy_from_slice(&d[idx..idx + 4]);
                }
                continue;
            }

            let mut rs = Vec::new();
            let mut gs = Vec::new();
            let mut bs = Vec::new();
            let mut alphas = Vec::new();

            for dy in 0..scale {
                for dx in 0..scale {
                    let sx = (tx * scale + dx) as usize;
                    let sy = (ty * scale + dy) as usize;
                    if sx >= img.width as usize || sy >= img.height as usize {
                        continue;
                    }
                    let idx = (sy * sw + sx) * 4;
                    alphas.push(d[idx + 3]);
                    if d[idx + 3] > 128 {
                        rs.push(d[idx]);
                        gs.push(d[idx + 1]);
                        bs.push(d[idx + 2]);
                    }
                }
            }

            if alphas.is_empty() {
                continue;
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
        width: target_w,
        height: target_h,
    }
}

fn downscale_dominant(img: &RgbaImage, scale: u32, threshold: f64) -> RgbaImage {
    let target_w = img.width / scale;
    let target_h = img.height / scale;
    let mut out = vec![0u8; (target_w * target_h * 4) as usize];
    let d = &img.data;
    let sw = img.width as usize;

    for ty in 0..target_h {
        for tx in 0..target_w {
            use std::collections::HashMap;
            let mut color_counts: HashMap<u32, u32> = HashMap::new();
            let mut alphas = Vec::new();
            let mut opaque_packed = Vec::new();

            for dy in 0..scale {
                for dx in 0..scale {
                    let sx = (tx * scale + dx) as usize;
                    let sy = (ty * scale + dy) as usize;
                    if sx >= sw || sy >= img.height as usize {
                        continue;
                    }
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

            let out_idx = ((ty * target_w + tx) * 4) as usize;
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
        width: target_w,
        height: target_h,
    }
}

fn downscale_qvote(img: &RgbaImage, scale: u32) -> RgbaImage {
    let quantized = quantize_rgba(&img.data, img.width, img.height, 24);
    let qimg = RgbaImage {
        data: quantized,
        width: img.width,
        height: img.height,
    };
    downscale_block(&qimg, scale, DownscaleMethod::Mode)
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
