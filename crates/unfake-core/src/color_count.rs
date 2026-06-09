//! Auto color-count heuristic (parity with `utils.js` detectOptimalColorCount:
//! area-downsample → median(5) → gaussian(5,σ1) → count quantized dominant colors).

use crate::image::{resize_area, RgbaImage};
use std::collections::HashMap;

pub struct ColorCountOpts {
    pub downsample_to: u32,
    pub quantize: u32,
    pub dominance: f64,
    pub max_colors: u32,
}

impl Default for ColorCountOpts {
    fn default() -> Self {
        Self {
            downsample_to: 64,
            quantize: 48,
            dominance: 0.015,
            max_colors: 32,
        }
    }
}

pub fn detect_optimal_color_count(rgba: &[u8], width: u32, height: u32, opts: &ColorCountOpts) -> u32 {
    let tw = opts.downsample_to.max(1);
    let th = ((tw as f64) * (height as f64) / (width as f64)).round().max(1.0) as u32;
    let small = resize_area(rgba, width, height, tw, th);
    let denoised = gaussian5(&median5(&small));

    let q = opts.quantize.max(1) as f64;
    let mut counts: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let total = (tw * th) as f64;
    for px in denoised.data.chunks_exact(4) {
        if px[3] < 200 {
            continue;
        }
        let key = (
            ((px[0] as f64 / q).round() * q) as i32,
            ((px[1] as f64 / q).round() * q) as i32,
            ((px[2] as f64 / q).round() * q) as i32,
        );
        *counts.entry(key).or_insert(0) += 1;
    }

    let min_dom = (3.0_f64).max((total * opts.dominance).round()) as u32;
    let significant: Vec<u32> = counts.values().copied().filter(|&c| c >= min_dom).collect();
    let mut n = significant.len() as u32;
    if n > opts.max_colors {
        let strict = min_dom.max((total * 0.02).round() as u32);
        n = significant.iter().filter(|&&c| c >= strict).count() as u32;
    }
    n.clamp(2, opts.max_colors)
}

/// 5x5 per-channel median (border replicate).
fn median5(img: &RgbaImage) -> RgbaImage {
    let w = img.width as i32;
    let h = img.height as i32;
    let mut out = vec![0u8; img.data.len()];
    let mut window = [0u8; 25];
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let mut k = 0;
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let xx = (x + dx).clamp(0, w - 1);
                        let yy = (y + dy).clamp(0, h - 1);
                        window[k] = img.data[((yy * w + xx) * 4) as usize + c];
                        k += 1;
                    }
                }
                window.sort_unstable();
                out[((y * w + x) * 4) as usize + c] = window[12];
            }
        }
    }
    RgbaImage { data: out, width: img.width, height: img.height }
}

/// Separable 5x5 Gaussian, σ≈1 (matches cv.GaussianBlur(5,5,1,1)).
fn gaussian5(img: &RgbaImage) -> RgbaImage {
    const K: [f64; 5] = [0.05448868, 0.24420134, 0.40261996, 0.24420134, 0.05448868];
    let w = img.width as i32;
    let h = img.height as i32;
    let conv = |src: &[u8], horizontal: bool| -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    let mut acc = 0.0;
                    for (t, &kk) in K.iter().enumerate() {
                        let off = t as i32 - 2;
                        let (xx, yy) = if horizontal {
                            ((x + off).clamp(0, w - 1), y)
                        } else {
                            (x, (y + off).clamp(0, h - 1))
                        };
                        acc += kk * src[((yy * w + xx) * 4) as usize + c] as f64;
                    }
                    out[((y * w + x) * 4) as usize + c] = acc.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        out
    };
    let pass1 = conv(&img.data, true);
    let pass2 = conv(&pass1, false);
    RgbaImage { data: pass2, width: img.width, height: img.height }
}
