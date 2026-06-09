//! Vector-mode image filters — port of `opencv-ops.js` vectorPreProcess /
//! vectorFillBackground / vectorPostProcess. These feed imagetracer.js; exact
//! OpenCV parity isn't required (perceptual prefilters), so bilateral/gaussian
//! use textbook formulas.

use crate::image::RgbaImage;

/// filter: 0=none 1=bilateral 2=median. Optional NxN morphological close after.
pub fn vector_preprocess(
    rgba: &[u8],
    width: u32,
    height: u32,
    filter: u8,
    value: u32,
    morphology: bool,
    morph_kernel: u32,
) -> RgbaImage {
    let mut img = match filter {
        1 => bilateral_rgb(rgba, width, height, value.max(1)),
        2 => median_k(rgba, width, height, odd(value.max(1))),
        _ => RgbaImage { data: rgba.to_vec(), width, height },
    };
    if morphology {
        img = morph_close_k(&img.data, width, height, morph_kernel.max(1));
    }
    img
}

/// Composite src over a solid background where alpha != 0 (cv copyTo+mask).
pub fn vector_fill_background(rgba: &[u8], width: u32, height: u32, bg: [u8; 4]) -> RgbaImage {
    let mut out = vec![0u8; rgba.len()];
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        let o = i * 4;
        if px[3] != 0 {
            out[o..o + 4].copy_from_slice(px);
        } else {
            out[o..o + 4].copy_from_slice(&bg);
        }
    }
    RgbaImage { data: out, width, height }
}

/// filter: 0=none 2=median 3=gaussian.
pub fn vector_postprocess(rgba: &[u8], width: u32, height: u32, filter: u8, value: u32) -> RgbaImage {
    let k = odd(value.max(1));
    match filter {
        3 => gaussian_k(rgba, width, height, k),
        2 => median_k(rgba, width, height, k),
        _ => RgbaImage { data: rgba.to_vec(), width, height },
    }
}

fn odd(k: u32) -> u32 {
    if k % 2 == 0 {
        k + 1
    } else {
        k
    }
}

fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

/// Textbook bilateral on RGB (alpha passed through). d = diameter.
fn bilateral_rgb(rgba: &[u8], width: u32, height: u32, d: u32) -> RgbaImage {
    let w = width as i32;
    let h = height as i32;
    let radius = (d / 2).max(1) as i32;
    let sigma_color = (d as f64) * 2.0;
    let sigma_space = (d as f64) / 2.0;
    let gc = -0.5 / (sigma_color * sigma_color);
    let gs = -0.5 / (sigma_space * sigma_space);
    let mut out = vec![0u8; rgba.len()];
    for y in 0..h {
        for x in 0..w {
            let ci = ((y * w + x) * 4) as usize;
            let (cr, cg, cb) = (rgba[ci] as f64, rgba[ci + 1] as f64, rgba[ci + 2] as f64);
            let (mut ar, mut ag, mut ab, mut wsum) = (0.0, 0.0, 0.0, 0.0);
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let xx = clampi(x + dx, 0, w - 1);
                    let yy = clampi(y + dy, 0, h - 1);
                    let ni = ((yy * w + xx) * 4) as usize;
                    let (nr, ng, nb) = (rgba[ni] as f64, rgba[ni + 1] as f64, rgba[ni + 2] as f64);
                    let cd = (nr - cr).powi(2) + (ng - cg).powi(2) + (nb - cb).powi(2);
                    let sd = (dx * dx + dy * dy) as f64;
                    let wt = (gc * cd + gs * sd).exp();
                    ar += wt * nr;
                    ag += wt * ng;
                    ab += wt * nb;
                    wsum += wt;
                }
            }
            out[ci] = (ar / wsum).round().clamp(0.0, 255.0) as u8;
            out[ci + 1] = (ag / wsum).round().clamp(0.0, 255.0) as u8;
            out[ci + 2] = (ab / wsum).round().clamp(0.0, 255.0) as u8;
            out[ci + 3] = rgba[ci + 3];
        }
    }
    RgbaImage { data: out, width, height }
}

/// Per-channel KxK median (border replicate).
fn median_k(rgba: &[u8], width: u32, height: u32, k: u32) -> RgbaImage {
    let w = width as i32;
    let h = height as i32;
    let r = (k / 2) as i32;
    let mut out = vec![0u8; rgba.len()];
    let mut buf: Vec<u8> = Vec::with_capacity((k * k) as usize);
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                buf.clear();
                for dy in -r..=r {
                    for dx in -r..=r {
                        let xx = clampi(x + dx, 0, w - 1);
                        let yy = clampi(y + dy, 0, h - 1);
                        buf.push(rgba[((yy * w + xx) * 4) as usize + c as usize]);
                    }
                }
                buf.sort_unstable();
                out[((y * w + x) * 4) as usize + c as usize] = buf[buf.len() / 2];
            }
        }
    }
    RgbaImage { data: out, width, height }
}

/// Separable KxK Gaussian (σ from ksize, like cv σ=0), all 4 channels.
fn gaussian_k(rgba: &[u8], width: u32, height: u32, k: u32) -> RgbaImage {
    let sigma = 0.3 * (((k as f64) - 1.0) * 0.5 - 1.0) + 0.8;
    let r = (k / 2) as i32;
    let mut kernel = vec![0.0f64; k as usize];
    let mut sum = 0.0;
    for (idx, kv) in kernel.iter_mut().enumerate() {
        let off = idx as i32 - r;
        *kv = (-(off * off) as f64 / (2.0 * sigma * sigma)).exp();
        sum += *kv;
    }
    for kv in kernel.iter_mut() {
        *kv /= sum;
    }
    let w = width as i32;
    let h = height as i32;
    let conv = |src: &[u8], horizontal: bool| -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    let mut acc = 0.0;
                    for (t, &kv) in kernel.iter().enumerate() {
                        let off = t as i32 - r;
                        let (xx, yy) = if horizontal {
                            (clampi(x + off, 0, w - 1), y)
                        } else {
                            (x, clampi(y + off, 0, h - 1))
                        };
                        acc += kv * src[((yy * w + xx) * 4) as usize + c as usize] as f64;
                    }
                    out[((y * w + x) * 4) as usize + c as usize] = acc.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        out
    };
    let p1 = conv(rgba, true);
    let p2 = conv(&p1, false);
    RgbaImage { data: p2, width, height }
}

/// NxN morphological close (dilate then erode), per channel, anchor center, replicate.
fn morph_close_k(rgba: &[u8], width: u32, height: u32, k: u32) -> RgbaImage {
    let dil = neigh_k(rgba, width, height, k, 0, |a, b| a.max(b));
    let ero = neigh_k(&dil, width, height, k, 255, |a, b| a.min(b));
    RgbaImage { data: ero, width, height }
}

fn neigh_k(rgba: &[u8], width: u32, height: u32, k: u32, init: u8, pick: fn(u8, u8) -> u8) -> Vec<u8> {
    let w = width as i32;
    let h = height as i32;
    let r = (k / 2) as i32;
    let mut out = vec![0u8; rgba.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let mut acc = init;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let xx = clampi(x + dx, 0, w - 1);
                        let yy = clampi(y + dy, 0, h - 1);
                        acc = pick(acc, rgba[((yy * w + xx) * 4) as usize + c as usize]);
                    }
                }
                out[((y * w + x) * 4) as usize + c as usize] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_bg_replaces_transparent() {
        let rgba = vec![10, 20, 30, 0, 40, 50, 60, 255];
        let out = vector_fill_background(&rgba, 2, 1, [255, 0, 255, 255]);
        assert_eq!(&out.data[0..4], &[255, 0, 255, 255]); // transparent -> bg
        assert_eq!(&out.data[4..8], &[40, 50, 60, 255]); // opaque -> src
    }

    #[test]
    fn filters_run_and_keep_dims() {
        let (w, h) = (24u32, 24u32);
        let mut d = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            d[i * 4] = (i % 200) as u8;
            d[i * 4 + 1] = (i % 100) as u8;
            d[i * 4 + 2] = (i % 50) as u8;
            d[i * 4 + 3] = 255;
        }
        assert_eq!(bilateral_rgb(&d, w, h, 9).data.len(), d.len());
        assert_eq!(median_k(&d, w, h, 5).data.len(), d.len());
        assert_eq!(gaussian_k(&d, w, h, 5).data.len(), d.len());
        assert_eq!(morph_close_k(&d, w, h, 3).data.len(), d.len());
    }
}
