//! Content-adaptive (perceptual) downscaling — port of `opencv-ops.js`
//! contentAdaptiveCore (Öztireli–Gross EM). Works in CIE-Lab (standard sRGB/D65,
//! not OpenCV's exact Lab — output is perceptually equivalent and gets quantized
//! downstream). 2x2 symmetric eigendecomposition replaces SVD.js. Includes the
//! RATIO_CAP=3 pre-shrink so cost stays bounded on large upscales.

use crate::image::{resize_area, RgbaImage};

const NUM_ITERATIONS: usize = 5;
const RATIO_CAP: f64 = 3.0;

pub fn content_adaptive_downscale(
    rgba: &[u8],
    width: u32,
    height: u32,
    target_w: u32,
    target_h: u32,
) -> RgbaImage {
    let tw = target_w.max(1);
    let th = target_h.max(1);

    // Output alpha = area-resized source alpha.
    let alpha_src = resize_area(rgba, width, height, tw, th);

    // Full-res Lab planes.
    let (mut lp, mut ap, mut bp, mut wi, mut hi) = rgb_to_lab_planes(rgba, width, height);

    // Pre-shrink so the Gaussian kernel support stays ~RATIO_CAP.
    let rx0 = wi as f64 / tw as f64;
    let ry0 = hi as f64 / th as f64;
    if rx0.max(ry0) > RATIO_CAP {
        let sw = (tw as f64 * RATIO_CAP).round().max(tw as f64) as u32;
        let sh = (th as f64 * RATIO_CAP).round().max(th as f64) as u32;
        lp = resize_plane_area(&lp, wi, hi, sw, sh);
        ap = resize_plane_area(&ap, wi, hi, sw, sh);
        bp = resize_plane_area(&bp, wi, hi, sw, sh);
        wi = sw;
        hi = sh;
    }

    let out_lab = content_adaptive_core(&lp, &ap, &bp, wi, hi, tw, th);

    let mut out = vec![0u8; (tw * th * 4) as usize];
    for k in 0..(tw * th) as usize {
        let (r, g, b) = lab_to_rgb(out_lab[k * 3], out_lab[k * 3 + 1], out_lab[k * 3 + 2]);
        out[k * 4] = r;
        out[k * 4 + 1] = g;
        out[k * 4 + 2] = b;
        out[k * 4 + 3] = alpha_src.data[k * 4 + 3];
    }
    RgbaImage { data: out, width: tw, height: th }
}

fn content_adaptive_core(
    lp: &[f64],
    ap: &[f64],
    bp: &[f64],
    wi: u32,
    hi: u32,
    wo: u32,
    ho: u32,
) -> Vec<f64> {
    let (wi, hi) = (wi as usize, hi as usize);
    let (wo, ho) = (wo as usize, ho as usize);
    let rx = wi as f64 / wo as f64;
    let ry = hi as f64 / ho as f64;
    let r_avg = (rx + ry) / 2.0;
    let min_sv = 0.5;
    let max_sv = (0.5 * r_avg).max(1.0);
    let n_out = wo * ho;

    // [mu_x, mu_y], [s0,s1,s2,s3], [L,a,b]
    let mut mu = vec![[0.0f64; 2]; n_out];
    let mut sigma = vec![[0.0f64; 4]; n_out];
    let mut nu = vec![[0.0f64; 3]; n_out];
    for yk in 0..ho {
        for xk in 0..wo {
            let k = yk * wo + xk;
            mu[k] = [(xk as f64 + 0.5) * rx, (yk as f64 + 0.5) * ry];
            sigma[k] = [(rx / 3.0).powi(2), 0.0, 0.0, (ry / 3.0).powi(2)];
            nu[k] = [50.0, 0.0, 0.0];
        }
    }

    for _iter in 0..NUM_ITERATIONS {
        let mut gamma_sum = vec![1e-9f64; wi * hi];
        let mut w_ki: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_out];

        for k in 0..n_out {
            let [s0, s1, s2, s3] = sigma[k];
            let det = s0 * s3 - s1 * s2;
            let inv = 1.0 / (det + 1e-9);
            let si = [s3 * inv, -s1 * inv, -s2 * inv, s0 * inv];
            let [mx, my] = mu[k];
            let i_min_x = (mx - 2.0 * rx).floor().max(0.0) as usize;
            let i_max_x = ((mx + 2.0 * rx).ceil() as i64).clamp(0, wi as i64) as usize;
            let i_min_y = (my - 2.0 * ry).floor().max(0.0) as usize;
            let i_max_y = ((my + 2.0 * ry).ceil() as i64).clamp(0, hi as i64) as usize;
            let mut w_sum = 1e-9;
            let mut entries = Vec::new();
            for yi in i_min_y..i_max_y {
                for xi in i_min_x..i_max_x {
                    let dx = xi as f64 - mx;
                    let dy = yi as f64 - my;
                    let expo = dx * dx * si[0] + 2.0 * dx * dy * si[1] + dy * dy * si[3];
                    let w = (-0.5 * expo).exp();
                    if w > 1e-5 {
                        entries.push((yi * wi + xi, w));
                        w_sum += w;
                    }
                }
            }
            for (i, w) in entries.iter_mut() {
                *w /= w_sum;
                gamma_sum[*i] += *w;
            }
            w_ki[k] = entries;
        }

        for k in 0..n_out {
            let mut w_sum = 1e-9;
            let mut new_mu = [0.0f64; 2];
            let mut new_nu = [0.0f64; 3];
            for &(i, wk) in &w_ki[k] {
                let gamma = wk / gamma_sum[i];
                w_sum += gamma;
                let yi = (i / wi) as f64;
                let xi = (i % wi) as f64;
                new_mu[0] += gamma * xi;
                new_mu[1] += gamma * yi;
                new_nu[0] += gamma * lp[i];
                new_nu[1] += gamma * ap[i];
                new_nu[2] += gamma * bp[i];
            }
            for v in new_mu.iter_mut() {
                *v /= w_sum;
            }
            for v in new_nu.iter_mut() {
                *v /= w_sum;
            }
            let mut s = [0.0f64; 4];
            for &(i, wk) in &w_ki[k] {
                let gamma = wk / gamma_sum[i];
                let dx = (i % wi) as f64 - new_mu[0];
                let dy = (i / wi) as f64 - new_mu[1];
                s[0] += gamma * dx * dx;
                s[1] += gamma * dx * dy;
                s[3] += gamma * dy * dy;
            }
            s[0] /= w_sum;
            s[1] /= w_sum;
            s[3] /= w_sum;
            s[2] = s[1];
            mu[k] = new_mu;
            sigma[k] = clamp_covariance(s[0], s[1], s[3], min_sv, max_sv);
            nu[k] = new_nu;
        }
    }

    let mut out = vec![0.0f64; n_out * 3];
    for k in 0..n_out {
        out[k * 3] = nu[k][0];
        out[k * 3 + 1] = nu[k][1];
        out[k * 3 + 2] = nu[k][2];
    }
    out
}

/// Clamp eigenvalues of symmetric PSD [[s0,s1],[s1,s3]] to [min,max], rebuild.
fn clamp_covariance(s0: f64, s1: f64, s3: f64, min_sv: f64, max_sv: f64) -> [f64; 4] {
    let tr = s0 + s3;
    let disc = (((s0 - s3) * 0.5).powi(2) + s1 * s1).sqrt();
    let l1 = (tr * 0.5 + disc).clamp(min_sv, max_sv);
    let l2 = (tr * 0.5 - disc).clamp(min_sv, max_sv);
    if s1.abs() < 1e-12 {
        // Already diagonal: eigenvectors are axis-aligned.
        let (e0, e3) = if s0 >= s3 { (l1, l2) } else { (l2, l1) };
        return [e0, 0.0, 0.0, e3];
    }
    // Eigenvector for l1 (pre-clamp direction): (s1, lambda1 - s0).
    let lam1 = tr * 0.5 + disc;
    let (mut vx, mut vy) = (s1, lam1 - s0);
    let norm = (vx * vx + vy * vy).sqrt().max(1e-12);
    vx /= norm;
    vy /= norm;
    // M' = l1 (v⊗v) + l2 (v⊥⊗v⊥), v⊥ = (-vy, vx)
    let m00 = l1 * vx * vx + l2 * vy * vy;
    let m01 = (l1 - l2) * vx * vy;
    let m11 = l1 * vy * vy + l2 * vx * vx;
    [m00, m01, m01, m11]
}

// --- Lab (sRGB / D65) -------------------------------------------------------
fn rgb_to_lab_planes(rgba: &[u8], w: u32, h: u32) -> (Vec<f64>, Vec<f64>, Vec<f64>, u32, u32) {
    let n = (w * h) as usize;
    let (mut lp, mut ap, mut bp) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let p = i * 4;
        let (l, a, b) = rgb_to_lab(rgba[p], rgba[p + 1], rgba[p + 2]);
        lp[i] = l;
        ap[i] = a;
        bp[i] = b;
    }
    (lp, ap, bp, w, h)
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn fwd(t: f64) -> f64 {
    if t > 0.008856 {
        t.cbrt()
    } else {
        7.787 * t + 16.0 / 116.0
    }
}

fn inv(t: f64) -> f64 {
    let t3 = t * t * t;
    if t3 > 0.008856 {
        t3
    } else {
        (t - 16.0 / 116.0) / 7.787
    }
}

const XN: f64 = 0.95047;
const YN: f64 = 1.0;
const ZN: f64 = 1.08883;

fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let rl = srgb_to_linear(r as f64 / 255.0);
    let gl = srgb_to_linear(g as f64 / 255.0);
    let bl = srgb_to_linear(b as f64 / 255.0);
    let x = (0.4124 * rl + 0.3576 * gl + 0.1805 * bl) / XN;
    let y = (0.2126 * rl + 0.7152 * gl + 0.0722 * bl) / YN;
    let z = (0.0193 * rl + 0.1192 * gl + 0.9505 * bl) / ZN;
    let (fx, fy, fz) = (fwd(x), fwd(y), fwd(z));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

fn lab_to_rgb(l: f64, a: f64, b: f64) -> (u8, u8, u8) {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let x = XN * inv(fx);
    let y = YN * inv(fy);
    let z = ZN * inv(fz);
    let rl = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let gl = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let bl = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    let to_u8 = |c: f64| (linear_to_srgb(c) * 255.0).round().clamp(0.0, 255.0) as u8;
    (to_u8(rl), to_u8(gl), to_u8(bl))
}

/// Box (area) resize for a single f64 plane.
fn resize_plane_area(plane: &[f64], width: u32, height: u32, tw: u32, th: u32) -> Vec<f64> {
    let (wf, hf) = (width as f64, height as f64);
    let mut out = vec![0.0f64; (tw * th) as usize];
    for ty in 0..th {
        let y0 = ((ty as f64) * hf / th as f64).floor() as u32;
        let y1 = ((((ty + 1) as f64) * hf / th as f64).ceil() as u32).min(height).max(y0 + 1);
        for tx in 0..tw {
            let x0 = ((tx as f64) * wf / tw as f64).floor() as u32;
            let x1 = ((((tx + 1) as f64) * wf / tw as f64).ceil() as u32).min(width).max(x0 + 1);
            let mut acc = 0.0;
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    acc += plane[(sy * width + sx) as usize];
                    n += 1;
                }
            }
            out[(ty * tw + tx) as usize] = acc / n as f64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_roundtrip_is_close() {
        for &(r, g, b) in &[(0u8, 0u8, 0u8), (255, 255, 255), (200, 30, 90), (12, 200, 240)] {
            let (l, a, bb) = rgb_to_lab(r, g, b);
            let (r2, g2, b2) = lab_to_rgb(l, a, bb);
            assert!((r as i32 - r2 as i32).abs() <= 1, "r {r}->{r2}");
            assert!((g as i32 - g2 as i32).abs() <= 1, "g {g}->{g2}");
            assert!((b as i32 - b2 as i32).abs() <= 1, "b {b}->{b2}");
        }
    }

    #[test]
    fn downscale_runs_and_sizes() {
        let (w, h) = (48u32, 48u32);
        let mut d = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let c = if (x / 6 + y / 6) % 2 == 0 { 210 } else { 40 };
                d[i] = c;
                d[i + 1] = c;
                d[i + 2] = c;
                d[i + 3] = 255;
            }
        }
        let out = content_adaptive_downscale(&d, w, h, 8, 8);
        assert_eq!(out.width, 8);
        assert_eq!(out.height, 8);
        assert_eq!(out.data.len(), 8 * 8 * 4);
        assert!(out.data[3] > 0, "alpha preserved");
    }
}
