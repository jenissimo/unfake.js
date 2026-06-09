//! Tiled Sobel edge detector (canonical prod geometry + autocorr period).

const KX: [i32; 9] = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
const KY: [i32; 9] = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

pub fn tiled_edge_detect(rgba: &[u8], width: u32, height: u32) -> u32 {
    let gray = to_gray(rgba, width, height);
    let (w, h) = (width, height);
    let tile_w = (w / 3) as i32;
    let tile_h = (h / 3) as i32;
    let overlap_w = (tile_w as f64 * 0.25).floor() as i32;
    let overlap_h = (tile_h as f64 * 0.25).floor() as i32;

    if tile_w < 50 || tile_h < 50 {
        return single_region_edge_detect(rgba, width, height);
    }

    let mut all_scales = Vec::new();
    for ty in 0..3i32 {
        for tx in 0..3i32 {
            let x0 = (tx * tile_w - overlap_w).max(0);
            let y0 = (ty * tile_h - overlap_h).max(0);
            let rw = (tile_w + 2 * overlap_w).min(w as i32 - x0);
            let rh = (tile_h + 2 * overlap_h).min(h as i32 - y0);
            if rw < 30 || rh < 30 {
                continue;
            }
            if region_stddev(&gray, w, x0, y0, rw, rh) < 5.0 {
                continue;
            }
            let (h_scale, v_scale) = scales_from_roi(&gray, w, h, x0, y0, rw, rh);
            if h_scale > 1 {
                all_scales.push(h_scale);
            }
            if v_scale > 1 {
                all_scales.push(v_scale);
            }
        }
    }

    if all_scales.is_empty() {
        single_region_edge_detect(rgba, width, height)
    } else {
        mode(&all_scales).max(1)
    }
}

pub fn single_region_edge_detect(rgba: &[u8], width: u32, height: u32) -> u32 {
    let gray = to_gray(rgba, width, height);
    let (w, h) = (width, height);
    let x0 = (w as f64 * 0.125).floor() as i32;
    let y0 = (h as f64 * 0.125).floor() as i32;
    let rw = (w as f64 * 0.75).floor() as i32;
    let rh = (h as f64 * 0.75).floor() as i32;
    if rw < 3 || rh < 3 {
        return 1;
    }
    let (h_scale, v_scale) = scales_from_roi(&gray, w, h, x0, y0, rw, rh);
    if h_scale > 1 && v_scale > 1 && (h_scale as i32 - v_scale as i32).unsigned_abs() <= 2 {
        ((h_scale + v_scale) as f64 / 2.0).round() as u32
    } else {
        h_scale.max(v_scale).max(1)
    }
}

fn scales_from_roi(
    gray: &[f64],
    w: u32,
    h: u32,
    x0: i32,
    y0: i32,
    rw: i32,
    rh: i32,
) -> (u32, u32) {
    if rw < 3 || rh < 3 {
        return (1, 1);
    }
    let (prof_x, prof_y) = sobel_profiles(gray, w, h, x0, y0, rw, rh);
    (peak_lag(&prof_x), peak_lag(&prof_y))
}

fn to_gray(rgba: &[u8], width: u32, height: u32) -> Vec<f64> {
    let n = (width * height) as usize;
    let mut g = vec![0.0; n];
    for i in 0..n {
        let p = i * 4;
        g[i] = 0.299 * rgba[p] as f64 + 0.587 * rgba[p + 1] as f64 + 0.114 * rgba[p + 2] as f64;
    }
    g
}

fn reflect(i: i32, n: i32) -> i32 {
    if n == 1 {
        return 0;
    }
    let mut v = i;
    while v < 0 || v >= n {
        if v < 0 {
            v = -v;
        }
        if v >= n {
            v = 2 * (n - 1) - v;
        }
    }
    v
}

fn sobel_profiles(
    gray: &[f64],
    w: u32,
    h: u32,
    x0: i32,
    y0: i32,
    rw: i32,
    rh: i32,
) -> (Vec<f64>, Vec<f64>) {
    let wi = w as i32;
    let mut prof_x = vec![0.0; rw as usize];
    let mut prof_y = vec![0.0; rh as usize];
    for ly in 0..rh {
        for lx in 0..rw {
            let y = y0 + ly;
            let x = x0 + lx;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut k = 0usize;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let v = gray[(reflect(y + dy, h as i32) * wi + reflect(x + dx, wi)) as usize];
                    sx += v * KX[k] as f64;
                    sy += v * KY[k] as f64;
                    k += 1;
                }
            }
            prof_x[lx as usize] += sx.abs();
            prof_y[ly as usize] += sy.abs();
        }
    }
    (prof_x, prof_y)
}

fn region_stddev(gray: &[f64], w: u32, x0: i32, y0: i32, rw: i32, rh: i32) -> f64 {
    let wi = w as usize;
    let n = (rw * rh) as usize;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for ly in 0..rh {
        for lx in 0..rw {
            let v = gray[((y0 + ly) as usize * wi) + (x0 + lx) as usize];
            sum += v;
            sum2 += v * v;
        }
    }
    let mean = sum / n as f64;
    (sum2 / n as f64 - mean * mean).max(0.0).sqrt()
}

/// Autocorrelation period on a 1D profile (`peakLag` in cvfree.mjs).
pub fn peak_lag(sig: &[f64]) -> u32 {
    let n = sig.len();
    if n < 4 {
        return 1;
    }
    let max_lag = max_period(n);
    let mean: f64 = sig.iter().sum::<f64>() / n as f64;
    let s: Vec<f64> = sig.iter().map(|v| v - mean).collect();
    let norm: f64 = s.iter().map(|v| v * v).sum::<f64>().max(1e-9);
    let mut ac = vec![0.0; max_lag + 2];
    for lag in 2..=max_lag {
        let mut r = 0.0;
        for i in 0..n - lag {
            r += s[i] * s[i + lag];
        }
        ac[lag] = r / norm;
    }
    let mut start = 2usize;
    let cap = (max_lag / 2).max(3);
    while start < cap && ac[start + 1] <= ac[start] {
        start += 1;
    }
    let (mut gmax, mut gp) = (0.0f64, start);
    for p in start..=max_lag {
        if ac[p] > gmax {
            gmax = ac[p];
            gp = p;
        }
    }
    if gmax <= 0.0 {
        return 1;
    }
    let thr = 0.6 * gmax;
    for p in start..=max_lag {
        if ac[p] >= thr && ac[p] >= ac[p - 1] && ac[p] >= ac[p + 1] {
            return p as u32;
        }
    }
    gp as u32
}

fn max_period(n: usize) -> usize {
    (n / 8).min(128).max(2)
}

fn mode(values: &[u32]) -> u32 {
    use std::collections::HashMap;
    let mut counts = HashMap::new();
    let mut best = values[0];
    let mut best_n = 0u32;
    for &v in values {
        let c = counts.entry(v).or_insert(0u32);
        *c += 1;
        if *c > best_n {
            best_n = *c;
            best = v;
        }
    }
    best
}
