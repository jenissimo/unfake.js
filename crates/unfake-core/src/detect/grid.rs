//! Fractional grid estimation: non-integer cell size + phase per axis.
//!
//! Newer image models draw pixel art on an N×N logical grid and resample it to
//! the output size (200 cells → 2048 px = 10.24 px/cell; → 500 px = 2.5). The
//! integer detectors round that to 10 (the grid drifts a whole cell every ~4)
//! or double it to 5 (half the resolution is lost).
//!
//! Per axis: spectrum of the edge profile → lowest of the near-strongest peaks
//! that keeps the edges on its grid → sub-harmonic check (the true fundamental
//! has every harmonic below the peak present) → refine on the exact DFT →
//! phase from the fundamental's angle.

use std::f64::consts::PI;

/// Shortest period searched. Period ~2 is left to the integer detectors: near
/// Nyquist it is mostly noise (JPEG blocks, a nearest ×2 of a noisy image).
const PMIN: f64 = 2.1;
/// Harmonic counts as present at this share of the strongest peak.
const HARMONIC_RATIO: f64 = 0.3;
/// Peaks at this share of the strongest are candidates for the fundamental.
const NEAR_PEAK: f64 = 0.8;
/// A lower candidate replaces the peak only if it keeps this share of coherence.
const SUBHARMONIC_COHERENCE: f64 = 0.75;
/// Decision thresholds (calibrated on bghira + resampled synthetic GT).
const MIN_COHERENCE: f64 = 0.15;
/// Stronger evidence for periods JPEG block artifacts also produce.
const DCT_COHERENCE: f64 = 0.3;
const MIN_PROMINENCE: f64 = 5.0;
const MAX_AXIS_DISAGREEMENT: f64 = 0.02;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisGrid {
    /// Cell size in source pixels (may be non-integer).
    pub period: f64,
    /// A cell boundary lies at `offset + i * period`, `offset` in `[0, period)`.
    pub offset: f64,
    /// Share of edge energy aligned with the grid (0..1).
    pub coherence: f64,
    /// Spectral peak over the median of the searched band.
    pub prominence: f64,
}

impl AxisGrid {
    /// Integer grid of `round(period)` would give the same cell count within
    /// half a cell over `len` — the integer detectors already handle it.
    fn is_integer_like(&self, len: u32) -> bool {
        let r = self.period.round();
        r >= 1.0 && (len as f64 / self.period - len as f64 / r).abs() < 0.5
    }

    /// Cells whose center lies inside `[0, len)`.
    pub fn cells(&self, len: u32) -> u32 {
        let p = self.period;
        let first = (-self.offset / p - 0.5).ceil();
        let last = ((len as f64 - self.offset) / p - 0.5).ceil() - 1.0;
        (last - first + 1.0).max(0.0) as u32
    }

    /// Boundaries of those cells: `cells(len) + 1` values, the outer two may
    /// lie outside the image.
    pub fn bounds(&self, len: u32) -> Vec<f64> {
        let first = (-self.offset / self.period - 0.5).ceil();
        (0..=self.cells(len))
            .map(|i| self.offset + (first + i as f64) * self.period)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub x: AxisGrid,
    pub y: AxisGrid,
}

impl Grid {
    /// Mean cell size of both axes.
    pub fn scale(&self) -> f64 {
        (self.x.period + self.y.period) / 2.0
    }

    pub fn cells(&self, width: u32, height: u32) -> (u32, u32) {
        (self.x.cells(width), self.y.cells(height))
    }

    /// Integer scale with the snap crop offset as phase.
    pub fn integer(scale: u32, offset_x: u32, offset_y: u32) -> Self {
        let axis = |offset: u32| AxisGrid {
            period: scale as f64,
            offset: offset as f64,
            coherence: 0.0,
            prominence: 0.0,
        };
        Grid {
            x: axis(offset_x),
            y: axis(offset_y),
        }
    }

    pub fn bounds(&self, width: u32, height: u32) -> CellBounds {
        CellBounds {
            x: self.x.bounds(width),
            y: self.y.bounds(height),
        }
    }
}

/// Explicit cell boundaries per axis in source px, ascending — a regular grid
/// or one fitted to the drawn edges. Each consecutive pair is one output pixel;
/// the outer boundaries may lie outside the image.
#[derive(Debug, Clone, PartialEq)]
pub struct CellBounds {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl CellBounds {
    pub fn cells(&self) -> (u32, u32) {
        (
            self.x.len().saturating_sub(1) as u32,
            self.y.len().saturating_sub(1) as u32,
        )
    }
}

/// A confident non-integer grid, or `None` when the integer detectors should
/// be trusted (integer-like period, weak/disagreeing axes, no periodicity).
pub fn fractional_grid_detect(rgba: &[u8], width: u32, height: u32) -> Option<Grid> {
    let (prof_x, prof_y) = edge_profiles(rgba, width, height);
    let x = estimate_axis(&prof_x)?;
    let y = estimate_axis(&prof_y)?;
    let (a, b) = (x.period, y.period);
    let min_coherence = if is_dct_period(a) || is_dct_period(b) {
        DCT_COHERENCE
    } else {
        MIN_COHERENCE
    };
    if (a - b).abs() / a.max(b) > MAX_AXIS_DISAGREEMENT
        || x.coherence.min(y.coherence) < min_coherence
        || x.prominence.min(y.prominence) < MIN_PROMINENCE
        || (x.is_integer_like(width) && y.is_integer_like(height))
    {
        return None;
    }
    Some(Grid { x, y })
}

/// JPEG's 8×8 DCT leaves weak combs at periods 16/k (8/3, 3.2, 16/3…) in
/// otherwise smooth images.
fn is_dct_period(p: f64) -> bool {
    let k = (16.0 / p).round();
    k >= 1.0 && (p - 16.0 / k).abs() < 0.005
}

/// Grid for a user-given (possibly fractional) cell size: per-axis period
/// snapped to a whole number of cells when within a quarter cell, phase
/// estimated from the edges.
pub fn grid_for_scale(rgba: &[u8], width: u32, height: u32, scale: f64) -> Grid {
    let (prof_x, prof_y) = edge_profiles(rgba, width, height);
    Grid {
        x: axis_for_period(&prof_x, width, scale),
        y: axis_for_period(&prof_y, height, scale),
    }
}

fn axis_for_period(profile: &[f64], len: u32, scale: f64) -> AxisGrid {
    let n = len as f64;
    let cells = n / scale;
    let whole = cells.round();
    let period = if whole >= 1.0 && (cells - whole).abs() < 0.25 {
        n / whole
    } else {
        scale
    };
    let (offset, coherence) = phase_and_coherence(profile, period);
    AxisGrid {
        period,
        offset,
        coherence,
        prominence: 0.0,
    }
}

/// Column / row edge profiles; entry `i` is the boundary at coordinate `i + 1`.
/// Premultiplied RGB + alpha so transparent pixels' color doesn't add noise.
pub(super) fn edge_profiles(rgba: &[u8], width: u32, height: u32) -> (Vec<f64>, Vec<f64>) {
    let w = width as usize;
    let h = height as usize;
    let mut prof_x = vec![0.0f64; w.saturating_sub(1)];
    let mut prof_y = vec![0.0f64; h.saturating_sub(1)];
    let mut prev: Vec<[i32; 4]> = Vec::with_capacity(w);
    let mut cur: Vec<[i32; 4]> = Vec::with_capacity(w);
    for y in 0..h {
        cur.clear();
        cur.extend(rgba[y * w * 4..(y + 1) * w * 4].chunks_exact(4).map(premul));
        for x in 0..w.saturating_sub(1) {
            prof_x[x] += l1(&cur[x + 1], &cur[x]) as f64;
        }
        if y > 0 {
            let s: i64 = cur.iter().zip(&prev).map(|(a, b)| l1(a, b) as i64).sum();
            prof_y[y - 1] = s as f64;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    (prof_x, prof_y)
}

fn premul(p: &[u8]) -> [i32; 4] {
    let a = p[3] as i32;
    [
        p[0] as i32 * a / 255,
        p[1] as i32 * a / 255,
        p[2] as i32 * a / 255,
        a,
    ]
}

fn l1(a: &[i32; 4], b: &[i32; 4]) -> i32 {
    (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs() + (a[3] - b[3]).abs()
}

/// Period + phase of the boundary comb along one axis.
fn estimate_axis(profile: &[f64]) -> Option<AxisGrid> {
    let m = profile.len();
    if m < 16 {
        return None;
    }
    let n = (m + 1) as f64;
    let pmax = (n / 6.0).clamp(4.0, 160.0);
    let mean = profile.iter().sum::<f64>() / m as f64;
    let windowed: Vec<f64> = profile
        .iter()
        .enumerate()
        .map(|(i, v)| (v - mean) * (0.5 - 0.5 * (2.0 * PI * i as f64 / (m - 1) as f64).cos()))
        .collect();
    let spec = Spectrum::new(&windowed);

    let (fmin, fmax) = (1.0 / pmax, 1.0 / PMIN);
    let (k0, k1) = (spec.bin_ceil(fmin), spec.bin_floor(fmax));
    if k1 <= k0 {
        return None;
    }
    let band = &spec.mag[k0..=k1];
    let (peak_i, &amax) = band.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    if amax <= 0.0 {
        return None;
    }
    let mut sorted = band.to_vec();
    sorted.sort_by(f64::total_cmp);
    let prominence = amax / sorted[sorted.len() / 2].max(1e-9);

    // Near-equal peaks with equal coherence are one comb seen twice: edges on
    // every 2nd/3rd pixel only (a nearest ×2/×3 upscale) mirror the spectrum,
    // so 1/p and 1/2 − 1/p look alike. The lowest one is the cell size.
    let argmax_f = (k0 + peak_i) as f64 / spec.n_fft as f64;
    let coh_max = coherence_at(profile, argmax_f);
    let last = spec.mag.len() - 1;
    let fs = (k0..=k1)
        .filter(|&k| {
            let v = spec.mag[k];
            v >= NEAR_PEAK * amax && v >= spec.mag[k - 1] && v >= spec.mag[(k + 1).min(last)]
        })
        .map(|k| k as f64 / spec.n_fft as f64)
        .find(|&f| coherence_at(profile, f) >= SUBHARMONIC_COHERENCE * coh_max)
        .unwrap_or(argmax_f);

    // Largest divisor d with every harmonic k·fs/d (k < d) present: sharp
    // integer combs have equal harmonics and the argmax may land on any of them.
    // The coarser grid must also keep the edges on its lines — a nearest-
    // resampled 2.5 comb (cells 2,3,2,3 px) has a strong alias at period 5,
    // but half its edges fall mid-cell there.
    let floor = HARMONIC_RATIO * amax;
    let coh_fs = coherence_at(profile, fs);
    let mut best_d = 1usize;
    for d in 2..=(fs / fmin).floor() as usize {
        let f0 = fs / d as f64;
        if (1..d).all(|k| spec.local(k as f64 * f0) >= floor)
            && coherence_at(profile, f0) >= SUBHARMONIC_COHERENCE * coh_fs
        {
            best_d = d;
        }
    }
    let f0 = fs / best_d as f64;

    let harmonics: Vec<f64> = (1..=8)
        .map(|k| k as f64)
        .filter(|&k| k * f0 < 0.5 && spec.local(k * f0) >= floor)
        .collect();
    let score = |fr: f64| -> f64 { harmonics.iter().map(|&k| dft_mag(&windowed, k * fr)).sum() };

    let (mut lo, mut hi) = (f0 - 1.5 / m as f64, f0 + 1.5 / m as f64);
    let mut best_f = f0;
    for _ in 0..6 {
        let step = (hi - lo) / 32.0;
        let mut best_v = f64::MIN;
        for i in 0..=32 {
            let f = lo + step * i as f64;
            let v = score(f);
            if v > best_v {
                best_v = v;
                best_f = f;
            }
        }
        lo = best_f - 2.0 * step;
        hi = best_f + 2.0 * step;
    }

    let mut period = 1.0 / best_f;
    // Resampled from a whole N-cell canvas: lock to exactly n / N.
    let cells = n / period;
    let whole = cells.round();
    if whole >= 1.0 && (cells - whole).abs() < 0.25 && score(whole / n) >= 0.97 * score(best_f) {
        period = n / whole;
    }

    let (offset, coherence) = phase_and_coherence(profile, period);
    Some(AxisGrid {
        period,
        offset,
        coherence,
        prominence,
    })
}

fn coherence_at(profile: &[f64], f: f64) -> f64 {
    phase_and_coherence(profile, 1.0 / f).1
}

/// Boundary phase in `[0, period)` and the aligned share of edge energy.
fn phase_and_coherence(profile: &[f64], period: f64) -> (f64, f64) {
    let total: f64 = profile.iter().sum();
    if total <= 0.0 {
        return (0.0, 0.0);
    }
    let (re, im) = dft(profile, 1.0 / period);
    let coherence = (re * re + im * im).sqrt() / total;
    let offset = (-im.atan2(re) * period / (2.0 * PI)).rem_euclid(period);
    (offset, coherence)
}

/// Σ sig[i]·e^{-2πi·f·(i+1)} via a rotating phasor.
fn dft(sig: &[f64], f: f64) -> (f64, f64) {
    let w = -2.0 * PI * f;
    let (sw, cw) = w.sin_cos();
    let (mut zr, mut zi) = (cw, sw);
    let (mut re, mut im) = (0.0, 0.0);
    for (i, &v) in sig.iter().enumerate() {
        re += v * zr;
        im += v * zi;
        let nr = zr * cw - zi * sw;
        zi = zr * sw + zi * cw;
        zr = nr;
        if i % 1024 == 1023 {
            // Keep the phasor on the unit circle over long profiles.
            let norm = (zr * zr + zi * zi).sqrt();
            zr /= norm;
            zi /= norm;
        }
    }
    (re, im)
}

fn dft_mag(sig: &[f64], f: f64) -> f64 {
    let (re, im) = dft(sig, f);
    (re * re + im * im).sqrt()
}

/// Magnitude spectrum, zero-padded ×8 for peak resolution.
struct Spectrum {
    mag: Vec<f64>,
    n_fft: usize,
    /// Unpadded length — sets the main-lobe width.
    m: usize,
}

impl Spectrum {
    fn new(sig: &[f64]) -> Self {
        let m = sig.len();
        let n_fft = (m * 8).next_power_of_two();
        let mut re = vec![0.0; n_fft];
        let mut im = vec![0.0; n_fft];
        re[..m].copy_from_slice(sig);
        fft(&mut re, &mut im);
        let mag = (0..=n_fft / 2)
            .map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt())
            .collect();
        Self { mag, n_fft, m }
    }

    fn bin_ceil(&self, f: f64) -> usize {
        ((f * self.n_fft as f64).ceil() as usize).min(self.mag.len() - 1)
    }

    fn bin_floor(&self, f: f64) -> usize {
        ((f * self.n_fft as f64).floor() as usize).min(self.mag.len() - 1)
    }

    /// Peak magnitude within the main lobe around `f`.
    fn local(&self, f: f64) -> f64 {
        let half = 1.5 / self.m as f64;
        let lo = self.bin_ceil((f - half).max(0.0));
        let hi = self.bin_floor(f + half);
        if hi < lo {
            return 0.0;
        }
        self.mag[lo..=hi].iter().copied().fold(0.0, f64::max)
    }
}

/// In-place iterative radix-2 FFT (`re.len()` must be a power of two).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let (sw, cw) = (-2.0 * PI / len as f64).sin_cos();
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let a = start + k;
                let b = a + len / 2;
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nr = cr * cw - ci * sw;
                ci = cr * sw + ci * cw;
                cr = nr;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic "sprite": hashed colors per cell, `cells` cells resampled
    /// nearest-style to `size` px (boundaries at multiples of size/cells).
    fn fractional_sprite(cells: u32, size: u32) -> Vec<u8> {
        let p = size as f64 / cells as f64;
        let mut data = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let cx = (((x as f64 + 0.5) / p) as u32).min(cells - 1);
                let cy = (((y as f64 + 0.5) / p) as u32).min(cells - 1);
                let hsh = (cx.wrapping_mul(73_856_093) ^ cy.wrapping_mul(19_349_663)) % 5;
                let i = ((y * size + x) * 4) as usize;
                data[i] = (hsh * 60) as u8;
                data[i + 1] = (255 - hsh * 50) as u8;
                data[i + 2] = (hsh * 23 % 256) as u8;
                data[i + 3] = 255;
            }
        }
        data
    }

    #[test]
    fn fft_matches_dft() {
        let sig: Vec<f64> = (0..64).map(|i| ((i * 7) % 11) as f64).collect();
        let mut re = sig.clone();
        let mut im = vec![0.0; 64];
        fft(&mut re, &mut im);
        for k in [1usize, 5, 17, 31] {
            let (dr, di) = dft(&sig, k as f64 / 64.0);
            // dft() phases from position 1: rotate back by one sample.
            let (s, c) = (2.0 * PI * k as f64 / 64.0).sin_cos();
            let (er, ei) = (dr * c - di * s, dr * s + di * c);
            assert!((er - re[k]).abs() < 1e-9 && (ei - im[k]).abs() < 1e-9);
        }
    }

    #[test]
    fn detects_10_24_on_2048() {
        let data = fractional_sprite(200, 2048);
        let g = fractional_grid_detect(&data, 2048, 2048).expect("fractional grid");
        assert!((g.x.period - 10.24).abs() < 1e-9, "{:?}", g.x);
        assert!((g.y.period - 10.24).abs() < 1e-9, "{:?}", g.y);
        assert_eq!(g.cells(2048, 2048), (200, 200));
    }

    #[test]
    fn detects_2_5_on_500() {
        let data = fractional_sprite(200, 500);
        let g = fractional_grid_detect(&data, 500, 500).expect("fractional grid");
        assert!((g.scale() - 2.5).abs() < 1e-9, "{g:?}");
        assert_eq!(g.cells(500, 500), (200, 200));
    }

    #[test]
    fn nearest_2x_of_fractional_keeps_the_cell_size() {
        // Edges only on even pixels mirror the spectrum (1/p ↔ 1/2 − 1/p).
        let (size, cells) = (1024u32, 200u32);
        let src = fractional_sprite(cells, size);
        let big = size * 2;
        let mut data = vec![0u8; (big * big * 4) as usize];
        for y in 0..big {
            for x in 0..big {
                let s = (((y / 2) * size + x / 2) * 4) as usize;
                let d = ((y * big + x) * 4) as usize;
                data[d..d + 4].copy_from_slice(&src[s..s + 4]);
            }
        }
        let g = fractional_grid_detect(&data, big, big).expect("fractional grid");
        assert!((g.scale() - 10.24).abs() < 1e-9, "{g:?}");
    }

    #[test]
    fn integer_grid_is_left_to_integer_detectors() {
        let data = fractional_sprite(64, 512);
        assert_eq!(fractional_grid_detect(&data, 512, 512), None);
    }

    #[test]
    fn manual_scale_snaps_to_whole_cells() {
        let data = fractional_sprite(200, 2048);
        let g = grid_for_scale(&data, 2048, 2048, 10.24);
        assert_eq!(g.cells(2048, 2048), (200, 200));
        assert!(
            g.x.offset < 0.5 || g.x.offset > g.x.period - 0.5,
            "{:?}",
            g.x
        );
    }
}
