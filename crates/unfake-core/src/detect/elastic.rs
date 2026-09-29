//! Elastic grid: cell boundaries fitted to the drawn edges.
//!
//! Models that *draw* pixel art, rather than resample a canvas, jitter the cell
//! size — 40..49 px columns around a 43 px period — so any fixed grid drifts
//! half a cell somewhere and doubles or drops a pixel there. Per axis, a DP on
//! a ¼-px lattice picks boundaries that sit on edges while keeping every cell
//! within 0.75..1.3 of the period. The fit replaces the regular grid only on an
//! axis where it captures clearly more edge energy; clean grids stay regular.

use super::grid::{edge_profiles, CellBounds, Grid};

/// Boundary positions per source pixel (¼ px): a fractional period costs no
/// rounding bias, so flat stretches keep the regular cell count.
const LATTICE: usize = 4;
/// Smaller cells: edges on nearly every pixel, nothing to fit.
const MIN_PERIOD: f64 = 8.0;
/// Edge capture the fit must reach over the regular grid (regular grids and
/// resampled canvases stay below ~1.46; hand-drawn samples reach 2..4).
const MIN_GAIN: f64 = 1.5;
const MIN_SPACING: f64 = 0.75;
const MAX_SPACING: f64 = 1.3;
/// A cell off the period by this share costs as much as a strong edge earns.
const SLACK: f64 = 0.25;

/// Boundaries fitted to the edges on the axes that need it (the other axis
/// keeps `grid`), or `None` when the regular grid already fits both axes.
pub fn elastic_bounds(
    rgba: &[u8],
    width: u32,
    height: u32,
    grid: &Grid,
) -> Option<(CellBounds, [bool; 2])> {
    if grid.x.period < MIN_PERIOD && grid.y.period < MIN_PERIOD {
        return None;
    }
    let (prof_x, prof_y) = edge_profiles(rgba, width, height);
    let regular = grid.bounds(width, height);
    let fit_x = fit_axis(&prof_x, width, grid.x.period, &regular.x);
    let fit_y = fit_axis(&prof_y, height, grid.y.period, &regular.y);
    let adapted = [fit_x.is_some(), fit_y.is_some()];
    if !adapted[0] && !adapted[1] {
        return None;
    }
    Some((
        CellBounds {
            x: fit_x.unwrap_or(regular.x),
            y: fit_y.unwrap_or(regular.y),
        },
        adapted,
    ))
}

fn fit_axis(profile: &[f64], len: u32, period: f64, regular: &[f64]) -> Option<Vec<f64>> {
    if period < MIN_PERIOD || profile.is_empty() {
        return None;
    }
    let edges = edge_strength(profile, len);
    let fitted = fit_boundaries(&edges, len, period);
    let gain = capture(&edges, &fitted) / capture(&edges, regular).max(1e-9);
    (gain >= MIN_GAIN).then_some(fitted)
}

/// Edge strength at integer boundary coordinates `0..=len` (entry `c` is the
/// boundary between pixels `c - 1` and `c`), with half credit one pixel off
/// for edges spread over two pixels, normalized to its 90th percentile.
fn edge_strength(profile: &[f64], len: u32) -> Vec<f64> {
    let n = len as usize;
    let mut raw = vec![0.0; n + 1];
    raw[1..n].copy_from_slice(&profile[..n - 1]);
    let mut e: Vec<f64> = (0..=n)
        .map(|c| {
            let left = if c > 0 { raw[c - 1] } else { 0.0 };
            let right = if c < n { raw[c + 1] } else { 0.0 };
            raw[c].max(0.5 * left.max(right))
        })
        .collect();
    let mut inner: Vec<f64> = e[1..n].to_vec();
    inner.sort_by(f64::total_cmp);
    let p90 = inner
        .get(inner.len() * 9 / 10)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9);
    e.iter_mut().for_each(|v| *v /= p90);
    e
}

/// Share of the edge energy that lies on the boundaries.
fn capture(edges: &[f64], bounds: &[f64]) -> f64 {
    let n = edges.len() - 1;
    let total: f64 = edges[1..n].iter().sum();
    let hit: f64 = bounds
        .iter()
        .map(|b| b.round())
        .filter(|&c| c > 0.0 && (c as usize) < n)
        .map(|c| edges[c as usize])
        .sum();
    hit / total.max(1e-9)
}

/// DP over the ¼-px lattice: boundaries (the first ≤ 0, the last ≥ len) that
/// maximize edge strength on them minus a quadratic cost per cell for leaving
/// the period. Returns the boundaries of cells whose center is inside the image.
fn fit_boundaries(edges: &[f64], len: u32, period: f64) -> Vec<f64> {
    let q = LATTICE as f64;
    let n = len as usize;
    let steps = n * LATTICE;
    let smin = ((MIN_SPACING * period * q).floor() as usize).max(1);
    let smax = ((MAX_SPACING * period * q).ceil() as usize).max(smin + 1);
    let lambda = 1.0 / (SLACK * period).powi(2);
    let penalty: Vec<f64> = (smin..=smax)
        .map(|s| lambda * (s as f64 / q - period).powi(2))
        .collect();
    // Edge strength on the lattice (linear between pixel boundaries).
    let gain = |t: usize| -> f64 {
        if t == 0 || t >= steps {
            return 0.0;
        }
        let (c, f) = (t / LATTICE, (t % LATTICE) as f64 / q);
        edges[c] * (1.0 - f) + edges[(c + 1).min(n)] * f
    };

    // Index j = lattice position + smax, positions from -smax to steps + smax.
    let size = steps + 2 * smax + 1;
    let mut best = vec![f64::NEG_INFINITY; size];
    let mut prev = vec![usize::MAX; size];
    for b in best.iter_mut().take(smax + 1).skip(1) {
        *b = 0.0; // the first boundary: anywhere in (-smax, 0]
    }
    for j in smax + 1..size {
        let mut top = f64::NEG_INFINITY;
        let mut arg = usize::MAX;
        for (k, s) in (smin..=smax).enumerate() {
            if s > j {
                break;
            }
            let v = best[j - s] - penalty[k];
            if v > top {
                top = v;
                arg = j - s;
            }
        }
        if arg != usize::MAX {
            best[j] = top + gain(j - smax);
            prev[j] = arg;
        }
    }
    let mut j = (steps + smax..size)
        .max_by(|&a, &b| best[a].total_cmp(&best[b]))
        .unwrap_or(size - 1);
    let mut bounds = vec![(j as f64 - smax as f64) / q];
    while prev[j] != usize::MAX {
        j = prev[j];
        bounds.push((j as f64 - smax as f64) / q);
    }
    bounds.reverse();

    // Keep the run of cells whose center lies inside [0, len).
    let inside = |i: usize| {
        let c = (bounds[i] + bounds[i + 1]) / 2.0;
        c >= 0.0 && c < len as f64
    };
    let first = (0..bounds.len() - 1).find(|&i| inside(i));
    let last = (0..bounds.len() - 1).rev().find(|&i| inside(i));
    match (first, last) {
        (Some(a), Some(b)) => bounds[a..=b + 1].to_vec(),
        _ => bounds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color(cx: u32, cy: u32) -> [u8; 4] {
        let h = (cx.wrapping_mul(73_856_093) ^ cy.wrapping_mul(19_349_663)) % 7;
        [
            (h * 36) as u8,
            (250 - h * 30) as u8,
            (h * 91 % 256) as u8,
            255,
        ]
    }

    /// Cells with the given column widths / row heights, hashed colors.
    fn drawn(widths: &[u32], heights: &[u32]) -> (Vec<u8>, u32, u32) {
        let (w, h) = (widths.iter().sum::<u32>(), heights.iter().sum::<u32>());
        let col = |x: u32| {
            let mut acc = 0;
            widths.iter().position(|&cw| {
                acc += cw;
                x < acc
            })
        };
        let row = |y: u32| {
            let mut acc = 0;
            heights.iter().position(|&ch| {
                acc += ch;
                y < acc
            })
        };
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&color(col(x).unwrap() as u32, row(y).unwrap() as u32));
            }
        }
        (data, w, h)
    }

    /// 40..49 px jitter around a 43 px period, like the hand-drawn duck sample.
    fn jitter(n: usize, seed: u32) -> Vec<u32> {
        (0..n)
            .map(|i| 40 + (((i as u32 + seed).wrapping_mul(2_654_435_761) >> 7) % 10))
            .collect()
    }

    #[test]
    fn follows_hand_drawn_cells() {
        let (widths, heights) = (jitter(24, 1), jitter(24, 7));
        let (data, w, h) = drawn(&widths, &heights);
        let (bounds, adapted) =
            elastic_bounds(&data, w, h, &Grid::integer(43, 0, 0)).expect("elastic fit");
        assert_eq!(adapted, [true, true]);
        assert_eq!(bounds.cells(), (24, 24));
        // Inner boundaries sit on the drawn edges (the outer two may lie outside).
        let mut edge = 0u32;
        for (b, cw) in bounds.x[1..bounds.x.len() - 1].iter().zip(&widths) {
            edge += cw;
            assert!(
                (b - edge as f64).abs() <= 1.0,
                "boundary {b} vs edge {edge}"
            );
        }
    }

    #[test]
    fn regular_grids_stay_regular() {
        let (data, w, h) = drawn(&[16; 32], &[16; 32]);
        assert_eq!(elastic_bounds(&data, w, h, &Grid::integer(16, 0, 0)), None);
    }
}
