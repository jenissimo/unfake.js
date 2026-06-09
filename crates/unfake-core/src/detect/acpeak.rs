//! Autocorrelation peak detector (single-window acpeak).

use super::edge::peak_lag;

pub fn acpeak_detect(rgba: &[u8], width: u32, height: u32) -> u32 {
    let (col, row) = boundary_signals(rgba, width, height);
    reconcile_close(peak_lag(&col), peak_lag(&row))
}

fn boundary_signals(rgba: &[u8], width: u32, height: u32) -> (Vec<f64>, Vec<f64>) {
    let w = width as usize;
    let h = height as usize;
    let mut col = vec![0.0; w];
    let mut row = vec![0.0; h];

    for y in 0..h {
        for x in 1..w {
            let i = (y * w + x) * 4;
            let j = i - 4;
            for c in 0..4 {
                col[x] += (rgba[i + c] as i32 - rgba[j + c] as i32).unsigned_abs() as f64;
            }
        }
    }
    for x in 0..w {
        for y in 1..h {
            let i = (y * w + x) * 4;
            let j = ((y - 1) * w + x) * 4;
            for c in 0..4 {
                row[y] += (rgba[i + c] as i32 - rgba[j + c] as i32).unsigned_abs() as f64;
            }
        }
    }
    (col, row)
}

fn reconcile_close(a: u32, b: u32) -> u32 {
    let diff = (a as i32 - b as i32).unsigned_abs() as f64;
    if diff <= (a.min(b) as f64 * 0.12).max(1.0) {
        return a.min(b);
    }
    if a == b {
        return a;
    }
    let g = gcd(a, b);
    if g >= 2 { g } else { a.min(b) }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
