//! Grid phase / crop offset (parity with `experiments/lib/cvfree.mjs`).

use crate::image::RgbaImage;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CropOffset {
    pub x: u32,
    pub y: u32,
}

pub fn find_optimal_crop_offset(rgba: &[u8], width: u32, height: u32, scale: u32) -> CropOffset {
    if scale <= 1 {
        return CropOffset { x: 0, y: 0 };
    }
    let w = width as usize;
    let h = height as usize;
    let mut gray = vec![0.0f64; w * h];
    for i in 0..w * h {
        let p = i * 4;
        gray[i] = (0.299 * rgba[p] as f64 + 0.587 * rgba[p + 1] as f64 + 0.114 * rgba[p + 2] as f64)
            .round();
    }

    const KX: [i32; 9] = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const KY: [i32; 9] = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    let mut prof_x = vec![0.0f64; w];
    let mut prof_y = vec![0.0f64; h];
    for y in 0..h {
        for x in 0..w {
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut k = 0usize;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let v = gray[reflect(y as i32 + dy, h as i32) * w + reflect(x as i32 + dx, w as i32)];
                    sx += v * KX[k] as f64;
                    sy += v * KY[k] as f64;
                    k += 1;
                }
            }
            prof_x[x] += sx.abs();
            prof_y[y] += sy.abs();
        }
    }

    CropOffset {
        x: best_offset(&prof_x, scale),
        y: best_offset(&prof_y, scale),
    }
}

fn reflect(i: i32, n: i32) -> usize {
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
    v as usize
}

fn best_offset(profile: &[f64], scale: u32) -> u32 {
    let s = scale as usize;
    let mut best = 0u32;
    let mut max_score = -1.0f64;
    for offset in 0..s {
        let mut score = 0.0;
        let mut i = offset;
        while i < profile.len() {
            score += profile[i];
            i += s;
        }
        if score > max_score {
            max_score = score;
            best = offset as u32;
        }
    }
    best
}

pub fn snap_to_grid(rgba: &[u8], width: u32, height: u32, scale: u32) -> RgbaImage {
    let crop = find_optimal_crop_offset(rgba, width, height, scale);
    let new_w = ((width.saturating_sub(crop.x)) / scale) * scale;
    let new_h = ((height.saturating_sub(crop.y)) / scale) * scale;
    if new_w < scale || new_h < scale {
        return RgbaImage {
            data: rgba.to_vec(),
            width,
            height,
        };
    }
    RgbaImage::from_raw(rgba.to_vec(), width, height)
        .and_then(|img| img.crop(crop.x, crop.y, new_w, new_h))
        .unwrap_or(RgbaImage {
            data: rgba.to_vec(),
            width,
            height,
        })
}
