//! Morphological open+close cleanup (parity with `utils.js` morphologicalCleanup:
//! MORPH_OPEN then MORPH_CLOSE with a 2x2 kernel). Per-channel (incl. alpha)
//! erode = local min, dilate = local max over a 2x2 neighborhood (anchor
//! top-left, border replicate). Approximate parity with OpenCV — this is a
//! cleanup filter, not a bit-exact transform.

use crate::image::RgbaImage;

pub fn morph_open_close(rgba: &[u8], width: u32, height: u32) -> RgbaImage {
    // open = erode -> dilate ; close = dilate -> erode
    let opened = dilate2(&erode2(rgba, width, height), width, height);
    let closed = erode2(&dilate2(&opened, width, height), width, height);
    RgbaImage {
        data: closed,
        width,
        height,
    }
}

fn erode2(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    neigh2(rgba, width, height, 255, |a, b| a.min(b))
}

fn dilate2(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    neigh2(rgba, width, height, 0, |a, b| a.max(b))
}

fn neigh2(rgba: &[u8], width: u32, height: u32, init: u8, pick: fn(u8, u8) -> u8) -> Vec<u8> {
    let w = width as i32;
    let h = height as i32;
    let mut out = vec![0u8; rgba.len()];
    for y in 0..h {
        for x in 0..w {
            let base = ((y * w + x) * 4) as usize;
            for c in 0..4 {
                let mut acc = init;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let xx = (x + dx).min(w - 1);
                        let yy = (y + dy).min(h - 1);
                        acc = pick(acc, rgba[((yy * w + xx) * 4) as usize + c]);
                    }
                }
                out[base + c] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_removes_single_speckle() {
        // 4x4 opaque black with one white speckle -> open should erase it.
        let (w, h) = (4u32, 4u32);
        let mut d = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            d[i * 4 + 3] = 255;
        }
        let s = ((1 * w + 1) * 4) as usize;
        d[s] = 255;
        d[s + 1] = 255;
        d[s + 2] = 255;
        let out = morph_open_close(&d, w, h);
        assert_eq!(out.data[s], 0, "speckle should be removed by open");
    }
}
