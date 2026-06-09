//! Palette quantization (replaces image-q). Auto palettes via `imagequant`
//! (libimagequant); a fixed palette is applied by nearest-color remap (memoized
//! per unique source color, like the old JS applyPaletteNearest).

use crate::image::RgbaImage;
use std::collections::HashMap;

pub struct QuantizeResult {
    pub image: RgbaImage,
    pub palette: Vec<[u8; 4]>,
}

pub fn quantize(
    rgba: &[u8],
    width: u32,
    height: u32,
    max_colors: u32,
    fixed_palette: &[[u8; 4]],
) -> QuantizeResult {
    if !fixed_palette.is_empty() {
        return remap_fixed(rgba, width, height, fixed_palette);
    }
    quantize_auto(rgba, width, height, max_colors)
}

fn quantize_auto(rgba: &[u8], width: u32, height: u32, max_colors: u32) -> QuantizeResult {
    use imagequant::{new, RGBA};

    let mut liq = new();
    let _ = liq.set_max_colors(max_colors.clamp(2, 256));
    let _ = liq.set_speed(1);

    let pixels: Vec<RGBA> = rgba
        .chunks_exact(4)
        .map(|c| RGBA { r: c[0], g: c[1], b: c[2], a: c[3] })
        .collect();

    let build = (|| {
        let mut img = liq.new_image(pixels, width as usize, height as usize, 0.0).ok()?;
        let mut res = liq.quantize(&mut img).ok()?;
        res.set_dithering_level(0.0).ok();
        let (pal, idx) = res.remapped(&mut img).ok()?;
        Some((pal, idx))
    })();

    let (pal, idx) = match build {
        Some(v) => v,
        None => return passthrough(rgba, width, height),
    };

    let palette: Vec<[u8; 4]> = pal.iter().map(|c| [c.r, c.g, c.b, c.a]).collect();
    let mut out = vec![0u8; rgba.len()];
    for (i, &id) in idx.iter().enumerate() {
        let c = palette.get(id as usize).copied().unwrap_or([0, 0, 0, 0]);
        out[i * 4..i * 4 + 4].copy_from_slice(&c);
    }
    QuantizeResult {
        image: RgbaImage { data: out, width, height },
        palette,
    }
}

fn remap_fixed(rgba: &[u8], width: u32, height: u32, palette: &[[u8; 4]]) -> QuantizeResult {
    let mut cache: HashMap<u32, usize> = HashMap::new();
    let mut out = vec![0u8; rgba.len()];
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        let key = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
        let idx = *cache.entry(key).or_insert_with(|| nearest(px, palette));
        out[i * 4..i * 4 + 4].copy_from_slice(&palette[idx]);
    }
    QuantizeResult {
        image: RgbaImage { data: out, width, height },
        palette: palette.to_vec(),
    }
}

fn nearest(px: &[u8], palette: &[[u8; 4]]) -> usize {
    let mut best = 0usize;
    let mut best_d = i64::MAX;
    for (i, c) in palette.iter().enumerate() {
        let dr = px[0] as i64 - c[0] as i64;
        let dg = px[1] as i64 - c[1] as i64;
        let db = px[2] as i64 - c[2] as i64;
        let da = px[3] as i64 - c[3] as i64;
        let d = dr * dr + dg * dg + db * db + da * da;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn passthrough(rgba: &[u8], width: u32, height: u32) -> QuantizeResult {
    QuantizeResult {
        image: RgbaImage { data: rgba.to_vec(), width, height },
        palette: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_reduces_palette() {
        // 8x8 with 4 distinct colors -> quantize to 2 yields <=2 palette colors.
        let (w, h) = (8u32, 8u32);
        let mut d = vec![0u8; (w * h * 4) as usize];
        let cols = [[255u8, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]];
        for i in 0..(w * h) as usize {
            d[i * 4..i * 4 + 4].copy_from_slice(&cols[i % 4]);
        }
        let r = quantize(&d, w, h, 2, &[]);
        assert!(r.palette.len() <= 2 && !r.palette.is_empty());
        assert_eq!(r.image.data.len(), d.len());
    }

    #[test]
    fn fixed_palette_remaps_to_nearest() {
        let rgba = vec![250, 5, 5, 255, 5, 250, 5, 255];
        let pal = [[255u8, 0, 0, 255], [0, 255, 0, 255]];
        let r = quantize(&rgba, 2, 1, 2, &pal);
        assert_eq!(&r.image.data[0..4], &[255, 0, 0, 255]);
        assert_eq!(&r.image.data[4..8], &[0, 255, 0, 255]);
        assert_eq!(r.palette.len(), 2);
    }
}
