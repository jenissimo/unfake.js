//! Palette quantization (replaces image-q). Auto palettes via `imagequant`
//! (libimagequant); a fixed palette is applied by nearest-color remap (memoized
//! per unique source color, like the old JS applyPaletteNearest).

use crate::image::RgbaImage;
use std::collections::HashMap;

pub struct QuantizeResult {
    pub image: RgbaImage,
    pub palette: Vec<[u8; 4]>,
}

/// Options for [`quantize_rgb`]: RGB-only quantization with an optional
/// reserved (chroma) color that survives bit-exactly.
#[derive(Debug, Clone, Default)]
pub struct QuantizeRgbOptions {
    /// Total palette budget, including the reserved color when set.
    pub max_colors: u32,
    /// Fixed palette (RGB). When non-empty, colors are remapped to the nearest
    /// entry instead of building an adaptive palette; `max_colors` is ignored.
    pub fixed_palette: Vec<[u8; 3]>,
    /// Pixels matching this color (within `reserve_tolerance` per channel) are
    /// excluded from palette selection and emitted exactly as this color.
    pub reserve_color: Option<[u8; 3]>,
    /// Per-channel tolerance for matching `reserve_color` (0 = exact).
    pub reserve_tolerance: u8,
}

pub struct QuantizeRgbResult {
    pub image: RgbaImage,
    /// Final palette (RGB). Contains the reserved color as its own entry when
    /// `reserve_color` was set.
    pub palette: Vec<[u8; 3]>,
}

/// Quantize RGB channels only; alpha is copied through untouched.
///
/// Reserved pixels (== `reserve_color` within tolerance) do not participate in
/// adaptive palette selection and come out bit-exact, so a chroma background
/// like `#ff00ff` survives quantization. The reserved color occupies one slot
/// of the `max_colors` budget.
pub fn quantize_rgb(
    rgba: &[u8],
    width: u32,
    height: u32,
    opts: &QuantizeRgbOptions,
) -> QuantizeRgbResult {
    let is_reserved = |px: &[u8]| -> bool {
        match opts.reserve_color {
            Some(rc) => {
                let t = opts.reserve_tolerance as i16;
                (px[0] as i16 - rc[0] as i16).abs() <= t
                    && (px[1] as i16 - rc[1] as i16).abs() <= t
                    && (px[2] as i16 - rc[2] as i16).abs() <= t
            }
            None => false,
        }
    };

    let palette: Vec<[u8; 3]> = if !opts.fixed_palette.is_empty() {
        opts.fixed_palette.clone()
    } else {
        let budget = if opts.reserve_color.is_some() {
            opts.max_colors.clamp(2, 256).saturating_sub(1).max(1)
        } else {
            opts.max_colors.clamp(1, 256)
        };
        adaptive_rgb_palette(rgba, budget, &is_reserved)
    };

    let mut out = vec![0u8; rgba.len()];
    let mut cache: HashMap<u32, [u8; 3]> = HashMap::new();
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        let rgb = if is_reserved(px) {
            opts.reserve_color.unwrap()
        } else if palette.is_empty() {
            [px[0], px[1], px[2]]
        } else {
            let key = u32::from_le_bytes([px[0], px[1], px[2], 0]);
            *cache
                .entry(key)
                .or_insert_with(|| palette[nearest_rgb(px, &palette)])
        };
        let o = i * 4;
        out[o..o + 3].copy_from_slice(&rgb);
        out[o + 3] = px[3]; // alpha untouched
    }

    let mut final_palette = palette;
    if let Some(rc) = opts.reserve_color {
        if !final_palette.contains(&rc) {
            final_palette.push(rc);
        }
    }

    QuantizeRgbResult {
        image: RgbaImage { data: out, width, height },
        palette: final_palette,
    }
}

/// Build an adaptive RGB palette from non-reserved pixels via a libimagequant
/// histogram, so reserved pixels contribute nothing to color selection.
fn adaptive_rgb_palette(
    rgba: &[u8],
    max_colors: u32,
    is_reserved: &dyn Fn(&[u8]) -> bool,
) -> Vec<[u8; 3]> {
    use imagequant::{new, HistogramEntry, Histogram, RGBA};

    let mut counts: HashMap<[u8; 3], u32> = HashMap::new();
    for px in rgba.chunks_exact(4) {
        if !is_reserved(px) {
            *counts.entry([px[0], px[1], px[2]]).or_insert(0) += 1;
        }
    }
    if counts.is_empty() {
        return Vec::new();
    }
    if counts.len() as u32 <= max_colors {
        let mut pal: Vec<[u8; 3]> = counts.into_keys().collect();
        pal.sort_unstable();
        return pal;
    }

    let mut liq = new();
    let _ = liq.set_max_colors(max_colors);
    let _ = liq.set_speed(1);
    let entries: Vec<HistogramEntry> = counts
        .iter()
        .map(|(c, &n)| HistogramEntry {
            color: RGBA { r: c[0], g: c[1], b: c[2], a: 255 },
            count: n,
        })
        .collect();

    let build = (|| {
        let mut hist = Histogram::new(&liq);
        hist.add_colors(&entries, 0.0).ok()?;
        let mut res = hist.quantize(&liq).ok()?;
        Some(res.palette_vec())
    })();

    match build {
        Some(pal) => pal.iter().map(|c| [c.r, c.g, c.b]).collect(),
        // Fallback: keep the most frequent colors.
        None => {
            let mut v: Vec<([u8; 3], u32)> = counts.into_iter().collect();
            v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            v.truncate(max_colors as usize);
            v.into_iter().map(|(c, _)| c).collect()
        }
    }
}

fn nearest_rgb(px: &[u8], palette: &[[u8; 3]]) -> usize {
    let mut best = 0usize;
    let mut best_d = i64::MAX;
    for (i, c) in palette.iter().enumerate() {
        let dr = px[0] as i64 - c[0] as i64;
        let dg = px[1] as i64 - c[1] as i64;
        let db = px[2] as i64 - c[2] as i64;
        let d = dr * dr + dg * dg + db * db;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
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
    fn rgb_reserve_color_survives_bit_exact() {
        // 8x8: many distinct colors + a #ff00ff chroma block with alpha 128.
        let (w, h) = (8u32, 8u32);
        let mut d = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            let v = (i * 7 % 200) as u8;
            d[i * 4..i * 4 + 4].copy_from_slice(&[v, 255 - v, (i % 13) as u8 * 19, 255]);
        }
        for i in 0..8usize {
            d[i * 4..i * 4 + 4].copy_from_slice(&[255, 0, 255, 128]);
        }
        let r = quantize_rgb(
            &d,
            w,
            h,
            &QuantizeRgbOptions {
                max_colors: 4,
                reserve_color: Some([255, 0, 255]),
                ..Default::default()
            },
        );
        assert!(r.palette.len() <= 4);
        assert!(r.palette.contains(&[255, 0, 255]));
        // Chroma pixels: RGB bit-exact, alpha untouched.
        for i in 0..8usize {
            assert_eq!(&r.image.data[i * 4..i * 4 + 4], &[255, 0, 255, 128]);
        }
        // Non-reserved pixels never become the chroma color; alpha preserved.
        let mut uniq = std::collections::HashSet::new();
        for (i, px) in r.image.data.chunks_exact(4).enumerate() {
            uniq.insert([px[0], px[1], px[2]]);
            assert_eq!(px[3], d[i * 4 + 3]);
            if i >= 8 {
                assert_ne!([px[0], px[1], px[2]], [255u8, 0, 255]);
            }
        }
        assert!(uniq.len() <= 4);
    }

    #[test]
    fn rgb_fixed_palette_with_reserve() {
        let rgba = vec![
            250, 5, 5, 255, // near red
            255, 0, 255, 255, // chroma
        ];
        let r = quantize_rgb(
            &rgba,
            2,
            1,
            &QuantizeRgbOptions {
                max_colors: 2,
                fixed_palette: vec![[255, 0, 0], [0, 0, 255]],
                reserve_color: Some([255, 0, 255]),
                ..Default::default()
            },
        );
        assert_eq!(&r.image.data[0..4], &[255, 0, 0, 255]);
        assert_eq!(&r.image.data[4..8], &[255, 0, 255, 255]);
        assert_eq!(r.palette, vec![[255, 0, 0], [0, 0, 255], [255, 0, 255]]);
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
