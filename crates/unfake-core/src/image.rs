//! RGBA buffer helpers.

#[derive(Debug, Clone)]
pub struct RgbaImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl RgbaImage {
    pub fn from_raw(data: Vec<u8>, width: u32, height: u32) -> Option<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return None;
        }
        Some(Self { data, width, height })
    }

    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Option<Self> {
        if x + w > self.width || y + h > self.height {
            return None;
        }
        let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
        for row in 0..h as usize {
            let src = ((y as usize + row) * self.width as usize + x as usize) * 4;
            let dst = row * w as usize * 4;
            out[dst..dst + w as usize * 4]
                .copy_from_slice(&self.data[src..src + w as usize * 4]);
        }
        Some(Self {
            data: out,
            width: w,
            height: h,
        })
    }
}

/// Encode RGBA8 to a PNG byte stream (replaces UPNG.js).
pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width, height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        if let Ok(mut writer) = enc.write_header() {
            let _ = writer.write_image_data(rgba);
        }
    }
    out
}

/// Box (area) resize — averages the source footprint of each target pixel.
/// Matches OpenCV INTER_AREA closely enough for downscaling heuristics.
pub fn resize_area(rgba: &[u8], width: u32, height: u32, tw: u32, th: u32) -> RgbaImage {
    let (wf, hf) = (width as f64, height as f64);
    let tw = tw.max(1);
    let th = th.max(1);
    let mut out = vec![0u8; (tw * th * 4) as usize];
    for ty in 0..th {
        let y0 = ((ty as f64) * hf / th as f64).floor() as u32;
        let y1 = ((((ty + 1) as f64) * hf / th as f64).ceil() as u32).min(height).max(y0 + 1);
        for tx in 0..tw {
            let x0 = ((tx as f64) * wf / tw as f64).floor() as u32;
            let x1 = ((((tx + 1) as f64) * wf / tw as f64).ceil() as u32).min(width).max(x0 + 1);
            let mut acc = [0.0f64; 4];
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = ((sy * width + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += rgba[i + c] as f64;
                    }
                    n += 1;
                }
            }
            let o = ((ty * tw + tx) * 4) as usize;
            for c in 0..4 {
                out[o + c] = (acc[c] / n as f64).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    RgbaImage { data: out, width: tw, height: th }
}
