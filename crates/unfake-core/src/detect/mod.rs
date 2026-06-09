//! Scale detectors — parity with `experiments/lib/detectors.mjs` + `edge-detect.mjs`.

mod acpeak;
mod edge;
mod runs;

pub use acpeak::acpeak_detect;
pub use edge::{single_region_edge_detect, tiled_edge_detect};
pub use runs::runs_based_detect;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectMethod {
    Runs,
    Acpeak,
    Tiled,
    EdgeLegacy,
    AutoTiled,
}

impl DetectMethod {
    pub fn detect(self, rgba: &[u8], width: u32, height: u32) -> u32 {
        match self {
            Self::Runs => runs_based_detect(rgba, width, height),
            Self::Acpeak => acpeak_detect(rgba, width, height),
            Self::Tiled => tiled_edge_detect(rgba, width, height),
            Self::EdgeLegacy => single_region_edge_detect(rgba, width, height),
            Self::AutoTiled => auto_tiled_detect(rgba, width, height),
        }
    }
}

pub fn auto_tiled_detect(rgba: &[u8], width: u32, height: u32) -> u32 {
    let runs = runs_based_detect(rgba, width, height);
    if runs > 1 {
        runs
    } else {
        tiled_edge_detect(rgba, width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_on_uniform_grid() {
        let w = 64u32;
        let h = 64u32;
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let c = if (x / 4 + y / 4) % 2 == 0 { 200 } else { 50 };
                data[i] = c;
                data[i + 1] = c;
                data[i + 2] = c;
                data[i + 3] = 255;
            }
        }
        assert_eq!(tiled_edge_detect(&data, w, h), 4);
    }
}
