//! Canonical pixel-art CV core (ported from `experiments/lib/`).

pub mod color_count;
pub mod content_adaptive;
pub mod detect;
pub mod downscale;
pub mod image;
pub mod math;
pub mod morphology;
pub mod quantize;
pub mod snap;
pub mod vector;

pub use color_count::{detect_optimal_color_count, ColorCountOpts};
pub use content_adaptive::content_adaptive_downscale;
pub use detect::{
    acpeak_detect, auto_tiled_detect, elastic_bounds, fractional_grid_detect, grid_for_scale,
    runs_based_detect, single_region_edge_detect, tiled_edge_detect, AxisGrid, CellBounds,
    DetectMethod, Grid,
};
pub use downscale::{downscale, downscale_bounds, downscale_grid, DownscaleMethod, DownscaleOptions};
pub use image::{encode_png, resize_area, RgbaImage};
pub use morphology::morph_open_close;
pub use quantize::{quantize, quantize_rgb, QuantizeResult, QuantizeRgbOptions, QuantizeRgbResult};
pub use snap::{find_optimal_crop_offset, snap_to_grid, CropOffset};
pub use vector::{vector_fill_background, vector_postprocess, vector_preprocess};
