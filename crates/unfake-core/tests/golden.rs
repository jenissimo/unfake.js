//! Bit-exact golden tests (synthetic checkerboards — no external PNGs).

use unfake_core::{
    auto_tiled_detect, downscale, downscale_grid, find_optimal_crop_offset,
    fractional_grid_detect, runs_based_detect, single_region_edge_detect, snap_to_grid,
    tiled_edge_detect, DownscaleMethod, DownscaleOptions,
};

fn checker_rgba(w: u32, h: u32, block: u32) -> Vec<u8> {
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let c = if ((x / block) + (y / block)) % 2 == 0 { 220u8 } else { 30u8 };
            data[i] = c;
            data[i + 1] = c;
            data[i + 2] = c;
            data[i + 3] = 255;
        }
    }
    data
}

#[test]
fn golden_checker_4x_detect_and_snap() {
    let w = 64u32;
    let h = 64u32;
    let block = 4u32;
    let rgba = checker_rgba(w, h, block);

    assert_eq!(tiled_edge_detect(&rgba, w, h), block);
    assert_eq!(single_region_edge_detect(&rgba, w, h), block);
    assert_eq!(runs_based_detect(&rgba, w, h), block);
    assert_eq!(auto_tiled_detect(&rgba, w, h), block);

    let off = find_optimal_crop_offset(&rgba, w, h, block);
    assert_eq!(off.x, 0);
    assert_eq!(off.y, 0);

    let snapped = snap_to_grid(&rgba, w, h, block);
    assert_eq!(snapped.width, w);
    assert_eq!(snapped.height, h);
    assert_eq!(snapped.data, rgba);
}

#[test]
fn golden_checker_3x_detect() {
    let w = 48u32;
    let h = 48u32;
    let block = 3u32;
    let rgba = checker_rgba(w, h, block);
    assert_eq!(tiled_edge_detect(&rgba, w, h), block);
    assert_eq!(auto_tiled_detect(&rgba, w, h), block);
}

#[test]
fn golden_downscale_nearest_bit_exact() {
    let w = 64u32;
    let h = 64u32;
    let block = 4u32;
    let rgba = checker_rgba(w, h, block);
    let out = downscale(
        &rgba,
        w,
        h,
        block,
        &DownscaleOptions {
            method: DownscaleMethod::Nearest,
            dom_threshold: 0.15,
            align_grid: false,
        },
    );
    assert_eq!(out.width, 16);
    assert_eq!(out.height, 16);
    // Every cell samples the center of a uniform block → same checker at 16×16
    for ty in 0..16u32 {
        for tx in 0..16u32 {
            let i = ((ty * 16 + tx) * 4) as usize;
            let c = if (tx + ty) % 2 == 0 { 220 } else { 30 };
            assert_eq!(out.data[i], c, "pixel {tx},{ty}");
            assert_eq!(out.data[i + 3], 255);
        }
    }
}

#[test]
fn golden_tiled_matches_manifest() {
    // Rust side of the shared contract in experiments/golden/manifest.json.
    // Cross-language parity (JS == Rust) is enforced by scripts/verify-golden.mjs.
    for (w, h, block) in [(64u32, 64, 4), (48, 48, 3)] {
        let rgba = checker_rgba(w, h, block);
        assert_eq!(tiled_edge_detect(&rgba, w, h), block);
    }
}

/// `cells`×`cells` sprite (hashed colors) and its nearest resample to `size` px
/// — cell boundaries at multiples of size/cells, like a model's canvas resize.
fn fractional_sprite(cells: u32, size: u32) -> (Vec<u8>, Vec<u8>) {
    let color = |cx: u32, cy: u32| -> [u8; 4] {
        let h = (cx.wrapping_mul(73_856_093) ^ cy.wrapping_mul(19_349_663)) % 7;
        [(h * 36) as u8, (250 - h * 30) as u8, (h * 91 % 256) as u8, 255]
    };
    let mut src = Vec::with_capacity((cells * cells * 4) as usize);
    for cy in 0..cells {
        for cx in 0..cells {
            src.extend_from_slice(&color(cx, cy));
        }
    }
    let p = size as f64 / cells as f64;
    let mut up = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let cx = (((x as f64 + 0.5) / p) as u32).min(cells - 1);
            let cy = (((y as f64 + 0.5) / p) as u32).min(cells - 1);
            up.extend_from_slice(&color(cx, cy));
        }
    }
    (src, up)
}

#[test]
fn golden_fractional_grid_roundtrip() {
    // 2048/200 = 10.24 and 500/200 = 2.5 (the reported model outputs), 1024/82.
    for (cells, size) in [(200u32, 2048u32), (200, 500), (82, 1024)] {
        let (src, up) = fractional_sprite(cells, size);
        let grid = fractional_grid_detect(&up, size, size).expect("fractional grid");
        assert_eq!(grid.cells(size, size), (cells, cells), "{size}/{cells}");
        for method in [
            DownscaleMethod::Nearest,
            DownscaleMethod::Median,
            DownscaleMethod::Mode,
            DownscaleMethod::Dominant,
        ] {
            let out = downscale_grid(
                &up,
                size,
                size,
                &grid,
                &DownscaleOptions {
                    method,
                    dom_threshold: 0.15,
                    align_grid: false,
                },
            );
            assert_eq!((out.width, out.height), (cells, cells));
            assert!(out.data == src, "{size}/{cells} {method:?}: pixels differ");
        }
    }
}

#[test]
fn golden_integer_grid_not_fractional() {
    // Integer upscales stay on the integer path (bit-exact snap + block downscale).
    for (w, h, block) in [(64u32, 64, 4), (48, 48, 3)] {
        assert_eq!(fractional_grid_detect(&checker_rgba(w, h, block), w, h), None);
    }
    let (_, up) = fractional_sprite(64, 512);
    assert_eq!(fractional_grid_detect(&up, 512, 512), None);
}
