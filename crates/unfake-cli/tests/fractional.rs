//! Integration test for fractional grids: `unfake detect` → `unfake downscale -s <float>`.

use std::path::PathBuf;
use std::process::Command;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("unfake-cli-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn color(cx: u32, cy: u32) -> [u8; 4] {
    let h = (cx.wrapping_mul(73_856_093) ^ cy.wrapping_mul(19_349_663)) % 7;
    [(h * 36) as u8, (250 - h * 30) as u8, (h * 91 % 256) as u8, 255]
}

/// 200×200-cell sprite resampled to 500 px (2.5 px/cell, cells 2,3,2,3…).
fn make_input(path: &std::path::Path) {
    let (cells, size) = (200u32, 500u32);
    let p = size as f64 / cells as f64;
    let img = image::RgbaImage::from_fn(size, size, |x, y| {
        let cx = (((x as f64 + 0.5) / p) as u32).min(cells - 1);
        let cy = (((y as f64 + 0.5) / p) as u32).min(cells - 1);
        image::Rgba(color(cx, cy))
    });
    img.save(path).unwrap();
}

fn unfake(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_unfake")).args(args).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

#[test]
fn detect_and_downscale_fractional_grid() {
    let dir = temp_dir("fractional");
    let input = dir.join("in.png");
    let output = dir.join("out.png");
    let manifest = dir.join("manifest.json");
    make_input(&input);
    let input_s = input.to_str().unwrap();

    let scale = unfake(&["detect", input_s]);
    assert_eq!(scale, "2.5");
    let integer = unfake(&["detect", input_s, "--integer"]);
    assert!(integer.parse::<u32>().is_ok(), "--integer printed {integer:?}");

    unfake(&[
        "downscale",
        input_s,
        "-s",
        &scale,
        "-m",
        "nearest",
        "-o",
        output.to_str().unwrap(),
        "--manifest",
        manifest.to_str().unwrap(),
    ]);
    let out = image::open(&output).unwrap().into_rgba8();
    assert_eq!(out.dimensions(), (200, 200));
    for (x, y, px) in out.enumerate_pixels() {
        assert_eq!(px.0, color(x, y), "cell {x},{y}");
    }

    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    assert_eq!(m["downscale"]["grid"]["cells"], serde_json::json!([200, 200]));

    let _ = std::fs::remove_dir_all(&dir);
}
