//! Integration test for `unfake quantize` (binary spawned via CARGO_BIN_EXE).

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

const CHROMA: [u8; 4] = [255, 0, 255, 255];

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("unfake-cli-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// 16x16: chroma border + a many-colored interior.
fn make_input(path: &std::path::Path) {
    let (w, h) = (16u32, 16u32);
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let px = if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                CHROMA
            } else {
                [
                    (x * 16) as u8,
                    (y * 16) as u8,
                    ((x + y) * 8) as u8,
                    255,
                ]
            };
            img.put_pixel(x, y, image::Rgba(px));
        }
    }
    img.save(path).unwrap();
}

#[test]
fn quantize_reserves_chroma_and_limits_colors() {
    let dir = temp_dir("quantize");
    let input = dir.join("in.png");
    let output = dir.join("out.png");
    let manifest = dir.join("manifest.json");
    make_input(&input);

    let status = Command::new(env!("CARGO_BIN_EXE_unfake"))
        .args([
            "quantize",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--colors",
            "8",
            "--reserve-color",
            "#ff00ff",
            "--manifest",
            manifest.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let out = image::open(&output).unwrap().into_rgba8();
    assert_eq!(out.dimensions(), (16, 16));

    // Chroma border survives bit-exact; total colors <= 8.
    let mut colors: HashSet<[u8; 3]> = HashSet::new();
    for (x, y, p) in out.enumerate_pixels() {
        colors.insert([p[0], p[1], p[2]]);
        if x == 0 || y == 0 || x == 15 || y == 15 {
            assert_eq!(p.0, CHROMA, "chroma pixel changed at {x},{y}");
        } else {
            assert_ne!([p[0], p[1], p[2]], [255, 0, 255], "interior became chroma");
        }
    }
    assert!(colors.len() <= 8, "expected <=8 colors, got {}", colors.len());
    assert!(colors.contains(&[255, 0, 255]));

    // Manifest is valid JSON with the quantize block.
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    let q = &m["quantize"];
    assert_eq!(q["colors"], 8);
    assert_eq!(q["reserve_color"], "#ff00ff");
    let pal = q["palette"].as_array().unwrap();
    assert!(pal.len() <= 8 && pal.iter().any(|c| c == "#ff00ff"));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn quantize_fixed_palette_file() {
    let dir = temp_dir("palette-file");
    let input = dir.join("in.png");
    let output = dir.join("out.png");
    let pal_file = dir.join("palette.txt");
    make_input(&input);
    std::fs::write(&pal_file, "// test palette\n#000000\n#ffffff\n").unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_unfake"))
        .args([
            "quantize",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--palette",
            pal_file.to_str().unwrap(),
            "--reserve-color",
            "#ff00ff",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let out = image::open(&output).unwrap().into_rgba8();
    let mut colors: HashSet<[u8; 3]> = HashSet::new();
    for p in out.pixels() {
        colors.insert([p[0], p[1], p[2]]);
    }
    // Black, white, and the reserved chroma only.
    assert!(colors.is_subset(
        &[[0, 0, 0], [255, 255, 255], [255, 0, 255]].into_iter().collect()
    ));
    assert!(colors.contains(&[255, 0, 255]));

    std::fs::remove_dir_all(&dir).ok();
}
