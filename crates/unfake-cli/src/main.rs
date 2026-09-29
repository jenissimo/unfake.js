use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use image::ImageEncoder;
use serde_json::{json, Value};
use unfake_core::{
    downscale, downscale_bounds, downscale_grid, elastic_bounds, find_optimal_crop_offset,
    fractional_grid_detect, grid_for_scale, quantize_rgb, snap_to_grid, DetectMethod,
    DownscaleMethod, DownscaleOptions, Grid, QuantizeRgbOptions, RgbaImage,
};

#[derive(Parser)]
#[command(name = "unfake", about = "Pixel-art CV core CLI (detect / snap / downscale / quantize)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect pixel grid scale (fractional, e.g. 10.24, for resampled grids)
    Detect {
        input: PathBuf,
        #[arg(short, long, value_enum, default_value_t = DetectArg::Tiled)]
        method: DetectArg,
        /// Report only the integer scale (skip fractional grid detection)
        #[arg(long)]
        integer: bool,
        /// Write/update a JSON manifest with the detected scale
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// Snap image to grid (crop to aligned bounds)
    Snap {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(short, long)]
        scale: u32,
        /// Write/update a JSON manifest with crop offset and dimensions
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// Downscale by block factor
    Downscale {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Cell size in px; non-integer (e.g. 10.24) samples a fractional grid
        #[arg(short, long)]
        scale: f64,
        #[arg(short, long, value_enum, default_value_t = DsArg::Dominant)]
        method: DsArg,
        #[arg(long, default_value_t = 0.15)]
        dom_threshold: f64,
        #[arg(long)]
        no_align: bool,
        /// Keep a fixed grid even where hand-drawn cells jitter around the scale
        #[arg(long)]
        no_elastic: bool,
        /// Write/update a JSON manifest with output dimensions
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// Quantize palette (RGB only; alpha is copied through untouched)
    Quantize {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Max palette size, including the reserved color when set
        #[arg(short, long, default_value_t = 16)]
        colors: u32,
        /// Fixed palette: comma-separated hex colors ("#112233,#445566")
        /// or a path to a file with one hex color per line
        #[arg(long)]
        palette: Option<String>,
        /// Reserved chroma color ("#RRGGBB"): matching pixels skip palette
        /// selection and stay bit-exact; the color gets its own palette entry
        #[arg(long)]
        reserve_color: Option<String>,
        /// Per-channel tolerance for matching --reserve-color (0 = exact)
        #[arg(long, default_value_t = 0)]
        reserve_tolerance: u8,
        /// Write/update a JSON manifest with the final palette and parameters
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// Evaluate detectors on bghira corpus (__psN in filename)
    EvalBghira {
        #[arg(default_value = "experiments/data/bghira")]
        dir: PathBuf,
        #[arg(short, long, value_enum, default_value_t = DetectArg::Tiled)]
        method: DetectArg,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum DetectArg {
    Runs,
    Acpeak,
    Tiled,
    Auto,
}

#[derive(Clone, Copy, ValueEnum)]
enum DsArg {
    Nearest,
    Median,
    Mode,
    Dominant,
    Qvote,
}

impl From<DetectArg> for DetectMethod {
    fn from(v: DetectArg) -> Self {
        match v {
            DetectArg::Runs => Self::Runs,
            DetectArg::Acpeak => Self::Acpeak,
            DetectArg::Tiled => Self::Tiled,
            DetectArg::Auto => Self::AutoTiled,
        }
    }
}

impl From<DsArg> for DownscaleMethod {
    fn from(v: DsArg) -> Self {
        match v {
            DsArg::Nearest => Self::Nearest,
            DsArg::Median => Self::Median,
            DsArg::Mode => Self::Mode,
            DsArg::Dominant => Self::Dominant,
            DsArg::Qvote => Self::Qvote,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Detect {
            input,
            method,
            integer,
            manifest,
        } => detect_cmd(&input, method.into(), integer, manifest.as_deref()),
        Commands::Snap {
            input,
            output,
            scale,
            manifest,
        } => snap_cmd(&input, output.as_deref(), scale, manifest.as_deref()),
        Commands::Downscale {
            input,
            output,
            scale,
            method,
            dom_threshold,
            no_align,
            no_elastic,
            manifest,
        } => downscale_cmd(
            &input,
            output.as_deref(),
            scale,
            method.into(),
            dom_threshold,
            !no_align,
            !no_elastic,
            manifest.as_deref(),
        ),
        Commands::Quantize {
            input,
            output,
            colors,
            palette,
            reserve_color,
            reserve_tolerance,
            manifest,
        } => quantize_cmd(
            &input,
            output.as_deref(),
            colors,
            palette.as_deref(),
            reserve_color.as_deref(),
            reserve_tolerance,
            manifest.as_deref(),
        ),
        Commands::EvalBghira { dir, method } => eval_bghira_cmd(&dir, method.into()),
    }
}

fn load_rgba(path: &Path) -> Result<RgbaImage, Box<dyn std::error::Error>> {
    let img = image::open(path)?.into_rgba8();
    let (w, h) = img.dimensions();
    Ok(RgbaImage {
        data: img.into_raw(),
        width: w,
        height: h,
    })
}

fn save_rgba(img: &RgbaImage, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let encoder = image::codecs::png::PngEncoder::new(std::fs::File::create(path)?);
    encoder.write_image(&img.data, img.width, img.height, image::ExtendedColorType::Rgba8)?;
    Ok(())
}

fn detect_cmd(
    input: &Path,
    method: DetectMethod,
    integer_only: bool,
    manifest: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let detect = |img: &RgbaImage| -> (u32, Option<Grid>) {
        let scale = method.detect(&img.data, img.width, img.height);
        let grid = if integer_only {
            None
        } else {
            fractional_grid_detect(&img.data, img.width, img.height)
        };
        (scale, grid)
    };
    if input.is_dir() {
        if manifest.is_some() {
            eprintln!("--manifest is ignored for directory input");
        }
        for entry in std::fs::read_dir(input)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("png") {
                let img = load_rgba(&path)?;
                match detect(&img) {
                    (_, Some(g)) => {
                        let (cw, ch) = g.cells(img.width, img.height);
                        println!(
                            "{}  scale={} ({cw}x{ch} cells)",
                            path.display(),
                            fmt_scale(g.scale())
                        );
                    }
                    (scale, None) => println!("{}  scale={}", path.display(), scale),
                }
            }
        }
    } else {
        let img = load_rgba(input)?;
        let (scale, grid) = detect(&img);
        let mut block = json!({
            "input": input.display().to_string(),
            "method": format!("{method:?}").to_lowercase(),
            "scale": scale,
            "width": img.width,
            "height": img.height,
        });
        if let Some(g) = grid {
            println!("{}", fmt_scale(g.scale()));
            block["scale"] = json!(g.scale());
            block["integer_scale"] = json!(scale);
            block["grid"] = grid_json(&g, img.width, img.height);
        } else {
            println!("{scale}");
        }
        if let Some(m) = manifest {
            update_manifest(m, "detect", block)?;
        }
    }
    Ok(())
}

/// Up to 4 decimals, trailing zeros trimmed ("10.24", "6.7308").
fn fmt_scale(v: f64) -> String {
    let s = format!("{v:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn grid_json(g: &Grid, width: u32, height: u32) -> Value {
    let (cw, ch) = g.cells(width, height);
    json!({
        "period": [g.x.period, g.y.period],
        "offset": [g.x.offset, g.y.offset],
        "coherence": [g.x.coherence, g.y.coherence],
        "cells": [cw, ch],
    })
}

fn snap_cmd(
    input: &Path,
    output: Option<&Path>,
    scale: u32,
    manifest: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let src = load_rgba(input)?;
    let crop = find_optimal_crop_offset(&src.data, src.width, src.height, scale);
    let out = snap_to_grid(&src.data, src.width, src.height, scale);
    let dest = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| input.with_extension("snapped.png"));
    save_rgba(&out, &dest)?;
    eprintln!(
        "{} -> {} ({}x{})",
        input.display(),
        dest.display(),
        out.width,
        out.height
    );
    if let Some(m) = manifest {
        update_manifest(
            m,
            "snap",
            json!({
                "input": input.display().to_string(),
                "output": dest.display().to_string(),
                "scale": scale,
                "crop": { "x": crop.x, "y": crop.y },
                "width": out.width,
                "height": out.height,
            }),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn downscale_cmd(
    input: &Path,
    output: Option<&Path>,
    scale: f64,
    method: DownscaleMethod,
    dom_threshold: f64,
    align_grid: bool,
    elastic: bool,
    manifest: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if scale.is_nan() || scale < 1.0 {
        return Err(format!("--scale must be >= 1 (got {scale})").into());
    }
    let src = load_rgba(input)?;
    let opts = DownscaleOptions {
        method,
        dom_threshold,
        align_grid,
    };
    let grid =
        (scale.fract() != 0.0).then(|| grid_for_scale(&src.data, src.width, src.height, scale));
    // Hand-drawn cells jitter around the scale: fit the boundaries to the edges
    // on the axes where a fixed grid drifts.
    let fitted = if elastic && scale >= 2.0 {
        let base = grid.unwrap_or_else(|| {
            let s = scale as u32;
            let off = if align_grid {
                find_optimal_crop_offset(&src.data, src.width, src.height, s)
            } else {
                Default::default()
            };
            Grid::integer(s, off.x, off.y)
        });
        elastic_bounds(&src.data, src.width, src.height, &base)
    } else {
        None
    };
    let out = match (&fitted, &grid) {
        (Some((bounds, _)), _) => downscale_bounds(&src.data, src.width, src.height, bounds, &opts),
        (None, Some(g)) => downscale_grid(&src.data, src.width, src.height, g, &opts),
        (None, None) => downscale(&src.data, src.width, src.height, scale as u32, &opts),
    };
    let dest = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| input.with_extension("down.png"));
    save_rgba(&out, &dest)?;
    eprintln!(
        "{} -> {} ({}x{})",
        input.display(),
        dest.display(),
        out.width,
        out.height
    );
    if let Some(m) = manifest {
        let mut block = json!({
            "input": input.display().to_string(),
            "output": dest.display().to_string(),
            "scale": scale,
            "method": format!("{method:?}").to_lowercase(),
            "dom_threshold": dom_threshold,
            "align_grid": align_grid,
            "width": out.width,
            "height": out.height,
        });
        if let Some(g) = &grid {
            block["grid"] = grid_json(g, src.width, src.height);
        }
        if let Some((_, adapted)) = &fitted {
            block["elastic"] = json!(adapted);
        }
        update_manifest(m, "downscale", block)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn quantize_cmd(
    input: &Path,
    output: Option<&Path>,
    colors: u32,
    palette: Option<&str>,
    reserve_color: Option<&str>,
    reserve_tolerance: u8,
    manifest: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let src = load_rgba(input)?;
    let fixed_palette = match palette {
        Some(spec) => parse_palette_arg(spec)?,
        None => Vec::new(),
    };
    let reserve = reserve_color.map(parse_hex_color).transpose()?;

    let opts = QuantizeRgbOptions {
        max_colors: colors,
        fixed_palette,
        reserve_color: reserve,
        reserve_tolerance,
    };
    let res = quantize_rgb(&src.data, src.width, src.height, &opts);
    let dest = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| input.with_extension("quant.png"));
    save_rgba(&res.image, &dest)?;
    let palette_hex: Vec<String> = res.palette.iter().map(|c| to_hex(*c)).collect();
    eprintln!(
        "{} -> {} ({}x{}, {} colors)",
        input.display(),
        dest.display(),
        res.image.width,
        res.image.height,
        palette_hex.len()
    );
    if let Some(m) = manifest {
        update_manifest(
            m,
            "quantize",
            json!({
                "input": input.display().to_string(),
                "output": dest.display().to_string(),
                "colors": colors,
                "palette": palette_hex,
                "reserve_color": reserve.map(to_hex),
                "reserve_tolerance": reserve_tolerance,
                "width": res.image.width,
                "height": res.image.height,
            }),
        )?;
    }
    Ok(())
}

fn parse_hex_color(s: &str) -> Result<[u8; 3], Box<dyn std::error::Error>> {
    let hex = s.trim().trim_start_matches('#');
    if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("invalid hex color: {s:?} (expected #RRGGBB)").into());
    }
    Ok([
        u8::from_str_radix(&hex[0..2], 16)?,
        u8::from_str_radix(&hex[2..4], 16)?,
        u8::from_str_radix(&hex[4..6], 16)?,
    ])
}

fn to_hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// `--palette` accepts a comma-separated hex list or a path to a file with
/// one hex color per line (blank lines and `//` comments ignored).
fn parse_palette_arg(spec: &str) -> Result<Vec<[u8; 3]>, Box<dyn std::error::Error>> {
    let text;
    let entries: Vec<&str> = if Path::new(spec).is_file() {
        text = std::fs::read_to_string(spec)?;
        text.lines().collect()
    } else {
        spec.split(',').collect()
    };
    let mut palette = Vec::new();
    for e in entries {
        let e = e.trim();
        if e.is_empty() || e.starts_with("//") {
            continue;
        }
        palette.push(parse_hex_color(e)?);
    }
    if palette.is_empty() {
        return Err(format!("--palette {spec:?} contains no colors").into());
    }
    Ok(palette)
}

/// Read-modify-write: merge `block` under `key` into an existing JSON
/// manifest, or start a fresh one if the file is missing or invalid.
fn update_manifest(
    path: &Path,
    key: &str,
    block: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut root = match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => Value::Object(map),
            Ok(_) | Err(_) => {
                eprintln!("manifest {} is not a JSON object; rewriting", path.display());
                json!({})
            }
        },
        Err(_) => json!({}),
    };
    root[key] = block;
    std::fs::write(path, serde_json::to_string_pretty(&root)? + "\n")?;
    eprintln!("manifest: {} [{key}]", path.display());
    Ok(())
}

fn parse_bghira_gt(name: &str) -> Option<u32> {
    let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
    let idx = stem.rfind("__ps")?;
    stem[idx + 4..].parse().ok()
}

fn eval_bghira_cmd(dir: &Path, method: DetectMethod) -> Result<(), Box<dyn std::error::Error>> {
    if !dir.is_dir() {
        return Err(format!("bghira dir not found: {}", dir.display()).into());
    }
    let mut total = 0u32;
    let mut exact = 0u32;
    let mut pm1 = 0u32;

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("png"))
        .collect();
    files.sort();

    for path in &files {
        let gt = match path.file_name().and_then(|n| n.to_str()).and_then(parse_bghira_gt) {
            Some(g) => g,
            None => continue,
        };
        let img = load_rgba(path)?;
        let pred = method.detect(&img.data, img.width, img.height);
        total += 1;
        if pred == gt {
            exact += 1;
        }
        if pred.abs_diff(gt) <= 1 {
            pm1 += 1;
        }
    }

    if total == 0 {
        return Err("no bghira PNGs with __psN GT found".into());
    }

    println!(
        "method={:?}  n={}  exact={:.1}%  ±1={:.1}%",
        method,
        total,
        100.0 * exact as f64 / total as f64,
        100.0 * pm1 as f64 / total as f64,
    );
    Ok(())
}
