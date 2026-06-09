use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use image::ImageEncoder;
use unfake_core::{
    downscale, snap_to_grid, DetectMethod, DownscaleMethod, DownscaleOptions, RgbaImage,
};

#[derive(Parser)]
#[command(name = "unfake", about = "Pixel-art CV core CLI (detect / snap / downscale)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect pixel grid scale
    Detect {
        input: PathBuf,
        #[arg(short, long, value_enum, default_value_t = DetectArg::Tiled)]
        method: DetectArg,
    },
    /// Snap image to grid (crop to aligned bounds)
    Snap {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(short, long)]
        scale: u32,
    },
    /// Downscale by block factor
    Downscale {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(short, long)]
        scale: u32,
        #[arg(short, long, value_enum, default_value_t = DsArg::Dominant)]
        method: DsArg,
        #[arg(long, default_value_t = 0.15)]
        dom_threshold: f64,
        #[arg(long)]
        no_align: bool,
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
        Commands::Detect { input, method } => detect_cmd(&input, method.into()),
        Commands::Snap {
            input,
            output,
            scale,
        } => snap_cmd(&input, output.as_deref(), scale),
        Commands::Downscale {
            input,
            output,
            scale,
            method,
            dom_threshold,
            no_align,
        } => downscale_cmd(
            &input,
            output.as_deref(),
            scale,
            method.into(),
            dom_threshold,
            !no_align,
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

fn detect_cmd(input: &Path, method: DetectMethod) -> Result<(), Box<dyn std::error::Error>> {
    if input.is_dir() {
        for entry in std::fs::read_dir(input)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("png") {
                let img = load_rgba(&path)?;
                let scale = method.detect(&img.data, img.width, img.height);
                println!("{}  scale={}", path.display(), scale);
            }
        }
    } else {
        let img = load_rgba(input)?;
        println!("{}", method.detect(&img.data, img.width, img.height));
    }
    Ok(())
}

fn snap_cmd(
    input: &Path,
    output: Option<&Path>,
    scale: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let src = load_rgba(input)?;
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
    Ok(())
}

fn downscale_cmd(
    input: &Path,
    output: Option<&Path>,
    scale: u32,
    method: DownscaleMethod,
    dom_threshold: f64,
    align_grid: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let src = load_rgba(input)?;
    let out = downscale(
        &src.data,
        src.width,
        src.height,
        scale,
        &DownscaleOptions {
            method,
            dom_threshold,
            align_grid,
        },
    );
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
