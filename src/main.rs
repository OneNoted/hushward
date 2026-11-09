use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use config::{Paths, Settings};
use reuse_rs::{GpuDevice, Offline};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

mod assets;
mod audio;
mod config;
mod live;
mod setup;

#[derive(Parser)]
#[command(
    version,
    about = "A quieter microphone, guarded by RE-USE.",
    after_help = "First visit? Run `hushward setup`, then `hushward start`."
)]
struct Cli {
    #[arg(long, global = true, help = "Override the model file")]
    model: Option<PathBuf>,
    #[arg(long, global = true, value_enum)]
    backend: Option<Backend>,
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Cpu,
    #[default]
    Vulkan,
}

#[derive(Args, Default)]
struct LiveArgs {
    #[arg(long, help = "Microphone name from `hushward devices`")]
    source: Option<String>,
    #[arg(long, help = "Audio window in seconds")]
    window: Option<f32>,
    #[arg(long, help = "8000 for live speech; 16000 for wider bandwidth")]
    rate: Option<usize>,
    #[arg(long, help = "Crossfade overlap in milliseconds")]
    overlap_ms: Option<u32>,
    #[arg(long, help = "Stop after this many seconds of output")]
    seconds: Option<f32>,
    #[arg(
        long,
        help = "Switch applications to Hushward and restore them on exit"
    )]
    make_default: bool,
    #[arg(long, help = "Print timing for every window")]
    verbose: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Choose a microphone, download the model and install user commands.
    Setup {
        #[arg(long)]
        source: Option<String>,
        #[arg(
            long,
            help = "Use --source or the current microphone without prompting"
        )]
        yes: bool,
    },
    /// Check the audio tools, server and model installation.
    Doctor,
    /// Start the background microphone service.
    Start,
    /// Stop processing and restore the previous microphone.
    Stop,
    /// Show whether the ward is running.
    Status,
    /// Remove the user command and service, keeping settings and model files.
    Uninstall,
    /// Download and verify the original model weights.
    Fetch,
    /// List microphones and their names for --source.
    Devices {
        #[arg(long)]
        json: bool,
    },
    /// Run in the foreground; Ctrl-C dismisses the ward.
    Live(LiveArgs),
    /// Check whether the saved live preset keeps up on this computer.
    Check,
    /// Measure inference speed with synthetic audio.
    Bench {
        #[arg(long, default_value_t = 0.16)]
        seconds: f32,
        #[arg(long, default_value_t = 5)]
        runs: usize,
        #[arg(long, default_value_t = 8000)]
        rate: u32,
    },
    /// Clean a WAV recording without changing the original.
    Clean {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 5.0)]
        chunk_seconds: f32,
    },
}

fn load(path: &Path, backend: Backend) -> Result<Offline> {
    ensure!(
        path.is_file(),
        "Model not found at {}. Run `hushward setup` or `hushward fetch`.",
        path.display()
    );
    assets::verify(path)?;
    eprintln!("Loading RE-USE on {backend:?}…");
    let model = match backend {
        Backend::Cpu => Offline::load(path),
        Backend::Vulkan => Offline::load_gpu(path, GpuDevice::Vulkan),
    };
    model.map_err(|e| anyhow::anyhow!("Cannot load the {backend:?} model: {e:#}"))
}

#[derive(Serialize)]
struct Measurement {
    backend: Backend,
    threads: usize,
    sample_rate: u32,
    audio_seconds: f64,
    inference_seconds: f64,
    realtime_factor: f64,
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let paths = Paths::discover()?;
    match &cli.command {
        Command::Doctor => return setup::doctor(&paths),
        Command::Devices { json } => return audio::devices(*json),
        Command::Start => return setup::control("start", &paths),
        Command::Stop => return setup::control("stop", &paths),
        Command::Status => return setup::status(&paths),
        Command::Uninstall => return setup::uninstall(&paths),
        _ => {}
    }
    let mut settings = paths.load()?;
    if let Some(backend) = cli.backend {
        settings.backend = backend;
    }
    if let Some(threads) = cli.threads {
        settings.threads = threads;
    }
    if let Some(model) = cli.model {
        settings.model = Some(std::path::absolute(model)?);
    }
    if let Command::Live(args) = &cli.command {
        if let Some(source) = &args.source {
            settings.source = Some(source.clone());
        }
        if let Some(rate) = args.rate {
            settings.rate = rate;
        }
        if let Some(window) = args.window {
            settings.window = window;
        }
        if let Some(overlap) = args.overlap_ms {
            settings.overlap_ms = overlap;
        }
    }
    settings.validate()?;
    let model_path = settings.model.as_deref().unwrap_or(&paths.model);
    match cli.command {
        Command::Setup { source, yes } => return setup::run(&paths, settings, source, yes),
        Command::Fetch => return assets::fetch(model_path),
        Command::Bench {
            seconds,
            runs,
            rate,
        } => {
            ensure!(
                seconds.is_finite()
                    && (0.1..=30.0).contains(&seconds)
                    && (1..=1000).contains(&runs)
                    && reuse_rs::audio::SAMPLE_RATES.contains(&rate),
                "Use 0.1–30 seconds, 1–1000 runs, and a supported sample rate."
            );
            let model = initialize(model_path, &settings)?;
            let samples: Vec<f32> = (0..(seconds * rate as f32) as usize)
                .map(|i| {
                    let t = i as f32 / rate as f32;
                    0.05 * (std::f32::consts::TAU * 180.0 * t).sin()
                        + 0.01 * (std::f32::consts::TAU * 1739.0 * t).sin()
                })
                .collect();
            eprintln!("Warming up…");
            enhance(&model, &samples, rate)?;
            for _ in 0..runs {
                let start = Instant::now();
                enhance(&model, &samples, rate)?;
                let elapsed = start.elapsed().as_secs_f64();
                println!(
                    "{}",
                    serde_json::to_string(&Measurement {
                        backend: settings.backend,
                        threads: settings.threads,
                        sample_rate: rate,
                        audio_seconds: seconds as f64,
                        inference_seconds: elapsed,
                        realtime_factor: elapsed / seconds as f64
                    })?
                );
            }
        }
        Command::Clean {
            input,
            output,
            chunk_seconds,
        } => {
            ensure!(
                chunk_seconds.is_finite() && (0.1..=30.0).contains(&chunk_seconds),
                "Chunk length must be between 0.1 and 30 seconds."
            );
            ensure!(
                !output.exists(),
                "Output already exists: {}",
                output.display()
            );
            let (samples, rate) = reuse_rs::audio::load_wav(
                input
                    .to_str()
                    .context("The input path is not valid UTF-8")?,
            )
            .map_err(|e| anyhow::anyhow!("Read audio: {e:#}"))?;
            ensure!(
                !samples.is_empty() && samples.iter().all(|s| s.is_finite()),
                "Input audio is empty or contains invalid samples."
            );
            let model = initialize(model_path, &settings)?;
            let start = Instant::now();
            let output_samples = model
                .enhance_chunked(&samples, rate, chunk_seconds, 0.5)
                .map_err(|e| anyhow::anyhow!("Enhance audio: {e:#}"))?;
            assets::write_wav(&output, &output_samples, rate)?;
            println!(
                "Cleaned {:.1}s of audio in {:.1}s → {}",
                samples.len() as f64 / rate as f64,
                start.elapsed().as_secs_f64(),
                output.display()
            );
        }
        Command::Live(args) => {
            let source = audio::resolve_source(settings.source.as_deref())?;
            let model = initialize(model_path, &settings)?;
            live::run(
                &model,
                &source,
                &settings,
                args.seconds,
                args.make_default,
                args.verbose,
            )?;
        }
        Command::Check => {
            let model = initialize(model_path, &settings)?;
            let elapsed = live::preflight(&model, &settings)?;
            println!(
                "Ready for live speech: {:.0} ms inference, {:.0} ms available per window.",
                elapsed * 1000.0,
                (settings.window - settings.overlap_ms as f32 / 1000.0) * 1000.0
            );
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn initialize(path: &Path, settings: &Settings) -> Result<Offline> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(settings.threads)
        .build_global()?;
    load(path, settings.backend)
}

fn enhance(model: &Offline, samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    let output = model
        .enhance(samples, rate)
        .map_err(|e| anyhow::anyhow!("Inference: {e:#}"))?;
    if output.len() != samples.len() || !output.iter().all(|v| v.is_finite()) {
        bail!("The model returned invalid audio.");
    }
    Ok(output)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hushward: {error:#}");
            ExitCode::FAILURE
        }
    }
}
