use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use config::Paths;
use engine::Backend;
use std::{path::PathBuf, process::ExitCode, time::Instant};

mod assets;
mod audio;
mod bench;
mod config;
mod engine;
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
    #[arg(long, global = true, help = "Vulkan device index from `hushward gpus`")]
    gpu: Option<usize>,
    #[command(subcommand)]
    command: Command,
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
    /// List Vulkan devices and their indices.
    Gpus,
    /// Run in the foreground; Ctrl-C dismisses the ward.
    Live(LiveArgs),
    /// Check whether the saved live preset keeps up on this computer.
    Check,
    /// Measure inference speed with synthetic audio.
    Bench(bench::Args),
    /// Clean a WAV recording without changing the original.
    Clean {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 5.0)]
        chunk_seconds: f32,
    },
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let paths = Paths::discover()?;
    match &cli.command {
        Command::Doctor => return setup::doctor(&paths),
        Command::Devices { json } => return audio::devices(*json),
        Command::Gpus => return engine::list_gpus(),
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
    if let Some(gpu) = cli.gpu {
        settings.gpu = Some(gpu);
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
        Command::Bench(args) => bench::run(args, model_path, &settings)?,
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
            .context("Read audio")?;
            ensure!(
                !samples.is_empty() && samples.iter().all(|s| s.is_finite()),
                "Input audio is empty or contains invalid samples."
            );
            let (model, _) = engine::load(model_path, &settings)?;
            let start = Instant::now();
            let output_samples = model
                .enhance_chunked(&samples, rate, chunk_seconds, 0.5)
                .context("Enhance audio")?;
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
            let (model, _) = engine::load(model_path, &settings)?;
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
            let (model, _) = engine::load(model_path, &settings)?;
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

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hushward: {error:#}");
            ExitCode::FAILURE
        }
    }
}
