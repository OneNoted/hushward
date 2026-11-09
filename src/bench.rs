use crate::{config::Settings, engine};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::{path::Path, time::Instant};

#[derive(clap::Args)]
pub struct Args {
    #[arg(long, default_value_t = 0.16)]
    seconds: f32,
    #[arg(long, default_value_t = 20)]
    runs: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long, default_value_t = 8000)]
    rate: u32,
}

#[derive(Serialize)]
struct Measurement {
    backend: engine::Backend,
    threads: usize,
    sample_rate: u32,
    audio_seconds: f64,
    inference_seconds: f64,
    realtime_factor: f64,
}

pub fn run(args: Args, path: &Path, settings: &Settings) -> Result<()> {
    ensure!(
        args.seconds.is_finite()
            && (0.1..=30.0).contains(&args.seconds)
            && (1..=1000).contains(&args.runs)
            && (1..=100).contains(&args.warmup)
            && reuse_rs::audio::SAMPLE_RATES.contains(&args.rate),
        "Use 0.1–30 seconds, 1–1000 runs, 1–100 warmup runs, and a supported sample rate."
    );
    let (model, backend) = engine::load(path, settings)?;
    let samples: Vec<f32> = (0..(args.seconds * args.rate as f32).round() as usize)
        .map(|i| {
            let t = i as f32 / args.rate as f32;
            0.05 * (std::f32::consts::TAU * 180.0 * t).sin()
                + 0.01 * (std::f32::consts::TAU * 1739.0 * t).sin()
        })
        .collect();
    let audio_seconds = samples.len() as f64 / args.rate as f64;
    eprintln!("Warming up ({} windows)…", args.warmup);
    for _ in 0..args.warmup {
        engine::enhance(&model, &samples, args.rate)?;
    }
    let mut timings = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        engine::enhance(&model, &samples, args.rate)?;
        let elapsed = start.elapsed().as_secs_f64();
        timings.push(elapsed);
        println!(
            "{}",
            serde_json::to_string(&Measurement {
                backend,
                threads: settings.threads,
                sample_rate: args.rate,
                audio_seconds,
                inference_seconds: elapsed,
                realtime_factor: elapsed / audio_seconds,
            })?
        );
    }
    timings.sort_by(f64::total_cmp);
    let percentile = |p: f64| timings[((args.runs as f64 * p).ceil() as usize).saturating_sub(1)];
    eprintln!(
        "{} runs · median {:.1} ms · p95 {:.1} ms · max {:.1} ms",
        args.runs,
        percentile(0.5) * 1000.0,
        percentile(0.95) * 1000.0,
        timings[args.runs - 1] * 1000.0
    );
    Ok(())
}
