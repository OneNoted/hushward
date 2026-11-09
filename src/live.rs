use crate::{
    audio::{self, VIRTUAL_SOURCE as NAME, move_readers, pactl},
    config::Settings,
    engine::enhance,
};
use anyhow::{Context, Result, bail, ensure};
use reuse_rs::Offline;
use std::{
    io::{IsTerminal, Read, Write},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant},
};

struct Children {
    capture: Option<Child>,
    source: Option<Child>,
    previous: String,
    changed_default: bool,
}
impl Drop for Children {
    fn drop(&mut self) {
        if self.changed_default {
            move_readers(NAME, &self.previous);
            if audio::default_source().is_ok_and(|name| name == NAME) {
                let _ = pactl(&["set-default-source", &self.previous]);
            }
        }
        for child in [&mut self.capture, &mut self.source].into_iter().flatten() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn receive(
    rx: &Receiver<Vec<f32>>,
    stop: &AtomicBool,
    overrun: &AtomicBool,
) -> Result<Option<Vec<f32>>> {
    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        ensure!(
            !overrun.load(Ordering::Relaxed),
            "Inference fell behind the microphone. Stopping instead of accumulating delay."
        );
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(value) => return Ok(Some(value)),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                bail!("Microphone disconnected or recording stopped")
            }
        }
    }
}

// Adjacent estimates are correlated; an equal-power fade would boost their overlap.
fn blend(previous: &[f32], next: &mut [f32]) {
    let denominator = next.len().saturating_sub(1).max(1) as f32;
    for (i, (sample, prior)) in next.iter_mut().zip(previous).enumerate() {
        let weight = i as f32 / denominator;
        *sample = prior * (1.0 - weight) + *sample * weight;
    }
}

pub fn preflight(model: &Offline, settings: &Settings) -> Result<f64> {
    settings.validate()?;
    let samples = vec![0.0; (settings.window as f64 * settings.rate as f64).round() as usize];
    eprintln!("Checking the live preset…");
    for _ in 0..3 {
        enhance(model, &samples, settings.rate as u32).context("Warmup")?;
    }
    let start = Instant::now();
    enhance(model, &samples, settings.rate as u32).context("Preflight")?;
    let elapsed = start.elapsed().as_secs_f64();
    let budget = settings.window as f64 - settings.overlap_ms as f64 / 1000.0;
    ensure!(
        elapsed < budget * 0.9,
        "This preset needs {:.0} ms per window, but only {:.0} ms is available. Try a larger --window or close other GPU-heavy applications. Your audio settings have not changed.",
        elapsed * 1000.0,
        budget * 1000.0
    );
    Ok(elapsed)
}

pub fn run(
    model: &Offline,
    source: &str,
    settings: &Settings,
    duration: Option<f32>,
    make_default: bool,
    verbose: bool,
) -> Result<()> {
    settings.validate()?;
    if let Some(value) = duration {
        ensure!(
            value.is_finite() && value > 0.0,
            "Duration must be positive"
        );
    }
    let previous = audio::default_source()?;
    ensure!(
        !audio::sources()?.iter().any(|s| s.name == NAME),
        "Hushward is already running. Use `hushward status` or `hushward stop`."
    );
    let (window, rate, overlap_ms) = (settings.window, settings.rate, settings.overlap_ms);
    let rate_arg = rate.to_string();
    let window_samples = (window as f64 * rate as f64).round() as usize;
    let overlap = overlap_ms as usize * rate / 1000;
    ensure!(
        overlap >= rate / 1000 && overlap * 2 < window_samples,
        "Overlap must be at least 1 ms and less than half the window"
    );
    let hop = window_samples - overlap;
    let budget = hop as f64 / rate as f64;
    preflight(model, settings)?;

    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed))?;
    let mut children = Children {
        capture: None,
        source: None,
        previous,
        changed_default: false,
    };
    let mut output = Command::new("pw-cat").args([
        "--playback", "--target", "0", "--rate", &rate_arg, "--channels", "1", "--channel-map", "MONO", "--format", "f32", "--raw", "--latency", "10ms",
        "--properties", "{ media.class = Audio/Source node.name = hushward node.description = Hushward application.name = Hushward node.always-process = true node.pause-on-idle = false }", "-"
    ]).stdin(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("Start pw-cat virtual source")?;
    let mut writer = output.stdin.take().context("Virtual source stdin")?;
    children.source = Some(output);
    writer.write_all(&vec![0; rate / 100 * 4])?;
    if make_default {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if pactl(&["set-default-source", NAME]).is_ok() {
                children.changed_default = true;
                break;
            }
            ensure!(
                Instant::now() < deadline,
                "Virtual microphone failed to appear"
            );
            thread::sleep(Duration::from_millis(50));
        }
        // Move current applications before opening our own raw-input capture stream.
        move_readers(&children.previous, NAME);
    }

    let mut capture = Command::new("pw-cat").args([
        "--record", "--target", source, "--rate", &rate_arg, "--channels", "1", "--channel-map", "MONO", "--format", "f32", "--raw", "--latency", "10ms",
        "--properties", "{ application.name = Hushward node.name = hushward.capture node.dont-reconnect = true }", "-"
    ]).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("Start microphone capture")?;
    let mut reader = capture.stdout.take().context("Microphone stdout")?;
    children.capture = Some(capture);
    let (tx, rx) = mpsc::sync_channel(64);
    let overrun = Arc::new(AtomicBool::new(false));
    let overflow = overrun.clone();
    let capture_stop = stop.clone();
    thread::spawn(move || {
        let mut bytes = vec![0u8; rate / 100 * 4];
        while !capture_stop.load(Ordering::Relaxed) {
            if reader.read_exact(&mut bytes).is_err() {
                break;
            }
            let block = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            match tx.try_send(block) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Full(_)) => {
                    overflow.store(true, Ordering::Relaxed);
                    break;
                }
                Err(mpsc::TrySendError::Disconnected(_)) => break,
            }
        }
    });
    eprintln!(
        "Hushward active · {rate} Hz · input {source} · choose Hushward in your app · Ctrl-C dismisses the ward"
    );
    let mut pending = std::collections::VecDeque::new();
    let mut tail = Vec::with_capacity(overlap);
    let mut samples = Vec::with_capacity(window_samples);
    let mut bytes = Vec::with_capacity(hop * 4);
    let terminal = std::io::stderr().is_terminal();
    let report_interval = if terminal {
        Duration::from_secs(1)
    } else {
        Duration::from_secs(30)
    };
    let mut last_report = Instant::now();
    let mut emitted = 0usize;
    let capture_start = Instant::now();
    'stream: loop {
        while pending.len() < window_samples {
            let Some(block) = receive(&rx, &stop, &overrun)? else {
                break 'stream;
            };
            pending.extend(block);
        }
        samples.clear();
        samples.extend(pending.iter().take(window_samples).copied());
        ensure!(
            samples.iter().all(|v| v.is_finite()),
            "Microphone produced invalid samples"
        );
        let start = Instant::now();
        let mut clean = enhance(model, &samples, rate as u32).context("Denoise")?;
        let seconds = start.elapsed().as_secs_f64();
        if !tail.is_empty() {
            blend(&tail, &mut clean[..overlap]);
        }
        bytes.clear();
        bytes.extend(clean[..hop].iter().flat_map(|v| v.to_le_bytes()));
        writer
            .write_all(&bytes)
            .context("The virtual microphone stopped accepting audio")?;
        if verbose || last_report.elapsed() >= report_interval {
            let age = capture_start.elapsed().as_secs_f64() - emitted as f64 / rate as f64;
            eprintln!(
                "Warded {:.1}s · {:.3}s inference / {:.3}s budget · oldest-sample age {:.3}s",
                (emitted + hop) as f64 / rate as f64,
                seconds,
                budget,
                age
            );
            last_report = Instant::now();
        }
        tail.clear();
        tail.extend_from_slice(&clean[hop..]);
        pending.drain(..hop);
        emitted += hop;
        if duration.is_some_and(|limit| emitted as f64 / rate as f64 >= limit as f64) {
            break;
        }
    }
    stop.store(true, Ordering::Relaxed);
    drop(writer);
    drop(children);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlapping_identical_signals_have_unity_gain() {
        let signal: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.03).sin()).collect();
        let mut output = signal.clone();
        blend(&signal, &mut output);
        let max_error = output
            .iter()
            .zip(&signal)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 1e-6);
    }
    #[test]
    fn blend_has_continuous_endpoints() {
        let mut output = vec![1.0; 16000];
        blend(&[0.0; 16000], &mut output);
        assert_eq!(output[0], 0.0);
        assert_eq!(*output.last().unwrap(), 1.0);
        assert!(
            output
                .windows(2)
                .all(|w| w[1] >= w[0] && w[1] - w[0] < 0.0001)
        );
    }
}
