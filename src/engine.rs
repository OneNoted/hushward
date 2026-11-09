use crate::{assets, config::Settings};
use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use reuse_rs::Offline;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Vulkan,
}

pub fn load(path: &Path, settings: &Settings) -> Result<(Offline, Backend)> {
    ensure!(
        path.is_file(),
        "Model not found at {}. Run `hushward setup` or `hushward fetch`.",
        path.display()
    );
    assets::verify(path)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(settings.threads)
        .build_global()?;
    if settings.backend != Backend::Cpu {
        match load_vulkan(path, settings.gpu) {
            Ok(model) => {
                eprintln!("RE-USE · Vulkan");
                return Ok((model, Backend::Vulkan));
            }
            Err(error) if settings.backend == Backend::Auto && settings.gpu.is_none() => {
                eprintln!("Vulkan unavailable ({error:#}); using CPU.");
            }
            Err(error) => return Err(error),
        }
    }
    eprintln!("RE-USE · CPU · {} threads", settings.threads);
    let model = Offline::load(path).context("Load CPU model")?;
    Ok((model, Backend::Cpu))
}

#[cfg(feature = "vulkan")]
fn load_vulkan(path: &Path, index: Option<usize>) -> Result<Offline> {
    gpu_call(|| {
        let device = index.map_or(
            reuse_rs::GpuDevice::Vulkan,
            reuse_rs::GpuDevice::VulkanIndex,
        );
        Offline::load_gpu(path, device).context("Load Vulkan model")
    })
}

#[cfg(not(feature = "vulkan"))]
fn load_vulkan(_: &Path, _: Option<usize>) -> Result<Offline> {
    bail!("this build is CPU-only; use `mise run build` for Vulkan support")
}

pub fn list_gpus() -> Result<()> {
    #[cfg(feature = "vulkan")]
    {
        let devices = gpu_call(|| Ok(reuse_rs::vulkan_devices()))?;
        if devices.is_empty() {
            println!("No Vulkan devices found. CPU inference is available.");
        }
        for (index, device) in devices.iter().enumerate() {
            println!("{index}: {device}");
        }
        Ok(())
    }
    #[cfg(not(feature = "vulkan"))]
    bail!("this build is CPU-only; use `mise run build` for Vulkan support")
}

// The GPU runtime reports some driver failures by panicking instead of returning an error.
#[cfg(feature = "vulkan")]
fn gpu_call<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|panic| {
        let message = panic.downcast_ref::<String>().map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("GPU runtime failed");
        bail!("Vulkan initialization failed: {message}. Try `--backend cpu` or check your Vulkan driver.")
    })
}
pub fn enhance(model: &Offline, samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    let output = model.enhance(samples, rate).context("Inference")?;
    if output.len() != samples.len() || !output.iter().all(|v| v.is_finite()) {
        bail!("The model returned invalid audio.");
    }
    Ok(output)
}
