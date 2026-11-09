use reuse_rs::{GpuDevice, Offline};
use std::{env, path::PathBuf};

#[test]
#[ignore = "requires downloaded model weights and a Vulkan device"]
fn vulkan_matches_cpu() {
    let path = env::var_os("HUSHWARD_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let data = env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(env::var_os("HOME").unwrap()).join(".local/share")
                });
            data.join("hushward/model.safetensors")
        });
    assert!(path.is_file(), "Run setup or set HUSHWARD_TEST_MODEL first");
    let samples: Vec<f32> = (0..4000)
        .map(|i| {
            let t = i as f32 / 8000.0;
            0.1 * (std::f32::consts::TAU * 180.0 * t).sin()
                + 0.03 * (std::f32::consts::TAU * 710.0 * t).sin()
        })
        .collect();
    let cpu = Offline::load(&path)
        .unwrap()
        .enhance(&samples, 8000)
        .unwrap();
    let gpu = Offline::load_gpu(path, GpuDevice::Vulkan)
        .unwrap()
        .enhance(&samples, 8000)
        .unwrap();
    assert_eq!(cpu.len(), samples.len());
    assert_eq!(gpu.len(), samples.len());
    assert!(cpu.iter().chain(&gpu).all(|v| v.is_finite()));
    let power: f64 = cpu.iter().map(|&v| (v as f64).powi(2)).sum();
    let error: f64 = cpu
        .iter()
        .zip(&gpu)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum();
    let snr = 10.0 * (power / error.max(1e-30)).log10();
    assert!(snr > 60.0, "CPU/Vulkan agreement is only {snr:.2} dB");
}
