use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{IsTerminal, Read},
    path::Path,
    process::Command,
};

pub const MODEL_SHA256: &str = "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1";
const MODEL_URL: &str = "https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/resolve/31f82f2efad1095034492f236e8dac9239ba713b/model.safetensors";

pub fn verify(path: &Path) -> Result<()> {
    let mut file = File::open(path).with_context(|| format!("Open model {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    ensure!(
        format!("{:x}", hasher.finalize()) == MODEL_SHA256,
        "Model checksum mismatch at {}. Remove that file and run `hushward fetch` to download the original weights again.",
        path.display()
    );
    Ok(())
}

pub fn fetch(path: &Path) -> Result<()> {
    if path.exists() {
        verify(path)?;
        println!("Model verified. Ready when you are.");
        return Ok(());
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let file = tempfile::NamedTempFile::new_in(parent)?;
    eprintln!("Fetching RE-USE (39 MB, downloaded once)…");
    let mut command = Command::new("curl");
    command.args([
        "--fail",
        "--location",
        "--retry",
        "2",
        "--connect-timeout",
        "15",
        "--max-time",
        "300",
        "--proto",
        "=https",
    ]);
    if std::io::stderr().is_terminal() {
        command.arg("--progress-bar");
    } else {
        command.args(["--silent", "--show-error"]);
    }
    let status = command
        .arg("--output")
        .arg(file.path())
        .arg(MODEL_URL)
        .status()
        .context("curl is required to download the model")?;
    ensure!(
        status.success(),
        "Model download failed. Check your connection and run `hushward fetch` again."
    );
    verify(file.path())?;
    file.persist_noclobber(path)
        .with_context(|| format!("Save model to {}", path.display()))?;
    println!("Model downloaded and verified.");
    Ok(())
}

pub fn write_wav(path: &Path, samples: &[f32], rate: u32) -> Result<()> {
    ensure!(
        !samples.is_empty() && samples.iter().all(|x| x.is_finite()),
        "The model returned invalid audio. No output was written."
    );
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::new(file.as_file_mut(), spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path).with_context(|| {
        format!(
            "Save {} without overwriting an existing file",
            path.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_wrong_weights() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model");
        fs::write(&path, b"wrong weights").unwrap();
        assert!(verify(&path).is_err());
    }
    #[test]
    fn preserves_existing_recordings_and_rejects_invalid_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("original.wav");
        fs::write(&path, b"original").unwrap();
        assert!(write_wav(&path, &[0.1; 640], 16000).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original");
        let invalid = dir.path().join("invalid.wav");
        assert!(write_wav(&invalid, &[f32::NAN], 16000).is_err());
        assert!(!invalid.exists());
        let output = dir.path().join("clean.wav");
        write_wav(&output, &[0.1; 640], 16000).unwrap();
        assert_eq!(hound::WavReader::open(output).unwrap().duration(), 640);
    }
}
