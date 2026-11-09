use crate::Backend;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub source: Option<String>,
    pub model: Option<PathBuf>,
    pub backend: Backend,
    pub threads: usize,
    pub rate: usize,
    pub window: f32,
    pub overlap_ms: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            source: None,
            model: None,
            backend: Backend::Vulkan,
            threads: 4,
            rate: 8000,
            window: 0.16,
            overlap_ms: 20,
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<()> {
        if let Some(model) = &self.model {
            ensure!(model.is_absolute(), "model must be an absolute path");
        }
        ensure!(
            (1..=256).contains(&self.threads),
            "threads must be between 1 and 256"
        );
        ensure!(
            [8000, 16000].contains(&self.rate),
            "live rate must be 8000 or 16000 Hz"
        );
        ensure!(
            self.window.is_finite() && (0.1..=10.0).contains(&self.window),
            "window must be between 0.1 and 10 seconds"
        );
        ensure!(
            self.overlap_ms > 0 && self.overlap_ms as f32 * 2.0 < self.window * 1000.0,
            "overlap_ms must be positive and less than half the window"
        );
        Ok(())
    }
}

pub struct Paths {
    pub config: PathBuf,
    pub model: PathBuf,
    pub binary: PathBuf,
    pub service: PathBuf,
}

impl Paths {
    pub fn discover() -> Result<Self> {
        let home = PathBuf::from(env::var_os("HOME").context("HOME is not set")?);
        let xdg = |name: &str, fallback: &str| {
            env::var_os(name)
                .map(PathBuf::from)
                .filter(|p| p.is_absolute())
                .unwrap_or_else(|| home.join(fallback))
        };
        Ok(Self::new(
            xdg("XDG_CONFIG_HOME", ".config"),
            xdg("XDG_DATA_HOME", ".local/share"),
            home.join(".local/bin"),
        ))
    }

    fn new(config: PathBuf, data: PathBuf, bin: PathBuf) -> Self {
        Self {
            config: config.join("hushward/config.toml"),
            model: data.join("hushward/model.safetensors"),
            binary: bin.join("hushward"),
            service: config.join("systemd/user/hushward.service"),
        }
    }

    pub fn load(&self) -> Result<Settings> {
        match fs::read_to_string(&self.config) {
            Ok(text) => {
                let settings: Settings = toml::from_str(&text)
                    .with_context(|| format!("Read {}", self.config.display()))?;
                settings.validate()?;
                Ok(settings)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Settings::default()),
            Err(e) => Err(e).with_context(|| format!("Read {}", self.config.display())),
        }
    }

    pub fn save(&self, settings: &Settings) -> Result<()> {
        settings.validate()?;
        let parent = self.config.parent().unwrap();
        fs::create_dir_all(parent)?;
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        use std::io::Write;
        file.write_all(toml::to_string_pretty(settings)?.as_bytes())?;
        file.as_file().sync_all()?;
        file.persist(&self.config)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn settings_roundtrip_and_reject_invalid_values() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(
            dir.path().join("config"),
            dir.path().join("data"),
            dir.path().join("bin"),
        );
        let settings = Settings {
            source: Some("usb microphone".into()),
            ..Settings::default()
        };
        paths.save(&settings).unwrap();
        assert_eq!(paths.load().unwrap().source, settings.source);
        assert!(
            Settings {
                window: f32::NAN,
                ..settings.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            Settings {
                overlap_ms: 80,
                ..settings
            }
            .validate()
            .is_err()
        );
        fs::write(&paths.config, "soruce = 'typo'\n").unwrap();
        assert!(paths.load().is_err());
    }
}
