use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::process::Command;

pub const VIRTUAL_SOURCE: &str = "hushward";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Source {
    pub index: u32,
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing)]
    active_port: Option<String>,
    #[serde(default, skip_serializing)]
    ports: Vec<Port>,
}

#[derive(Clone, Debug, Deserialize)]
struct Port {
    name: String,
    availability: String,
}

impl Source {
    fn available(&self) -> bool {
        !self.ports.iter().any(|p| {
            self.active_port.as_ref() == Some(&p.name) && p.availability == "not available"
        })
    }
}

pub fn pactl(args: &[&str]) -> Result<String> {
    let output = Command::new("pactl").args(args).output().context(
        "pactl is missing. Install your distribution's PulseAudio client tools and pipewire-pulse.",
    )?;
    ensure!(
        output.status.success(),
        "Audio server: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub fn sources() -> Result<Vec<Source>> {
    serde_json::from_str(&pactl(&["--format=json", "list", "sources"])?)
        .context("Read microphones from PipeWire")
}

pub fn microphones() -> Result<Vec<Source>> {
    Ok(sources()?
        .into_iter()
        .filter(|s| s.name != VIRTUAL_SOURCE && !s.name.ends_with(".monitor") && s.available())
        .collect())
}

pub fn default_source() -> Result<String> {
    pactl(&["get-default-source"])
}

pub fn resolve_source(preferred: Option<&str>) -> Result<String> {
    let name = preferred
        .map(str::to_owned)
        .map(Ok)
        .unwrap_or_else(default_source)?;
    ensure!(
        microphones()?.iter().any(|s| s.name == name),
        "Microphone '{name}' is unavailable. Plug it in or run `hushward setup` to choose another input."
    );
    Ok(name)
}

pub fn devices(json: bool) -> Result<()> {
    let sources = microphones()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&sources)?);
    } else {
        let default = default_source()?;
        for source in sources {
            let marker = if source.name == default {
                " (default)"
            } else {
                ""
            };
            println!("{}{marker}\n  {}", source.description, source.name);
        }
    }
    Ok(())
}

pub fn move_readers(from: &str, to: &str) {
    let Ok(sources) = sources() else {
        return;
    };
    let Some(source) = sources.iter().find(|s| s.name == from) else {
        return;
    };
    let Ok(streams) = pactl(&["list", "source-outputs", "short"]) else {
        return;
    };
    for line in streams.lines() {
        let mut fields = line.split_whitespace();
        if let (Some(id), Some(index)) = (fields.next(), fields.next())
            && index.parse::<u32>().ok() == Some(source.index)
        {
            // Streams may disappear between listing and moving them.
            let _ = pactl(&["move-source-output", id, to]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disconnected_ports_are_not_offered_as_microphones() {
        let source: Source = serde_json::from_str(r#"{"index":1,"name":"mic","description":"Headset","active_port":"jack","ports":[{"name":"jack","availability":"not available"}]}"#).unwrap();
        assert!(!source.available());
        let source: Source =
            serde_json::from_str(r#"{"index":2,"name":"usb","description":"USB mic"}"#).unwrap();
        assert!(source.available());
    }
}
