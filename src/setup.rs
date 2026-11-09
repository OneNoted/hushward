use crate::{
    assets, audio,
    config::{Paths, Settings},
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    fs,
    io::{self, IsTerminal, Write},
    os::unix::fs::PermissionsExt,
    process::Command,
    thread,
    time::{Duration, Instant},
};

const SERVICE: &str = include_str!("../packaging/hushward.service");

pub fn doctor(paths: &Paths) -> Result<()> {
    println!("Hushward — checking the essentials\n");
    for (tool, hint) in [
        ("pw-cat", "PipeWire audio tools"),
        ("pactl", "PulseAudio client tools"),
        ("curl", "curl"),
        ("systemctl", "systemd user services"),
    ] {
        ensure!(
            Command::new(tool).arg("--version").output().is_ok(),
            "Missing {tool}. Install {hint} using your distribution's package manager."
        );
        println!("  OK  {tool}");
    }
    println!(
        "  OK  Audio server ({} microphones)",
        audio::microphones()?.len()
    );
    let settings = paths.load()?;
    let model = settings.model.as_deref().unwrap_or(&paths.model);
    if model.exists() {
        assets::verify(model)?;
        println!("  OK  RE-USE model");
    } else {
        println!("  —   Model not downloaded yet; `hushward setup` handles this.");
    }
    println!("\nSettings: {}", paths.config.display());
    Ok(())
}

pub fn run(paths: &Paths, mut settings: Settings, source: Option<String>, yes: bool) -> Result<()> {
    doctor(paths)?;
    let sources = audio::microphones()?;
    ensure!(
        !sources.is_empty(),
        "No microphones found. Connect one, then run setup again."
    );
    let find = |name: &str| sources.iter().position(|s| s.name == name);
    let requested = source
        .as_deref()
        .map(|name| {
            find(name).with_context(|| {
                format!("Microphone '{name}' is unavailable. Run `hushward devices`.")
            })
        })
        .transpose()?;
    let choice = requested
        .or_else(|| settings.source.as_deref().and_then(find))
        .or_else(|| audio::default_source().ok().as_deref().and_then(find))
        .or_else(|| (sources.len() == 1).then_some(0));
    let index = if yes || requested.is_some() || sources.len() == 1 {
        choice
            .context("No available default microphone. Pass `--source` from `hushward devices`.")?
    } else {
        ensure!(
            io::stdin().is_terminal(),
            "Interactive setup needs a terminal. Use `setup --yes --source NAME` for unattended setup."
        );
        println!("\nWhich microphone should the ward protect?");
        for (i, source) in sources.iter().enumerate() {
            println!(
                "  {}. {}{}",
                i + 1,
                source.description,
                if Some(i) == choice { " (current)" } else { "" }
            );
        }
        loop {
            print!(
                "Microphone{}: ",
                choice.map(|i| format!(" [{}]", i + 1)).unwrap_or_default()
            );
            io::stdout().flush()?;
            let mut line = String::new();
            ensure!(io::stdin().read_line(&mut line)? != 0, "Setup cancelled.");
            if line.trim().is_empty()
                && let Some(index) = choice
            {
                break index;
            }
            if let Ok(index) = line.trim().parse::<usize>()
                && (1..=sources.len()).contains(&index)
            {
                break index - 1;
            }
            println!("Choose a number from 1 to {}.", sources.len());
        }
    };
    settings.source = Some(sources[index].name.clone());
    let model = settings.model.as_deref().unwrap_or(&paths.model);
    assets::fetch(model)?;
    install_binary(paths)?;
    write_service(paths)?;
    paths.save(&settings)?;
    systemctl(&["daemon-reload"])?;
    println!(
        "\nReady. {} is selected for the ward.",
        sources[index].description
    );
    println!("Start: hushward start\nStop:  hushward stop\nCheck performance: hushward check");
    println!("Setup does not enable login autostart.");
    Ok(())
}

fn install_binary(paths: &Paths) -> Result<()> {
    let current = std::env::current_exe()?;
    if current.canonicalize()? == paths.binary.canonicalize().unwrap_or_default() {
        return Ok(());
    }
    let parent = paths.binary.parent().unwrap();
    fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    io::copy(&mut fs::File::open(current)?, &mut file)?;
    file.as_file()
        .set_permissions(fs::Permissions::from_mode(0o755))?;
    file.as_file().sync_all()?;
    file.persist(&paths.binary)?;
    Ok(())
}

fn write_service(paths: &Paths) -> Result<()> {
    let parent = paths.service.parent().unwrap();
    fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(SERVICE.as_bytes())?;
    file.as_file().sync_all()?;
    file.persist(&paths.service)?;
    Ok(())
}

fn systemctl(args: &[&str]) -> Result<()> {
    let status = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .status()
        .context("Run systemctl --user")?;
    ensure!(
        status.success(),
        "systemctl {} failed. Run `hushward doctor` to check the setup.",
        args.join(" ")
    );
    Ok(())
}

pub fn control(action: &str, paths: &Paths) -> Result<()> {
    ensure!(
        paths.service.exists(),
        "Run `mise run setup` once to install Hushward."
    );
    systemctl(&[action, "hushward"])?;
    if action == "start" {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if audio::sources()?
                .iter()
                .any(|s| s.name == audio::VIRTUAL_SOURCE)
            {
                break;
            }
            let state = Command::new("systemctl")
                .args(["--user", "is-failed", "--quiet", "hushward"])
                .status()?;
            if state.success() || Instant::now() > deadline {
                let output = Command::new("journalctl")
                    .args([
                        "--user",
                        "-u",
                        "hushward",
                        "-n",
                        "8",
                        "--no-pager",
                        "--output=cat",
                    ])
                    .output()?;
                bail!(
                    "The ward could not start:\n{}",
                    String::from_utf8_lossy(&output.stdout)
                );
            }
            thread::sleep(Duration::from_millis(100));
        }
        println!("Ward raised. Choose Hushward as the microphone in your call app.");
    } else {
        println!("Ward dismissed.");
    }
    Ok(())
}

pub fn status(paths: &Paths) -> Result<()> {
    let active = Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", "hushward"])
        .status()?
        .success();
    if active {
        println!("Ward is up.");
    } else {
        println!("Ward is resting. Start it with `hushward start`.");
    }
    let settings = paths.load()?;
    if let Some(source) = settings.source {
        println!("Input: {source}");
    }
    println!(
        "Preset: {} Hz · {:.0} ms window",
        settings.rate,
        settings.window * 1000.0
    );
    Ok(())
}

pub fn uninstall(paths: &Paths) -> Result<()> {
    if paths.service.exists() {
        systemctl(&["disable", "--now", "hushward"])?;
        fs::remove_file(&paths.service)?;
        systemctl(&["daemon-reload"])?;
    }
    if paths.binary.symlink_metadata().is_ok() {
        fs::remove_file(&paths.binary)?;
    }
    println!("Removed the command and service. Your model and settings are kept.");
    Ok(())
}
