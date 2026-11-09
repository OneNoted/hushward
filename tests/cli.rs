use std::{fs, process::Command};

fn command(dir: &std::path::Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_hushward"));
    command
        .env("XDG_CONFIG_HOME", dir.join("config"))
        .env("XDG_DATA_HOME", dir.join("data"));
    command
}

#[test]
fn help_needs_no_model_or_audio_server() {
    let dir = tempfile::tempdir().unwrap();
    let result = command(dir.path()).arg("--help").output().unwrap();
    assert!(result.status.success());
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(text.contains("setup"));
    assert!(text.contains("start"));
}

#[test]
fn invalid_live_settings_fail_before_opening_audio() {
    let dir = tempfile::tempdir().unwrap();
    let result = command(dir.path())
        .args(["live", "--rate", "123"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("8000 or 16000"));
}

#[test]
fn clean_never_overwrites_an_existing_recording() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("existing.wav");
    fs::write(&output, b"original recording").unwrap();
    let result = command(dir.path())
        .args(["clean", "missing.wav"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(fs::read(output).unwrap(), b"original recording");
}

#[test]
fn fetch_rejects_corrupt_weights_without_overwriting_them() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.safetensors");
    fs::write(&model, b"damaged weights").unwrap();
    let result = command(dir.path())
        .arg("--model")
        .arg(&model)
        .arg("fetch")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(fs::read(model).unwrap(), b"damaged weights");
    assert!(String::from_utf8_lossy(&result.stderr).contains("checksum mismatch"));
}
