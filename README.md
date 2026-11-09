# Hushward

Keep the tavern out of your microphone.

Hushward turns a Linux microphone into a noise-suppressed virtual input for calls,
voice chat and dictation. It runs NVIDIA’s **NemotronLabs RE-USE** model locally
on your GPU, using the full FP32 weights. It can also clean WAV recordings.

## Get started

You need Linux with PipeWire, a working Vulkan driver, and [mise](https://mise.jdx.dev/getting-started.html).
The live service also needs `pw-cat`, `pactl`, `curl`, and systemd user services.
`pactl` comes from your distribution’s PulseAudio client tools; you do not need
to replace PipeWire with PulseAudio.

From this checkout:

```sh
mise trust
mise install
mise run setup
mise run start
```

Setup lets you choose a microphone, downloads and verifies the model once
(39 MB), and installs the `hushward` command and a user service. It does not need
root or enable login autostart.

Choose **Hushward** as the input in your call app. Apps following the system
default switch automatically; apps with a saved microphone choice may need
that one change.

```sh
mise run status
mise run stop
```

Stopping restores the previous microphone. After setup, `hushward start`,
`hushward stop` and `hushward status` also work outside the checkout.
`mise run uninstall` removes the command and service while keeping your model
and settings.

## What to expect

The default live preset uses **8 kHz audio**, a 160 ms window and a 20 ms
crossfade. It favours speech and has telephone bandwidth: frequencies above
4 kHz are not retained. This is useful for conversations, not music recording.

On a Ryzen AI 7 350 with Radeon 860M, a one-minute AI-Micro test completed all
429 windows without a growing backlog. Median inference took 102 ms against a
140 ms processing budget. Internal buffering and processing measured roughly
0.3–0.5 seconds; your call app and network add their own delay.

Performance depends on your hardware and other GPU work. Hushward checks the
preset before switching your audio and stops if processing falls too far behind.
It never silently switches to another model. [Measurements and method](docs/benchmarks.md).

## Check your setup

```sh
mise run doctor
mise run check
mise run run -- devices
```

`doctor` checks the audio tools, PipeWire and model file. `check` measures the
saved live preset without opening the microphone. To choose a different input,
run `mise run setup` again.

Settings live in `$XDG_CONFIG_HOME/hushward/config.toml` (normally
`~/.config/hushward/config.toml`). The model lives in
`$XDG_DATA_HOME/hushward/model.safetensors` (normally `~/.local/share/hushward/`).
The downloaded weights are pinned by revision and SHA-256. Audio processing
never uploads your recordings or live microphone audio.

For foreground use and temporary overrides:

```sh
mise run stop
mise run run -- live --window 0.20 --make-default
```

Press Ctrl-C to stop. `--rate 16000` offers wider bandwidth if your hardware can
keep up. `--source NAME` selects a microphone from `devices`. `--verbose` prints
per-window timings. Ordinary service logs stay quiet:

```sh
journalctl --user -u hushward -n 20
```

## Clean a recording

```sh
mise run enhance -- noisy.wav clean.wav
mise run run -- --backend cpu clean noisy.wav clean.wav
```

Input is downmixed to mono; output is a 32-bit float WAV at the original sample
rate. Existing output files are never overwritten. Supported rates are 8, 16,
22.05, 24, 32, 44.1 and 48 kHz. Long recordings use overlapping model windows,
although the input and output audio still reside in memory.

## Development

mise pins the Rust toolchain and owns the development commands:

| Command | Purpose |
|---|---|
| `mise run build` | Build the release binary |
| `mise run fmt` | Format Rust and mise configuration |
| `mise run lint` | Check formatting, Clippy and GitHub Actions |
| `mise run test` | Run application and inference tests |
| `mise run ci` | Run all checks |
| `mise run run -- --help` | Browse CLI options |

Builds target the local CPU for efficient inference. Build on the machine that
will run Hushward; do not redistribute that binary as a generic Linux release.
The source and setup flow work independently of the checkout location.

The inference engine keeps the convolutional encoder and decoder on the GPU
and combines small normalisation and residual operations into fewer GPU launches.
The trained weights and all 30 model blocks are retained. The local engine changes
are documented in [vendor/reuse-rs/HUSHWARD-CHANGES.md](vendor/reuse-rs/HUSHWARD-CHANGES.md).

## Credits and licences

- Hushward’s application code: [MIT](LICENSE).
- Inference implementation: adapted from [reuse-rs](https://github.com/altunenes/re-use-rs)
  by Enes Altun, under [Apache-2.0](vendor/reuse-rs/LICENSE).
- Model: [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE),
  under [OpenMDW-1.1](licenses/OpenMDW-1.1.txt). Weights are downloaded separately.

Hushward uses the full NemotronLabs checkpoint, not the separate Real-time RE-USE
model. See [model provenance](docs/model.md) for the pinned revision and checksum.
