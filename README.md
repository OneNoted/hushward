# Hushward

Microphone noise suppression for Linux using [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE).
Written in Rust, with PipeWire for audio and Vulkan for inference. Audio stays on your machine.

## Install

Requires Linux, PipeWire, Vulkan and [mise](https://mise.jdx.dev/getting-started.html),
plus `pw-cat`, `pactl`, `curl` and systemd user services.

```sh
git clone https://github.com/OneNoted/hushward.git
cd hushward
mise trust
mise install
mise run setup
mise run start
```

Setup asks which microphone to use and downloads the model (39 MB).
Select **Hushward** in your call app if it doesn't follow the system default.

## Use

```sh
mise run status
mise run stop
mise run start
```

Stopping restores your previous microphone. Run `mise run setup` to change inputs,
`mise run doctor` to troubleshoot, or `mise run uninstall` to remove Hushward.

To clean a recording:

```sh
mise run enhance -- noisy.wav clean.wav
```

## Quality and delay

Live mode defaults to 8 kHz speech: telephone bandwidth. On the tested Ryzen AI 7
350 / Radeon 860M, buffering and processing added roughly 0.3–0.5 seconds.
Run `mise run check` to see whether your computer keeps up.
[Measurements](docs/benchmarks.md).

Settings are in `~/.config/hushward/config.toml`, or under `XDG_CONFIG_HOME` if set.
The full FP32 model is used; short live windows give it less context than recordings.
[Model details](docs/model.md).

## Development

`mise run ci` runs all checks. `mise tasks` lists the other commands.
Builds target the local CPU, so build on the machine that will run Hushward.

## Licences

- Hushward: [MIT](LICENSE).
- Inference code: [reuse-rs](https://github.com/altunenes/re-use-rs) by Enes Altun,
  [Apache-2.0](vendor/reuse-rs/LICENSE).
- Model weights: [OpenMDW-1.1](licenses/OpenMDW-1.1.txt), downloaded separately.
