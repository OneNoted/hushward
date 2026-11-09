# Hushward

Microphone noise suppression for Linux using [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE).
Written in Rust, with PipeWire for audio and CPU or Vulkan inference. Audio stays on your machine.

## Install

Requires Linux, PipeWire and [mise](https://mise.jdx.dev/getting-started.html),
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
Vulkan supports AMD, Intel and NVIDIA GPUs with a working Vulkan driver;
Hushward falls back to CPU when Vulkan is unavailable.
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
Normal builds tune for the local CPU. Use `mise run build:portable` for a binary
you can share with other CPUs; it detects AVX2/AVX-512 and FMA at runtime.
`mise run build:cpu` skips GPU dependencies, and `mise run setup:cpu` installs it.
For a portable CPU-only build: `mise run build:portable -- --no-default-features`.

Use `mise run bench` for timing, `mise run run -- gpus` to list GPUs, or
`mise run run -- --gpu 0 check` to select one. Set `backend = "cpu"` or
`gpu = 0` in the settings file to keep that choice. CPU inference is useful for
recordings; live mode still needs to pass the timing check.

## Licences

- Hushward: [MIT](LICENSE).
- Inference code: [reuse-rs](https://github.com/altunenes/re-use-rs) by Enes Altun,
  [Apache-2.0](vendor/reuse-rs/LICENSE).
- Model weights: [OpenMDW-1.1](licenses/OpenMDW-1.1.txt), downloaded separately.
