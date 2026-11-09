# Measured performance

These results were collected on 2026-11-09 using a Ryzen AI 7 350 / Radeon 860M,
Linux 7.2.15 and the full FP32 NemotronLabs RE-USE weights. The build targeted
the local CPU. Results from one machine are not a performance guarantee for
other hardware or workloads.

## Live microphone

A one-minute run used the RØDE AI-Micro with the default speech preset:
8 kHz, 160 ms windows and 20 ms overlap. Each window had a 140 ms processing
budget.

| Metric | Result |
|---|---:|
| Windows processed | 429 |
| Median inference | 102 ms |
| 95th percentile inference | 136 ms |
| Maximum inference | 138 ms |
| Non-finite output samples | 0 |
| Samples read from the virtual microphone | 478,949 |
| Internal capture-to-write age | 313–477 ms |
| First / final internal age | 448 / 381 ms |

The internal age did not grow over the run. This measurement includes startup
and buffering; it does not measure the latency of a remote call. The virtual
microphone reader attached after startup, so its sample count is shorter than
the complete processing run.

At 8 kHz, audio bandwidth is limited to 4 kHz. The same low-delay preset at
16 kHz did not keep up on this laptop. Closing other GPU-heavy applications can
help, but each computer needs its own timing check.

## Numerical agreement

The optimised Vulkan path was compared with the original native CPU path on a
public 500 ms excerpt of NVIDIA's `mic_test2.wav`:

| Sample rate | Signal-to-error ratio | Maximum absolute error |
|---|---:|---:|
| 8 kHz | 102.78 dB | 6.18e-7 |
| 16 kHz | 90.19 dB | 1.25e-6 |

Both produced finite samples of the expected length. This checks implementation
agreement, not subjective noise-removal quality or equivalence to full-recording
inference when short live windows are used.

## Reproduce

After setup, check throughput without opening the microphone:

```sh
mise run check
mise run run -- bench --rate 8000 --seconds 0.16 --runs 20
```

The benchmark excludes model loading and its warmup pass. To collect live
per-window timing, stop the service first and run in the foreground:

```sh
mise run stop
mise run run -- live --seconds 60 --verbose
```

To compare the CPU and Vulkan implementations using deterministic test audio:

```sh
mise run test:gpu
```

That test needs the downloaded model and a working Vulkan device. It is skipped
in ordinary CI, which does not have a GPU. Set `HUSHWARD_TEST_MODEL` to test with
a model stored outside the standard data directory.
