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

## Build comparison

The hardware portability changes were checked against commit `553aa8c` on the
same laptop. Normal mise builds still tune for the local CPU. Portable builds
use runtime SIMD selection; in a separate paired CPU test, their median was
249 ms versus 234 ms for the previous native build.

Paired runs with the normal build, using 160 ms of synthetic 8 kHz audio:

| Backend | Previous median | Current median | Previous p95 | Current p95 |
|---|---:|---:|---:|---:|
| Vulkan, 40 windows each | 111 ms | 112 ms | 114 ms | 122 ms |
| CPU, 20 windows each | 222 ms | 232 ms | 289 ms | 268 ms |

Runs alternated between versions and excluded four warmup windows. These are
comparable timings, not evidence of a speedup. The changes reuse FFT plans and
windows, combine GPU readbacks, and avoid extra copies.

The CPU-only build has 71 external crates in its normal dependency graph and
a roughly 4 MB executable. The Vulkan build has 277, down from 282. The lockfile
contains 68 fewer packages, including optional dependencies that were never
compiled. The GPU shader compiler remains the largest part of the build.

A fresh 30-second live run used the built-in microphone at 8 kHz, with 160 ms
windows and 20 ms overlap. It processed 215 windows: median 109 ms, p95 123 ms,
maximum 139 ms. The virtual microphone delivered 242,661 finite samples.
Internal sample age ranged from 315 to 460 ms, with 400 ms at the start and
404 ms at the end. No queue overflow occurred.

On the same reference excerpt used above, CPU output remained bit-for-bit
identical. Vulkan agreement remained 102.78 dB, with maximum error 6.18e-7.
