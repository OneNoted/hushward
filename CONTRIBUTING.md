# Contributing

Run `mise install`, then `mise run ci` before opening a pull request. Keep fixes
focused and include a regression test when behaviour changes.
Run `mise run ci:cpu` for the CPU-only build, too.

Audio-path changes need more than a passing unit test. Compare output against
the current FP32 implementation, measure warm inference time, and exercise a
sustained live session. Report the hardware, sample rate, window, overlap, and
whether timings include loading or shader compilation. Keep personal recordings
and downloaded model files out of commits.

Changes to the vendored engine belong in its change log. Preserve upstream
attribution and licence notices. Model weights are fetched from the pinned
source; do not add them to the repository.
