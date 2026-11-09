# Local changes

Based on altunenes/re-use-rs 0.4.0, upstream commit
`88638f3e137848a50d6aca6906530ed2b2b72acc` (Apache-2.0).

- Added a full GPU encoder and decoder, preserving convolution, normalization,
  activation, subpixel ordering, and FP32 parameters.
- Kept intermediate activations on the GPU across the complete model.
- Fused bidirectional normalization, bias, outer residual addition and axis
  transpose into one kernel, specialized to the verified checkpoint's width 64.
- Kept recurrent states and computation in FP32. No model layers or trained
  parameters were removed or quantized.
- Reuse FFT plans and Hann windows across audio windows, and read magnitude
  and phase back from the GPU together.
- Detect AVX2/AVX-512 and FMA at runtime for CPU scans and activations, so portable builds keep
  the optimized path on supported processors.
- Use Burn's core crate and share `anyhow` with the application. Only the Vulkan
  GPU feature is retained; CPU builds do not depend on Burn or CubeCL.
- Optional development profiling: `HUSHWARD_PROFILE=1`. To compare the original
  CPU convolution path: `HUSHWARD_CPU_CONV=1`.

The separate Real-time RE-USE model, streaming decoder and unused streaming
convolution and audio helpers have been removed. Hushward uses only the full checkpoint.
