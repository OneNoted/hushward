# Model and inference provenance

Hushward downloads `nvidia/NVIDIA-NemotronLabs-RE-USE` from Hugging Face:

- Revision: `31f82f2efad1095034492f236e8dac9239ba713b`
- File: `model.safetensors`
- SHA-256: `06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1`
- Parameters: 9,607,747
- Precision: FP32
- Architecture: 30 bidirectional time/frequency Mamba blocks
- Licence: [OpenMDW-1.1](../licenses/OpenMDW-1.1.txt)

The hash is checked on download and before loading. The source repository does
not contain model weights or audio recordings.

The model uses surrounding audio context. Live processing runs it on short,
overlapping windows; this changes the available context, not the weights or
architecture. It should not be assumed to match full-recording enhancement
quality. The 8 kHz live preset limits output bandwidth to 4 kHz.

Inference derives from `altunenes/re-use-rs` at
`88638f3e137848a50d6aca6906530ed2b2b72acc`. Hushward vendors the relevant source so
its GPU optimisations are reviewable and builds do not depend on a moving Git
branch. The separate streaming model and its unused helpers have been removed.
See [local changes](../vendor/reuse-rs/HUSHWARD-CHANGES.md).
