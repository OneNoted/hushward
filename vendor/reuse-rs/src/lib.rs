//! FP32 NemotronLabs RE-USE inference, adapted from altunenes/re-use-rs 0.4.0.

pub mod audio;
#[cfg(feature = "gpu")]
mod gpu;
mod layers;
mod offline;
mod ops;
mod weights;

#[cfg(feature = "gpu")]
pub use gpu::GpuDevice;
pub use offline::Offline;
