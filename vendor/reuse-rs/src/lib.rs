//! FP32 NemotronLabs RE-USE inference, adapted from altunenes/re-use-rs 0.4.0.

pub mod audio;
#[cfg(feature = "vulkan")]
mod gpu;
mod layers;
mod offline;
mod ops;
mod weights;

#[cfg(feature = "vulkan")]
pub use gpu::{vulkan_devices, GpuDevice};
pub use offline::Offline;
