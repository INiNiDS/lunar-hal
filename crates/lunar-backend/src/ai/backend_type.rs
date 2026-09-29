#[cfg(feature = "wgpu")]
pub type B = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "cuda"))]
pub type B = burn::backend::Cuda;
#[cfg(all(not(feature = "wgpu"), not(feature = "cuda"), feature = "metal"))]
pub type B = burn::backend::Metal;
#[cfg(all(
    not(feature = "wgpu"),
    not(feature = "cuda"),
    not(feature = "metal"),
    feature = "rocm"
))]
pub type B = burn::backend::Rocm;
#[cfg(all(
    not(feature = "wgpu"),
    not(feature = "cuda"),
    not(feature = "metal"),
    not(feature = "rocm")
))]
pub type B = burn::backend::Wgpu;
