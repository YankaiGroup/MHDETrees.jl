# Changelog

All notable changes to MHDETrees.jl will be documented in this file.

## 1.0.1 - 2026-08-08

- Rewrite the README as a user-facing installation and usage guide.
- Clarify automatic CUDA.jl installation and NVIDIA GPU training.

## 1.0.0 - 2026-08-04

- Publish the package under the `MHDETrees` name and MIT license.
- Add a unified `fit`/`predict` API supporting full-tree DEOCT and MH-DEOCT on
  both CPU and CUDA backends.
- Bundle a single Iris dataset example.
- Add focused API, CPU, and GPU integration tests.
- Keep GPU training quiet by default and add `MHDEOCTConfig(verbose=true)` for
  detailed diagnostics.
- Support Julia 1.8 and later with CUDA.jl 4, 5, and 6.
