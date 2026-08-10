# Changelog

All notable changes to MHDETrees.jl will be documented in this file.

## Unreleased

- Add regression support for DEOCT and MH-DEOCT on CPU and CUDA backends.
- Minimize within-leaf squared error and predict mean training targets at
  nonempty leaves.
- Add MSE, RMSE, and R2 metrics plus the UCI Yacht Hydrodynamics dataset.
- Add a greedy regression CART baseline.
- Add regression integration tests for CPU and CUDA backends.
- Add independent multivariate affine regression leaves with configurable ridge
  regularization and global affine fallbacks for empty leaves.
- Redesign regression GPU fitness around persistent device data, direct feature
  indices, a two-dimensional sample/candidate grid, reusable workspaces,
  device-side DE replacement and best selection, and batched MH row routing.
- Store regression GPU leaf counts as `Int32` and sums, squared sums,
  thresholds, fitness costs, and DE state as `Float32`; add CPU/GPU parity,
  fixed-tree routing, depth-2/4/6/8, and warm-start integration gates.

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
