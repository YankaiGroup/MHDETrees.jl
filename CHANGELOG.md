# Changelog

All notable changes to MHDETrees.jl will be documented in this file.

## 1.2.0 - 2026-10-08

- Accumulate classification leaf counts per GPU block in shared memory instead
  of in a per-thread buffer that is summed after every fitness evaluation.
  Trained trees are unchanged, and the per-thread buffer and the sum step after
  each evaluation are gone. Set `MHDETrees.oct_gpu.SHARED_COUNTS[] = false` to
  use the previous kernel.
- Fit the CART warm start of the GPU backend on the GPU. The split search runs
  on the GPU and the random draws are replayed on the CPU, so warm-start trees
  and trained trees are unchanged. Set `MHDETrees.warmstart_gpu.GPU_CART[] =
  false` to fit it on the CPU.
- Keep each MH-DEOCT node's training data on the GPU and build its candidate
  thresholds there, so each column is sorted once and shared with the CART
  warm start. Set `MHDETrees.de_gpu.GPU_NODE_DATA[] = false` to extract node
  data and sort thresholds on the CPU.
- Together these changes cut a depth-8 MH-DEOCT fit on 8.25 million HIGGS
  samples from 71.5 to 31.5 minutes on an H100-2g.20gb MIG instance, and the
  fit now also runs on H100-1g.10gb.

## 1.1.1 - 2026-09-29

- Score complete trees on the GPU during MH-DEOCT subtree selection instead of
  on a single CPU core. Trained trees are unchanged; a depth-8 run on 8.25
  million HIGGS samples drops from 4.05 h to 1.19 h on an H100-2g.20gb MIG
  instance.

## 1.1.0 - 2026-08-23

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
