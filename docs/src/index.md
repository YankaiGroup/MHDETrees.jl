# MHDETrees.jl

MHDETrees.jl implements full-tree and moving-horizon differential evolution for
optimal classification and regression trees. Both DE algorithms support CPU
and CUDA backends; a CPU regression CART baseline is also available. Regression
leaves can predict either their mean training targets or independent affine
models of all input features.

| Algorithm | CPU | GPU |
|---|---:|---:|
| Regression CART (`algorithm=:cart`) | supported | no |
| `algorithm=:deoct` | supported | supported |
| `algorithm=:mhdeoct` | supported | supported |

```@contents
```

## Quick start

```julia
using MHDETrees

config = MHDEOCTConfig(
    task=:classification, # or :regression
    leaf_model=:mean, # or :linear for regression
    algorithm=:mhdeoct, # or :deoct
    backend=:cpu, # or :gpu
    depth=2,
    horizon=2,
    population_size=100,
    generations=600,
    seed=1,
    verbose=false,
)

model = fit(X_train, y_train; config)
y_pred = predict(model, X_test)
score = accuracy(model, X_test, y_test)
```

For regression, use `mean_squared_error`, `root_mean_squared_error`, or
`r2_score` instead of `accuracy`. `leaf_model=:linear` fits ridge-stabilized
affine leaves after the standard mean-SSE tree search. The bundled
`load_yacht()` dataset provides a small regression example.
The optimized Float32 regression-GPU implementation keeps training data and
split tables resident on the device, uses direct feature indices and a 2-D
sample/candidate grid, fuses moment reduction with device-side SSE evaluation,
keeps DE replacement on the GPU, and batches same-level MH row routing.

The GPU backend requires an NVIDIA GPU visible to CUDA.jl. The CPU backend and
prediction do not require a GPU. `horizon` applies to `algorithm=:mhdeoct`;
full-tree DEOCT ignores it. Set `verbose=true` to print GPU kernel, timing, and
optimization diagnostics; the default is quiet.

## Public API

```@autodocs
Modules = [MHDETrees]
Private = false
```
