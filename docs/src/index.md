# MHDETrees.jl

MHDETrees.jl implements full-tree and moving-horizon differential evolution for
optimal classification trees. Both algorithms support CPU and CUDA backends.

| Algorithm | CPU | GPU |
|---|---:|---:|
| `algorithm=:deoct` | supported | supported |
| `algorithm=:mhdeoct` | supported | supported |

```@contents
```

## Quick start

```julia
using MHDETrees

config = MHDEOCTConfig(
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

The GPU backend requires an NVIDIA GPU visible to CUDA.jl. The CPU backend and
prediction do not require a GPU. `horizon` applies to `algorithm=:mhdeoct`;
full-tree DEOCT ignores it. Set `verbose=true` to print GPU kernel, timing, and
optimization diagnostics; the default is quiet.

## Public API

```@autodocs
Modules = [MHDETrees]
Private = false
```
