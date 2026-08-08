# MHDETrees.jl

[![DOI](https://zenodo.org/badge/1323497859.svg)](https://doi.org/10.5281/zenodo.21854170)

[MHDETrees.jl](https://platform.juliahub.com/ui/Packages/General/MHDETrees) trains
fixed-depth classification trees with differential evolution. It provides the
Moving-Horizon Differential Evolution algorithm for Optimal Classification
Trees (MH-DEOCT) and a full-tree DEOCT baseline. Both algorithms can train on a
CPU or an NVIDIA GPU.

MH-DEOCT builds a deep tree through a sequence of smaller subtree optimization
problems instead of optimizing the entire tree at once. Its discrete decoding
maps evolutionary variables to features and data-derived split candidates, so
equivalent continuous thresholds are not searched repeatedly. The result is a
standard classification tree with inspectable split rules.

For the formulation, algorithm, and computational experiments, see
[*A Moving-Horizon Differential Evolution Algorithm for Training Deep
Classification Trees on Large Datasets*](https://doi.org/10.1016/j.eswa.2026.133846).

## Installation

MHDETrees.jl supports Julia 1.8 and later. Install the registered package from
the Julia General registry:

```julia
using Pkg
Pkg.add("MHDETrees")
```

Then load it with:

```julia
using MHDETrees
```

An NVIDIA GPU is not required to install or use MHDETrees.jl. CUDA.jl is an
internal package dependency and is installed automatically, so users do not
need to add or import CUDA.jl explicitly to use the GPU backend. On a computer
without a functional NVIDIA CUDA device (including a Mac), the default
`backend=:auto` uses the CPU. On a computer with a functional CUDA device, it
uses the GPU. You can always select a backend explicitly with `backend=:cpu` or
`backend=:gpu`.

## Quick start

The public API accepts a numeric feature matrix `X` with one sample per row and
a label vector `y`. This example uses the bundled Iris dataset and explicitly
selects CPU training:

```julia
using MHDETrees

X, y = load_iris()
train = vcat(1:40, 51:90, 101:140)
test = vcat(41:50, 91:100, 141:150)

config = MHDEOCTConfig(
    algorithm=:mhdeoct,
    backend=:cpu,
    depth=2,
    horizon=2,
    population_size=20,
    generations=20,
    seed=1,
)

model = fit(X[train, :], y[train]; config)
y_pred = predict(model, X[test, :])
score = accuracy(model, X[test, :], y[test])

println(model)
println("test accuracy = ", score)
```

Features are min-max scaled by default, and the fitted scaling parameters are
stored in the model. Prediction uses the resulting tree on the CPU regardless
of which backend was used for training.

## Choosing the algorithm and backend

| Configuration | Meaning |
|---|---|
| `algorithm=:mhdeoct` | Moving-horizon optimization; this is the default. |
| `algorithm=:deoct` | Full-tree differential evolution. |
| `backend=:auto` | Use CUDA when functional; otherwise use the CPU. This is the default. |
| `backend=:cpu` | Force CPU training. |
| `backend=:gpu` | Force NVIDIA GPU training; error if CUDA is unavailable. |

All four explicit algorithm/backend combinations are supported:

| Algorithm | CPU | NVIDIA GPU |
|---|---:|---:|
| Full-tree DEOCT (`algorithm=:deoct`) | yes | yes |
| MH-DEOCT (`algorithm=:mhdeoct`) | yes | yes |

`horizon` controls the optimized subtree depth for MH-DEOCT and is ignored by
full-tree DEOCT. `population_size` and `generations` control the evolutionary
search effort. The small values in the example are intended for a quick test;
larger optimization budgets should generally be considered for real datasets.

## NVIDIA GPU training

The GPU backend requires a CUDA-capable NVIDIA GPU and a driver supported by
CUDA.jl. CUDA.jl is installed automatically as an MHDETrees.jl dependency; it
does not need to be added or imported explicitly for GPU training:

```julia
using MHDETrees

gpu = gpu_available()
println("gpu_available() = ", gpu)
@assert gpu

X, y = load_iris()
train = vcat(1:40, 51:90, 101:140)
test = vcat(41:50, 91:100, 141:150)

config = MHDEOCTConfig(
    algorithm=:mhdeoct,  # or :deoct
    backend=:gpu,
    depth=2,
    horizon=2,
    population_size=20,
    generations=20,
    seed=1,
    verbose=false,
)

model = fit(X[train, :], y[train]; config)
y_pred = predict(model, X[test, :])
score = accuracy(model, X[test, :], y[test])

println(model)
println("test accuracy = ", score)
```

GPU diagnostics are quiet by default. Set `verbose=true` to print kernel
configuration, timing, and optimization progress. Selecting `backend=:gpu`
never silently falls back to the CPU.

To inspect CUDA.jl directly for troubleshooting, add and import it explicitly:

```julia
using Pkg
Pkg.add("CUDA")
using CUDA
CUDA.versioninfo()
```

This is optional and is not required for normal MHDETrees.jl GPU training.

## Main configuration options

| Option | Default | Description |
|---|---:|---|
| `depth` | `2` | Maximum tree depth. |
| `horizon` | `2` | Subtree depth for MH-DEOCT; must be between `1` and `depth`. |
| `population_size` | `100` | Differential-evolution population size. |
| `generations` | `600` | Number of differential-evolution generations. |
| `min_samples_leaf` | `1` | Minimum requested number of samples in a nonempty leaf. |
| `alpha` | `0.0` | Penalty per active branch node in the training objective. |
| `initialization` | `:cart` | Initialization strategy: `:cart`, `:de`, or `:none`. |
| `seed` | `1` | Random seed. |
| `scale_features` | `true` | Min-max scale features before training. |
| `verbose` | `false` | Print GPU training diagnostics. |

## Citation

If MHDETrees.jl contributes to your work, please cite the accompanying paper:

> Jiayang Ren, Valentín Osuna-Enciso, and Yankai Cao, “A Moving-Horizon
> Differential Evolution Algorithm for Training Deep Classification Trees on
> Large Datasets,” *Expert Systems with Applications*, 133846, 2026.
> [doi:10.1016/j.eswa.2026.133846](https://doi.org/10.1016/j.eswa.2026.133846)

Machine-readable citation metadata are provided in
[`CITATION.cff`](CITATION.cff).

## License

MHDETrees.jl is distributed under the [MIT License](LICENSE).
