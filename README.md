# MHDETrees.jl

[![DOI](https://zenodo.org/badge/1323497859.svg)](https://doi.org/10.5281/zenodo.21854170)

[MHDETrees.jl](https://platform.juliahub.com/ui/Packages/General/MHDETrees)
trains fixed-depth classification and regression trees with differential
evolution. It provides Moving-Horizon Differential Evolution (MH-DEOCT),
full-tree DEOCT, and a CPU regression CART baseline. DEOCT and MH-DEOCT support
both CPU and NVIDIA GPU training.

MH-DEOCT constructs a deep tree through a sequence of smaller subtree
optimization problems instead of optimizing the complete tree at once. Its
discrete encoding maps evolutionary variables to features and data-derived
split candidates. The fitted model remains a standard decision tree with
inspectable split rules. Classification leaves predict majority classes;
regression leaves predict either target means or affine functions.

For the formulation and algorithm, see
[*A Moving-Horizon Differential Evolution Algorithm for Training Deep
Classification Trees on Large Datasets*](https://doi.org/10.1016/j.eswa.2026.133846).

## Installation

MHDETrees.jl supports Julia 1.8 and later. Install it from the Julia General
registry:

```julia
using Pkg
Pkg.add("MHDETrees")
```

Then load the package:

```julia
using MHDETrees
```

An NVIDIA GPU is optional. With `backend=:auto`, MHDETrees.jl uses CUDA when a
functional NVIDIA device is available and otherwise uses the CPU. Use
`backend=:cpu` or `backend=:gpu` to select a backend explicitly.

## Classification quick start

The public API accepts a numeric feature matrix `X` with one sample per row and
a label or target vector `y`. This example trains a classifier on the bundled
Iris dataset:

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
predictions = predict(model, X[test, :])
score = accuracy(model, X[test, :], y[test])

println(model)
println("test accuracy = ", score)
```

Features are min-max scaled by default, and the fitted scaling parameters are
stored in the model. Prediction runs on the CPU regardless of the training
backend.

## Regression quick start

Set `task=:regression` to minimize within-leaf squared error. Mean leaves are
the default. Set `leaf_model=:linear` to fit an independent affine model of all
features in each leaf. This example uses the bundled
[UCI Yacht Hydrodynamics](https://doi.org/10.24432/C5XG7R) dataset:

```julia
using MHDETrees
using Random

X, y = load_yacht()
indices = randperm(MersenneTwister(7), length(y))
train = indices[1:246]
test = indices[247:end]

config = MHDEOCTConfig(
    task=:regression,
    leaf_model=:linear,
    algorithm=:mhdeoct,
    backend=:cpu,
    depth=3,
    horizon=2,
    population_size=20,
    generations=10,
    seed=7,
)

model = fit(X[train, :], y[train]; config)
predictions = predict(model, X[test, :])
rmse = root_mean_squared_error(model, X[test, :], y[test])
r2 = r2_score(model, X[test, :], y[test])

println(model)
println("test RMSE = ", rmse)
println("test R2 = ", r2)
```

Both regression leaf models use the same mean-SSE tree-structure objective.
Affine leaves are fitted after the structure search using ridge-stabilized
least squares on the internally scaled features. Empty leaves use the global
mean or global affine model as a fallback.

## Algorithms and backends

| Task and algorithm | CPU | NVIDIA GPU |
|---|---:|---:|
| Classification DEOCT | yes | yes |
| Classification MH-DEOCT | yes | yes |
| Regression CART | yes | no |
| Regression DEOCT | yes | yes |
| Regression MH-DEOCT | yes | yes |

- `algorithm=:mhdeoct` performs moving-horizon optimization and is the default.
- `algorithm=:deoct` optimizes the complete fixed-depth tree.
- `algorithm=:cart` selects the greedy regression CART baseline.
- `horizon` controls the local subtree depth for MH-DEOCT and is ignored by
  full-tree DEOCT and CART.
- `population_size` and `generations` control the differential-evolution search
  effort.

## Configuration

| Option | Default | Description |
|---|---:|---|
| `task` | `:classification` | Learning task: `:classification` or `:regression`. |
| `algorithm` | `:mhdeoct` | Training algorithm: `:mhdeoct`, `:deoct`, or regression-only `:cart`. |
| `backend` | `:auto` | Use CUDA when functional, otherwise use the CPU. |
| `leaf_model` | `:mean` | Regression leaf model: `:mean` or `:linear`. |
| `linear_regularization` | `1e-2` | Ridge regularization for affine regression leaves. |
| `depth` | `2` | Maximum tree depth. |
| `horizon` | `2` | Local subtree depth for MH-DEOCT. |
| `population_size` | `100` | Differential-evolution population size. |
| `generations` | `600` | Number of differential-evolution generations. |
| `min_samples_leaf` | `1` | Minimum requested samples in a nonempty leaf. |
| `alpha` | `0.0` | Penalty per active branch node. |
| `initialization` | `:cart` | Initialization strategy: `:cart`, `:de`, or `:none`. |
| `seed` | `1` | Random seed. |
| `scale_features` | `true` | Min-max scale features before training. |
| `verbose` | `false` | Print training diagnostics. |

The small search settings in the examples are intended for quick tests. Larger
populations and generation counts may be appropriate for real applications.

## NVIDIA GPU training

The GPU backend requires a CUDA-capable NVIDIA GPU and a driver supported by
CUDA.jl. CUDA.jl is installed automatically as an MHDETrees.jl dependency.
Check device availability before forcing GPU training:

```julia
using MHDETrees

println("gpu_available() = ", gpu_available())

config = MHDEOCTConfig(
    task=:regression,
    algorithm=:mhdeoct,
    backend=:gpu,
    depth=4,
    horizon=3,
)
```

Selecting `backend=:gpu` never silently falls back to the CPU. GPU diagnostics
are quiet by default; set `verbose=true` to print training details.

Regression GPU fitness uses a persistent device context. Scaled features,
targets, global split tables, and reusable workspaces remain on the GPU across
moving-horizon nodes. Candidates use direct feature indices, and fitness is
evaluated over a two-dimensional sample-group-by-candidate grid. Leaf counts
use `Int32`; target sums, squared sums, thresholds, costs, and differential-
evolution state use `Float32`. SSE evaluation, penalties, population
replacement, and best-candidate selection run on the GPU. The selected tree is
then copied to the CPU for final leaf fitting and prediction.

For direct CUDA troubleshooting, add CUDA.jl to the active environment and
inspect its configuration:

```julia
using Pkg
Pkg.add("CUDA")
using CUDA
CUDA.versioninfo()
```

## Evaluation

Use `accuracy(model, X, y)` for classification. Regression models support:

- `mean_squared_error(model, X, y)`
- `root_mean_squared_error(model, X, y)`
- `r2_score(model, X, y)`

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
