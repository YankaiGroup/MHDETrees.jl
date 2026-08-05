# MHDETrees.jl

MHDETrees.jl implements Moving-Horizon Differential Evolution for Optimal
Classification Trees (MH-DEOCT). It trains a deep classification tree through
a sequence of smaller subtree optimization problems. Its discrete tree-decoding
strategy searches data-derived feature and split choices, avoiding redundant
continuous threshold searches. The package also provides full-tree DEOCT as a
baseline, with CPU and NVIDIA GPU backends for both algorithms.

| Algorithm | `backend=:cpu` | `backend=:gpu` |
|---|---:|---:|
| Full-tree DEOCT (`algorithm=:deoct`) | supported | supported |
| Moving-horizon DEOCT (`algorithm=:mhdeoct`) | supported | supported |

For the algorithm, formulation, and computational experiments, see the paper
[*A Moving-Horizon Differential Evolution Algorithm for Training Deep
Classification Trees on Large Datasets*](https://doi.org/10.1016/j.eswa.2026.133846).

## Installation

MHDETrees.jl supports Julia 1.8 and later with CUDA.jl 4, 5, or 6. Install Julia
with Juliaup and instantiate this checkout:

```shell
juliaup add 1.12
julia +1.12 --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

After registration, installation will use:

```julia
using Pkg
Pkg.add("MHDETrees")
```

## Iris example

MHDETrees.jl bundles the small Iris dataset, so the example has no external dataset
dependency.

```julia
using MHDETrees

X, y = load_iris()
train = vcat(1:40, 51:90, 101:140)
test = vcat(41:50, 91:100, 141:150)

config = MHDEOCTConfig(
    algorithm=:mhdeoct,  # or :deoct
    backend=:cpu,        # or :gpu
    depth=2,
    horizon=2,
    population_size=10,
    generations=2,
    seed=1,
    verbose=false,
)

model = fit(X[train, :], y[train]; config)
predictions = predict(model, X[test, :])
test_accuracy = accuracy(model, X[test, :], y[test])
```

`algorithm=:mhdeoct` is the default. Use `backend=:auto` to select the GPU when
CUDA is functional and otherwise fall back to the CPU. The `horizon` parameter
controls subtree depth for MH-DEOCT and is not used by full-tree DEOCT.
Features are min-max scaled by default, and the fitted scaling parameters are
stored in `MHDEOCTModel`. Prediction runs on the CPU for models trained by
either backend. GPU training diagnostics are quiet by default; set
`verbose=true` to print kernel configuration, timing, and optimization progress.

The complete CPU/GPU example is in [`examples/iris.jl`](examples/iris.jl):

```shell
julia +1.12 --project=. examples/iris.jl
```

## Testing

Run the API and both Iris CPU algorithm tests on any machine:

```shell
julia +1.12 --project=. -e 'using Pkg; Pkg.test()'
```

On macOS, run the following commands in Terminal from the package directory:

```shell
juliaup add 1.12
julia +1.12 --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
MHDETREES_RUN_GPU_TESTS=false julia +1.12 --project=. -e 'using Pkg; Pkg.test()'
julia +1.12 --project=. examples/iris.jl
```

The macOS example automatically skips the CUDA configurations and runs
DEOCT/CPU and MH-DEOCT/CPU.

On Windows PowerShell with an NVIDIA GPU, first verify CUDA.jl and then enable
both GPU algorithm tests:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
$env:MHDETREES_RUN_GPU_TESTS = "true"
julia +1.12 --project=. -e 'using CUDA; CUDA.versioninfo(); @assert CUDA.functional()'
julia +1.12 --project=. -e 'using Pkg; Pkg.test()'
julia +1.12 --project=. examples/iris.jl
```

## Repository layout

- `src/MHDETrees.jl`: package entry point.
- `src/api.jl`: configuration, model, and public API.
- `src/cpu_backend.jl`: CPU full-tree and moving-horizon implementations.
- `src/gpu/`: CUDA full-tree and moving-horizon implementations.
- `src/cart/`: adapted DecisionTree.jl code used for CART warm starts.
- `test/`: API plus Iris CPU/GPU tests.
- `examples/iris.jl`: one end-to-end dataset example.
- `docs/`: Documenter.jl source.

The adaptation under `src/cart/` adds the misclassification-error objective
used at the deepest moving-horizon layer. Its upstream attribution and license,
along with the provenance of the bundled Iris data, are recorded in
[`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md).

## Documentation

Build the local documentation with:

```shell
julia +1.12 --project=docs -e \
  'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate(); include("docs/make.jl")'
```

## Citation

Please cite the accompanying paper when using this software:

> Jiayang Ren, Valentín Osuna-Enciso, and Yankai Cao, “A Moving-Horizon
> Differential Evolution Algorithm for Training Deep Classification Trees on
> Large Datasets,” *Expert Systems with Applications*, 133846, 2026.
> [doi:10.1016/j.eswa.2026.133846](https://doi.org/10.1016/j.eswa.2026.133846)

Machine-readable software and article metadata are provided in
[`CITATION.cff`](CITATION.cff).

## License

MHDETrees.jl is distributed under the [MIT License](LICENSE).
