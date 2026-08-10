"""
    MHDETrees

Full-tree and moving-horizon differential evolution for training optimal
classification and regression trees on CPUs and NVIDIA GPUs.
"""
module MHDETrees

include("cart/DecisionTree_modified.jl")

include("gpu/oct_gpu.jl")
include("gpu/warmstart_gpu.jl")
include("gpu/de_gpu.jl")

include("cpu_backend.jl")
include("regression_backend.jl")
include("api.jl")

export MHDEOCTConfig,
       MHDEOCTModel,
       MHDEOCTRegressorModel,
       accuracy,
       fit,
       gpu_available,
       load_iris,
       load_yacht,
       mean_squared_error,
       predict,
       r2_score,
       root_mean_squared_error

end # module MHDETrees
