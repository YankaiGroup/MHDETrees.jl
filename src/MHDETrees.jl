"""
    MHDETrees

Full-tree and moving-horizon differential evolution for training optimal
classification trees on CPUs and NVIDIA GPUs.
"""
module MHDETrees

include("cart/DecisionTree_modified.jl")

include("gpu/oct_gpu.jl")
include("gpu/warmstart_gpu.jl")
include("gpu/de_gpu.jl")

include("cpu_backend.jl")
include("api.jl")

export MHDEOCTConfig,
       MHDEOCTModel,
       accuracy,
       fit,
       gpu_available,
       load_iris,
       predict

end # module MHDETrees
