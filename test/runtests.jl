using MHDETrees
using Aqua
using Test

@testset "Package quality" begin
    Aqua.test_all(MHDETrees)
end

include("test_api.jl")
include("test_cpu.jl")

if lowercase(get(ENV, "MHDETREES_RUN_GPU_TESTS", "false")) in ("1", "true", "yes")
    include("test_gpu.jl")
else
    @info "Skipping GPU integration tests; set MHDETREES_RUN_GPU_TESTS=true to enable them"
end
