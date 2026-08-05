@testset "MHDETrees public API" begin
    config = MHDEOCTConfig(depth=1, horizon=1, generations=1, population_size=4)
    model = MHDEOCTModel(
        [1.0, 0.5],
        [1, 2],
        ["left", "right"],
        [[0.5]],
        [0.0],
        [1.0],
        config,
        :mhdeoct,
        :cpu,
        0,
        0,
    )

    X = reshape([0.2, 0.8], :, 1)
    y = ["left", "right"]
    @test predict(model, X) == y
    @test accuracy(model, X, y) == 1.0
    @test occursin("MHDEOCTModel", sprint(show, model))
    @test occursin("algorithm=mhdeoct", sprint(show, model))
    @test !config.verbose
    @test gpu_available() isa Bool

    @test_throws DimensionMismatch predict(model, zeros(2, 2))
    @test_throws DimensionMismatch accuracy(model, X, ["left"])
    @test_throws ArgumentError fit(
        X,
        y;
        config=MHDEOCTConfig(
            algorithm=:invalid,
            backend=:cpu,
            depth=1,
            horizon=1,
            generations=0,
            population_size=4,
        ),
    )
end
