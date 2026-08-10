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

    regression_config = MHDEOCTConfig(
        task=:regression,
        depth=1,
        horizon=1,
        generations=0,
        population_size=4,
    )
    regressor = MHDEOCTRegressorModel(
        [1.0, 0.5],
        [2.0, 10.0],
        [2, 2],
        [[0.5]],
        [0.0],
        [1.0],
        regression_config,
        :mhdeoct,
        :cpu,
        4.0,
        0,
    )
    regression_targets = [1.0, 11.0]
    @test predict(regressor, X) == [2.0, 10.0]
    @test mean_squared_error(regressor, X, regression_targets) == 1.0
    @test root_mean_squared_error(regressor, X, regression_targets) == 1.0
    @test r2_score(regressor, X, regression_targets) == 0.96
    @test occursin("MHDEOCTRegressorModel", sprint(show, regressor))
    @test occursin("leaf_model=mean", sprint(show, regressor))
    @test occursin("training_mse=1.0", sprint(show, regressor))
    @test regressor.leaf_coefficients == zeros(1, 2)
    @test_throws DimensionMismatch mean_squared_error(
        regressor,
        X,
        [1.0],
    )
    @test_throws ArgumentError fit(
        X,
        y;
        config=MHDEOCTConfig(
            task=:regression,
            backend=:cpu,
            depth=1,
            horizon=1,
            generations=0,
            population_size=4,
        ),
    )
    @test_throws ArgumentError fit(
        X,
        [1.0, NaN];
        config=MHDEOCTConfig(
            task=:regression,
            backend=:cpu,
            depth=1,
            horizon=1,
            generations=0,
            population_size=4,
        ),
    )
    @test_throws ArgumentError fit(
        X,
        regression_targets;
        config=MHDEOCTConfig(
            task=:invalid,
            backend=:cpu,
            depth=1,
            horizon=1,
            generations=0,
            population_size=4,
        ),
    )
    @test_throws ArgumentError fit(
        X,
        regression_targets;
        config=MHDEOCTConfig(
            task=:regression,
            algorithm=:cart,
            backend=:gpu,
            depth=1,
            horizon=1,
            generations=0,
            population_size=4,
        ),
    )
    @test_throws ArgumentError fit(
        X,
        regression_targets;
        config=MHDEOCTConfig(
            task=:regression,
            leaf_model=:invalid,
            backend=:cpu,
            depth=1,
            horizon=1,
            generations=0,
            population_size=4,
        ),
    )
    @test_throws ArgumentError fit(
        X,
        regression_targets;
        config=MHDEOCTConfig(
            task=:regression,
            leaf_model=:linear,
            linear_regularization=-1.0,
            backend=:cpu,
            depth=1,
            horizon=1,
            generations=0,
            population_size=4,
        ),
    )
    @test_throws ArgumentError fit(
        X,
        y;
        config=MHDEOCTConfig(
            task=:classification,
            leaf_model=:linear,
            backend=:cpu,
            depth=1,
            horizon=1,
            generations=0,
            population_size=4,
        ),
    )
end
