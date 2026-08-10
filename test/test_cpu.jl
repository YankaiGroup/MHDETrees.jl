@testset "Iris CPU backends" begin
    X, y = load_iris()
    train = vcat(1:40, 51:90, 101:140)
    test = vcat(41:50, 91:100, 141:150)

    for algorithm in (:deoct, :mhdeoct)
        config = MHDEOCTConfig(
            algorithm=algorithm,
            backend=:cpu,
            depth=2,
            horizon=2,
            population_size=10,
            generations=2,
            initialization=:cart,
            seed=1,
        )
        model = fit(X[train, :], y[train]; config)

        @test model.algorithm == algorithm
        @test model.backend == :cpu
        @test length(predict(model, X[test, :])) == length(test)
        @test accuracy(model, X[test, :], y[test]) >= 0.8
    end
end


@testset "Regression leaf means" begin
    X = reshape([0.1, 0.2, 0.8, 0.9], :, 1)
    y = [1.0, 3.0, 10.0, 14.0]
    config = MHDEOCTConfig(
        task=:regression,
        algorithm=:deoct,
        backend=:cpu,
        depth=1,
        horizon=1,
        population_size=4,
        generations=0,
        initialization=:cart,
        seed=1,
    )
    model = fit(X, y; config)

    @test model isa MHDEOCTRegressorModel
    @test sort(model.leaf_values[model.leaf_counts .> 0]) == [2.0, 12.0]
    @test predict(model, X) == [2.0, 2.0, 12.0, 12.0]
    @test model.leaf_coefficients == zeros(1, 2)
    @test model.training_sse == 10.0
    @test model.min_leaf_violations == 0
end

@testset "Affine regression leaves" begin
    X = [
        0.0 0.0
        1.0 0.0
        0.0 1.0
        1.0 1.0
        0.25 0.75
        0.8 0.2
    ]
    y = 4.0 .+ 3.0 .* X[:, 1] .- 2.0 .* X[:, 2]
    config = MHDEOCTConfig(
        task=:regression,
        leaf_model=:linear,
        linear_regularization=0.0,
        algorithm=:cart,
        backend=:cpu,
        depth=2,
        horizon=2,
        population_size=4,
        generations=0,
        initialization=:none,
        alpha=1.0e6,
        seed=1,
    )
    model = fit(X, y; config)

    @test model.config.leaf_model == :linear
    @test model.leaf_counts == [0, 0, 0, length(y)]
    @test predict(model, X) ≈ y atol=1.0e-10
    @test model.training_sse < 1.0e-20
    @test all(isfinite, model.leaf_coefficients)
    @test model.leaf_values ≈ fill(4.0, 4) atol=1.0e-10
    @test model.leaf_coefficients ≈ repeat([3.0, -2.0], 1, 4) atol=1.0e-10
end

@testset "Yacht regression CPU backends" begin
    X, y = load_yacht()
    @test size(X) == (308, 6)
    @test length(y) == 308
    @test all(isfinite, X)
    @test all(isfinite, y)
    test = collect(5:5:length(y))
    train = setdiff(eachindex(y), test)

    for leaf_model in (:mean, :linear), algorithm in (:cart, :deoct, :mhdeoct)
        config = MHDEOCTConfig(
            task=:regression,
            leaf_model=leaf_model,
            algorithm=algorithm,
            backend=:cpu,
            depth=2,
            horizon=2,
            population_size=10,
            generations=2,
            initialization=:cart,
            seed=1,
        )
        model = fit(X[train, :], y[train]; config)

        @test model.algorithm == algorithm
        @test model.backend == :cpu
        @test model.config.leaf_model == leaf_model
        @test length(predict(model, X[test, :])) == length(test)
        @test all(isfinite, predict(model, X[test, :]))
        @test root_mean_squared_error(model, X[test, :], y[test]) < 10.0
        @test r2_score(model, X[test, :], y[test]) > 0.5
    end
end

@testset "Full-tree regression CPU depth matrix" begin
    X, y = load_yacht()
    train = 1:96
    test = 97:112

    for depth in (2, 4, 6, 8)
        config = MHDEOCTConfig(
            task=:regression,
            algorithm=:deoct,
            backend=:cpu,
            depth=depth,
            horizon=min(3, depth),
            population_size=4,
            generations=0,
            min_samples_leaf=4,
            initialization=:none,
            seed=31,
        )
        model = fit(X[train, :], y[train]; config)
        predictions = predict(model, X[test, :])

        @test model.algorithm == :deoct
        @test model.backend == :cpu
        @test model.config.depth == depth
        @test length(predictions) == length(test)
        @test all(isfinite, predictions)
        @test isfinite(model.training_sse)
    end
end
