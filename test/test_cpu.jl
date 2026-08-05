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
