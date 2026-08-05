@testset "Iris GPU backends" begin
    @test gpu_available()

    X, y = load_iris()
    train = vcat(1:40, 51:90, 101:140)
    test = vcat(41:50, 91:100, 141:150)

    for algorithm in (:deoct, :mhdeoct)
        verbose = algorithm == :mhdeoct
        config = MHDEOCTConfig(
            algorithm=algorithm,
            backend=:gpu,
            depth=2,
            horizon=2,
            population_size=10,
            generations=1,
            initialization=:cart,
            seed=1,
            verbose=verbose,
        )
        model, diagnostics = mktemp() do _, output
            model = redirect_stdout(output) do
                fit(X[train, :], y[train]; config)
            end
            flush(output)
            seekstart(output)
            return model, read(output, String)
        end

        @test model.algorithm == algorithm
        @test model.backend == :gpu
        if verbose
            @test occursin("gpu_init time:", diagnostics)
            @test occursin("fitnesses:", diagnostics)
        else
            @test isempty(strip(diagnostics))
        end
        @test length(predict(model, X[test, :])) == length(test)
        @test accuracy(model, X[test, :], y[test]) >= 0.8
    end
end
