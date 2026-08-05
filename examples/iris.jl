using MHDETrees

X, y = load_iris()
train = vcat(1:40, 51:90, 101:140)
test = vcat(41:50, 91:100, 141:150)

for algorithm in (:deoct, :mhdeoct)
    for backend in (:cpu, :gpu)
        backend == :gpu && !gpu_available() && continue
        config = MHDEOCTConfig(
            algorithm=algorithm,
            backend=backend,
            depth=2,
            horizon=2,
            population_size=10,
            generations=2,
            seed=1,
        )
        model = fit(X[train, :], y[train]; config)
        println(model)
        println(
            "$(algorithm)/$(backend) Iris test accuracy = ",
            accuracy(model, X[test, :], y[test]),
        )
    end
end
