using CUDA
using Random

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

@testset "Shared-memory classification leaf counts" begin
    oct_gpu = MHDETrees.oct_gpu
    kernels, threads = oct_gpu.gpu_init(false)
    Random.seed!(7)
    for (n, p, classes, depth, population) in ((5_000, 6, 3, 3, 10), (777, 4, 7, 2, 20))
        size_branch, size_leaves = 2^depth - 1, 2^depth
        X = CuArray(rand(Float32, n, p))
        Y = CuArray(rand(Int32(1):Int32(classes), n))
        a = falses(p, size_branch * population)
        for j in axes(a, 2)
            rand() < 0.9 && (a[rand(1:p), j] = true)
        end
        d = sum(a, dims=1) .> 0
        d[rand(1:length(d), 3)] .= false
        a_d = CuArray(a)
        b_d = CuArray(rand(Float32, 1, size_branch * population))
        d_d = CuArray(d)

        stride = 7
        block_threads = min(n, threads[1])
        rows = cld(cld(n, stride), block_threads) * block_threads
        per_thread = CUDA.zeros(Float32, rows, size_leaves * population, classes)
        oct_gpu.leaf_counts!(per_thread, kernels, threads, X, a_d, b_d, d_d, population, Y, stride)
        shared = CUDA.zeros(Float32, 1, size_leaves * population, classes)
        launch = oct_gpu.shared_count_launch(kernels[3], n, length(shared))
        oct_gpu.leaf_counts!(shared, kernels, threads, X, a_d, b_d, d_d, population, Y, launch)
        @test Array(shared) == Array(sum(per_thread, dims=1))
        @test sum(Array(shared)) == n * population
    end
end

same_cart_tree(a::MHDETrees.DecisionTree_modified.Leaf, b::MHDETrees.DecisionTree_modified.Leaf) =
    a.majority == b.majority
same_cart_tree(a::MHDETrees.DecisionTree_modified.Node, b::MHDETrees.DecisionTree_modified.Node) =
    a.featid == b.featid && a.featval == b.featval &&
    same_cart_tree(a.left, b.left) && same_cart_tree(a.right, b.right)
same_cart_tree(a, b) = false

@testset "GPU CART matches the CPU CART" begin
    X_iris, y_iris = load_iris()
    Random.seed!(11)
    X_ties = round.(rand(2_000, 5); digits=1)
    y_ties = rand(1:3, 2_000)
    for (X, y, depth, min_samples_leaf, seed) in (
        (X_iris, y_iris, 3, 1, 1),
        (X_iris, y_iris, 8, 5, 2),
        (X_ties, y_ties, 4, 1, 3),
        (X_ties, y_ties, 6, 20, 4),
    )
        Random.seed!(seed)
        cpu = MHDETrees.DecisionTree_modified.build_tree(y, X, 0, depth, min_samples_leaf)
        cpu_next = rand(UInt64)
        Random.seed!(seed)
        gpu = MHDETrees.cart_gpu.build_tree_gpu(y, X, depth, min_samples_leaf)
        @test same_cart_tree(cpu.node, gpu.node)
        @test rand(UInt64) == cpu_next # same draws from the global RNG
    end
end

@testset "GPU split_x matches the CPU split_x" begin
    Random.seed!(12)
    X = round.(rand(3_000, 4); digits=2)
    X[1, 1] = -0.0 # unique() keeps -0.0 and 0.0 apart
    X[2, 1] = 0.0
    cpu = MHDETrees.oct_gpu.split_x(X)
    gpu = MHDETrees.oct_gpu.split_x(MHDETrees.cart_gpu.SortedColumns(CuArray(X)))
    @test all(isequal.(cpu[1], gpu[1]))
    @test all(isequal.(cpu[2], gpu[2]))
    @test typeof.(gpu[2]) == typeof.(cpu[2])
end

@testset "GPU fits are unchanged by shared-memory counts, the GPU CART and GPU node data" begin
    X, y = load_iris()
    function set_gpu_options(shared, gpu_cart, gpu_node_data)
        MHDETrees.oct_gpu.SHARED_COUNTS[] = shared
        MHDETrees.warmstart_gpu.GPU_CART[] = gpu_cart
        MHDETrees.de_gpu.GPU_NODE_DATA[] = gpu_node_data
    end
    for algorithm in (:deoct, :mhdeoct)
        config = MHDEOCTConfig(
            algorithm=algorithm,
            backend=:gpu,
            depth=3,
            horizon=2,
            population_size=10,
            generations=3,
            initialization=:cart,
            seed=1,
        )
        models = try
            options = ((false, false, false), (true, false, false), (true, true, false),
                       (true, false, true), (true, true, true))
            map(options) do option
                set_gpu_options(option...)
                fit(X, y; config)
            end
        finally
            set_gpu_options(true, true, true)
        end
        for model in models[2:end]
            @test model.candidate == models[1].candidate
            @test model.training_errors == models[1].training_errors
        end
    end
end

@testset "Float32 regression GPU fitness parity" begin
    X, y = load_yacht()
    X = X[1:64, :]
    y = y[1:64]
    config = MHDEOCTConfig(
        task=:regression,
        algorithm=:deoct,
        backend=:gpu,
        depth=2,
        horizon=2,
        population_size=8,
        generations=0,
        min_samples_leaf=10,
        alpha=0.25,
        initialization=:none,
        seed=19,
    )
    X_scaled, feature_min, feature_scale =
        MHDETrees._prepare_features(X, config.scale_features)
    context = MHDETrees._regression_gpu_context(X_scaled, y, config)
    workspace = MHDETrees.oct_gpu.regression_gpu_workspace(
        length(y),
        config.depth,
        config.population_size,
        size(X, 2);
        maximum_buffer_bytes=6_144,
    )
    context.workspaces[config.depth] = workspace
    @test MHDETrees._regression_gpu_workspace(context, config.depth) === workspace
    @test workspace.population_stride == 2
    @test eltype(workspace.partial_counts) == Int32
    @test eltype(workspace.leaf_counts) == Int32
    @test eltype(workspace.partial_moments) == Float32
    @test eltype(workspace.leaf_moments) == Float32
    @test eltype(workspace.thresholds) == Float32
    @test eltype(workspace.population) == Float32
    @test eltype(workspace.costs) == Float32
    @test eltype(context.y_device) == Float32
    @test eltype(context.split_values_device) == Float32

    branch_count = 2^config.depth - 1
    population = MHDETrees._random_cpu_population(
        MersenneTwister(config.seed),
        config.population_size,
        branch_count,
        size(X, 2),
    )
    population[1, 2:branch_count] .= 0.0
    copyto!(workspace.population, Float32.(population))
    splits, _ = MHDETrees.oct_gpu.split_x(X_scaled)
    split_values, split_offsets, split_lengths =
        MHDETrees.oct_gpu.regression_gpu_split_arrays(splits)
    lower_indices, upper_indices =
        MHDETrees._regression_local_split_bounds(X_scaled, splits)
    target_offset, total_sse_float32 = MHDETrees._float32_regression_scale(y)
    @test target_offset isa Float32
    @test target_offset == sum(Float32, y) / Float32(length(y))
    @test total_sse_float32 isa Float32
    MHDETrees.oct_gpu.regression_gpu_fitness!(
        workspace,
        workspace.population,
        workspace.costs,
        context.X_device,
        context.y_device,
        context.all_rows_device,
        split_values,
        split_offsets,
        split_lengths,
        CuArray(lower_indices),
        CuArray(upper_indices),
        target_offset,
        total_sse_float32,
        config.min_samples_leaf,
        config.alpha,
    )
    gpu_fitness = Float64.(Array(workspace.costs))
    leaf_count = 2^config.depth
    gpu_counts = reshape(Array(workspace.leaf_counts), leaf_count, :)
    gpu_moments = Array(workspace.leaf_moments)

    centered_y = y .- sum(y) / length(y)
    total_sse = sum(abs2, centered_y)
    cpu_fitness = [
        MHDETrees._regression_cpu_fitness(
            view(population, candidate, :),
            X_scaled,
            centered_y,
            config,
            config.depth,
            splits,
            total_sse,
        ) for candidate in 1:config.population_size
    ]
    @test gpu_fitness ≈ cpu_fitness rtol=2.0e-5 atol=2.0e-2
    for candidate in 1:config.population_size
        counts, sums, squared_sums, _ = MHDETrees._regression_leaf_moments(
            view(population, candidate, :),
            X_scaled,
            centered_y,
            config.depth,
            splits,
        )
        positions = ((candidate - 1) * leaf_count + 1):(candidate * leaf_count)
        @test gpu_counts[:, candidate] == counts
        @test Float64.(gpu_moments[1, positions]) ≈ sums rtol=2.0e-5 atol=2.0e-2
        @test Float64.(gpu_moments[2, positions]) ≈ squared_sums rtol=2.0e-5 atol=2.0e-2
    end

    fixed_result = MHDETrees._finish_regression_result(
        view(population, 1, :),
        X_scaled,
        y,
        config,
        splits,
    )
    function fixed_model(backend)
        return MHDEOCTRegressorModel(
            fixed_result.candidate,
            fixed_result.leaf_values,
            fixed_result.leaf_coefficients,
            fixed_result.leaf_counts,
            fixed_result.splits,
            feature_min,
            feature_scale,
            config,
            :deoct,
            backend,
            fixed_result.training_sse,
            fixed_result.min_leaf_violations,
        )
    end
    @test predict(fixed_model(:cpu), X) == predict(fixed_model(:gpu), X)
end

@testset "Regression GPU moving-horizon row routing" begin
    X, y = load_yacht()
    X = X[1:96, :]
    y = y[1:96]
    config = MHDEOCTConfig(
        task=:regression,
        algorithm=:mhdeoct,
        backend=:gpu,
        depth=4,
        horizon=3,
        population_size=4,
        generations=0,
        initialization=:cart,
        seed=23,
    )
    X_scaled, _, _ = MHDETrees._prepare_features(X, config.scale_features)
    context = MHDETrees._regression_gpu_context(X_scaled, y, config)
    candidate = MHDETrees._regression_cart_candidate(
        X_scaled,
        y,
        config,
        config.depth,
        context.global_splits,
    )

    for level in 0:(config.depth - 1)
        nodes = collect(2^level:min(2^(level + 1) - 1, 2^config.depth - 1))
        gpu_rows = MHDETrees._regression_gpu_level_rows(
            context,
            X_scaled,
            candidate,
            context.global_splits,
            level,
            nodes,
        )
        for (local_node, node) in pairs(nodes)
            cpu_rows = findall(MHDETrees._cpu_rows_at_node(
                X_scaled,
                node,
                candidate,
                context.global_splits,
            ))
            @test gpu_rows[local_node] == cpu_rows
        end
    end
end


@testset "Yacht regression GPU backends" begin
    @test gpu_available()

    X, y = load_yacht()
    test = collect(5:5:length(y))
    train = setdiff(eachindex(y), test)

    for leaf_model in (:mean, :linear), algorithm in (:deoct, :mhdeoct)
        config = MHDEOCTConfig(
            task=:regression,
            leaf_model=leaf_model,
            algorithm=algorithm,
            backend=:gpu,
            depth=2,
            horizon=2,
            population_size=10,
            generations=1,
            initialization=:cart,
            seed=1,
        )
        model = fit(X[train, :], y[train]; config)

        @test model isa MHDEOCTRegressorModel
        @test model.algorithm == algorithm
        @test model.backend == :gpu
        @test model.config.leaf_model == leaf_model
        @test length(predict(model, X[test, :])) == length(test)
        @test all(isfinite, predict(model, X[test, :]))
        @test root_mean_squared_error(model, X[test, :], y[test]) < 10.0
        @test r2_score(model, X[test, :], y[test]) > 0.5
    end
end

@testset "Full-tree regression GPU depth matrix" begin
    X, y = load_yacht()
    train = 1:96
    test = 97:112

    for depth in (2, 4, 6, 8)
        config = MHDEOCTConfig(
            task=:regression,
            algorithm=:deoct,
            backend=:gpu,
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
        @test model.backend == :gpu
        @test model.config.depth == depth
        @test length(predictions) == length(test)
        @test all(isfinite, predictions)
        @test isfinite(model.training_sse)
    end
end

@testset "MH regression GPU depth and warm-start matrix" begin
    X, y = load_yacht()
    train = 1:128
    test = 129:160

    for depth in (2, 4, 6, 8), initialization in (:none, :cart)
        config = MHDEOCTConfig(
            task=:regression,
            algorithm=:mhdeoct,
            backend=:gpu,
            depth=depth,
            horizon=min(3, depth),
            population_size=4,
            generations=0,
            min_samples_leaf=8,
            initialization=initialization,
            seed=5,
        )
        model = fit(X[train, :], y[train]; config)
        predictions = predict(model, X[test, :])

        @test model.backend == :gpu
        @test model.config.depth == depth
        @test model.config.horizon == min(3, depth)
        @test length(predictions) == length(test)
        @test all(isfinite, predictions)
        @test isfinite(model.training_sse)
        @test model.min_leaf_violations >= 0
        initialization == :cart && @test(model.min_leaf_violations == 0)
    end
end
