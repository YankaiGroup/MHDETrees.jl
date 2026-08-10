using CUDA
using LinearAlgebra
using Random

function _regression_leaf_moments(candidate, X, y, depth, splits)
    branch_count = 2^depth - 1
    leaf_count = 2^depth
    features, thresholds, active = oct_gpu.trans_params_azd(
        copy(candidate),
        size(X, 2),
        branch_count,
        splits,
    )
    counts = zeros(Int, leaf_count)
    sums = zeros(Float64, leaf_count)
    squared_sums = zeros(Float64, leaf_count)

    for sample in axes(X, 1)
        node = 1
        while node <= branch_count
            feature = findfirst(@view features[:, node])
            go_left =
                active[node] &&
                feature !== nothing &&
                X[sample, feature] < thresholds[node]
            node = go_left ? 2 * node : 2 * node + 1
        end
        leaf = node - branch_count
        target = y[sample]
        counts[leaf] += 1
        sums[leaf] += target
        squared_sums[leaf] += target * target
    end
    return counts, sums, squared_sums, sum(active)
end

function _regression_cpu_fitness(
    candidate,
    X,
    centered_y,
    config,
    depth,
    splits,
    total_sse,
)
    counts, sums, squared_sums, active_count = _regression_leaf_moments(
        candidate,
        X,
        centered_y,
        depth,
        splits,
    )
    sse = 0.0
    violations = 0
    for leaf in eachindex(counts)
        count = counts[leaf]
        if count > 0
            sse += max(0.0, squared_sums[leaf] - sums[leaf]^2 / count)
            violations += count < config.min_samples_leaf
        end
    end
    branch_count = 2^depth - 1
    invalid_leaf_penalty = total_sse + config.alpha * branch_count + 1.0
    return sse + config.alpha * active_count + invalid_leaf_penalty * violations
end

function _regression_cart_candidate(X, y, config, depth, splits)
    branch_count = 2^depth - 1
    candidate = zeros(Float64, 2 * branch_count)
    tolerance_scale = max(1.0, sum(abs2, y .- sum(y) / length(y)))

    function grow!(node, rows, level)
        if level >= depth || length(rows) < 2 * config.min_samples_leaf
            return
        end

        target_sum = sum(row -> y[row], rows)
        target_squared_sum = sum(row -> abs2(y[row]), rows)
        parent_sse = max(0.0, target_squared_sum - target_sum^2 / length(rows))
        best_cost = parent_sse
        best_feature = 0
        best_threshold = 0.0
        for feature in axes(X, 2)
            feature_values = X[rows, feature]
            order = sortperm(feature_values)
            left_sum = 0.0
            left_squared_sum = 0.0
            for position in 1:(length(rows) - 1)
                ordered_position = order[position]
                target = y[rows[ordered_position]]
                left_sum += target
                left_squared_sum += target * target
                left_count = position
                right_count = length(rows) - position
                left_count >= config.min_samples_leaf || continue
                right_count >= config.min_samples_leaf || continue

                current_value = feature_values[ordered_position]
                next_value = feature_values[order[position + 1]]
                current_value == next_value && continue
                right_sum = target_sum - left_sum
                right_squared_sum = target_squared_sum - left_squared_sum
                cost =
                    max(0.0, left_squared_sum - left_sum^2 / left_count) +
                    max(0.0, right_squared_sum - right_sum^2 / right_count)
                if cost < best_cost
                    best_cost = cost
                    best_feature = feature
                    best_threshold = (current_value + next_value) / 2
                end
            end
        end

        improvement = parent_sse - best_cost
        tolerance = 16 * eps(Float64) * tolerance_scale
        if best_feature == 0 || improvement <= config.alpha + tolerance
            return
        end

        candidate[node] = best_feature
        best_split = searchsortedfirst(splits[best_feature], best_threshold)
        candidate[branch_count + node] =
            (best_split - 1) / length(splits[best_feature])
        best_left = [row for row in rows if X[row, best_feature] < best_threshold]
        best_right = [row for row in rows if X[row, best_feature] >= best_threshold]
        grow!(2 * node, best_left, level + 1)
        grow!(2 * node + 1, best_right, level + 1)
        return
    end

    grow!(1, collect(axes(X, 1)), 0)
    return candidate
end

function _optimize_regression_cpu_tree(X, y, config, depth)
    rng = MersenneTwister(config.seed)
    branch_count = 2^depth - 1
    feature_count = size(X, 2)
    splits, sorted_features = oct_gpu.split_x(X)
    population = _random_cpu_population(
        rng,
        config.population_size,
        branch_count,
        feature_count,
    )

    use_cart = config.initialization == :cart ||
               (config.initialization == :de && config.warm_start)
    if use_cart
        population[1, :] .= _regression_cart_candidate(X, y, config, depth, splits)
    end

    centered_y = y .- sum(y) / length(y)
    total_sse = sum(abs2, centered_y)
    fitness = [
        _regression_cpu_fitness(
            view(population, index, :),
            X,
            centered_y,
            config,
            depth,
            splits,
            total_sse,
        ) for index in axes(population, 1)
    ]
    best_index = argmin(fitness)
    best_candidate = copy(view(population, best_index, :))
    best_fitness = fitness[best_index]

    for _ in 1:config.generations
        for index in axes(population, 1)
            first_parent = rand(rng, axes(population, 1))
            second_parent = rand(rng, axes(population, 1))
            mutation = best_candidate .+
                       rand(rng) .* (
                           view(population, first_parent, :) .-
                           view(population, second_parent, :)
                       )
            _repair_cpu_candidate!(mutation, branch_count, feature_count)

            crossover = rand(rng, length(best_candidate)) .< 0.1
            crossover[rand(rng, eachindex(crossover))] = true
            trial = copy(view(population, index, :))
            trial[crossover] .= mutation[crossover]
            _repair_cpu_candidate!(trial, branch_count, feature_count)

            trial_fitness = _regression_cpu_fitness(
                trial,
                X,
                centered_y,
                config,
                depth,
                splits,
                total_sse,
            )
            if trial_fitness <= fitness[index]
                population[index, :] .= trial
                fitness[index] = trial_fitness
                if trial_fitness < best_fitness
                    best_candidate = copy(trial)
                    best_fitness = trial_fitness
                end
            end
        end
    end

    return best_candidate, splits, sorted_features
end

mutable struct RegressionGPUFitContext
    X_device
    y_device
    all_rows_device
    selected_device
    global_splits
    global_sorted_features
    split_values_device
    split_offsets_device
    split_lengths_device
    workspaces
    maximum_sample_count::Int
    population_size::Int
    feature_count::Int
end

function _regression_gpu_context(X, y, config)
    CUDA.functional() ||
        error("The GPU backend requires a functional CUDA device; check CUDA.versioninfo()")
    sample_count = size(X, 1)
    maximum_level_nodes = max(1, 2^(config.depth - 1))
    global_splits, global_sorted_features = oct_gpu.split_x(X)
    split_values_device, split_offsets_device, split_lengths_device =
        oct_gpu.regression_gpu_split_arrays(global_splits)
    return RegressionGPUFitContext(
        CuArray(Float32.(X)),
        CuArray(Float32.(y)),
        CuArray(Int32.(collect(1:sample_count))),
        CUDA.zeros(Bool, sample_count, maximum_level_nodes),
        global_splits,
        global_sorted_features,
        split_values_device,
        split_offsets_device,
        split_lengths_device,
        Dict{Int,Any}(),
        sample_count,
        config.population_size,
        size(X, 2),
    )
end

function _regression_local_split_bounds(X, global_splits)
    feature_count = size(X, 2)
    lower_indices = Vector{Int32}(undef, feature_count)
    upper_indices = Vector{Int32}(undef, feature_count)
    for feature in 1:feature_count
        feature_values = @view X[:, feature]
        minimum_value = minimum(feature_values)
        maximum_value = maximum(feature_values)
        feature_splits = global_splits[feature]
        lower_index = clamp(
            searchsortedfirst(feature_splits, minimum_value),
            1,
            length(feature_splits),
        )
        upper_index = clamp(
            searchsortedlast(feature_splits, maximum_value),
            lower_index,
            length(feature_splits),
        )
        lower_indices[feature] = Int32(lower_index)
        upper_indices[feature] = Int32(upper_index)
    end
    return lower_indices, upper_indices
end

function _global_to_local_thresholds!(
    candidate,
    branch_count,
    global_splits,
    lower_indices,
    upper_indices,
)
    for node in 1:branch_count
        feature = Int(candidate[node])
        feature == 0 && continue
        global_count = length(global_splits[feature])
        encoded = clamp(candidate[branch_count + node], 0.0, 1.0)
        global_index = encoded >= 1.0 ?
                       global_count : floor(Int, encoded * global_count) + 1
        lower_index = Int(lower_indices[feature])
        upper_index = Int(upper_indices[feature])
        local_count = upper_index - lower_index + 1
        local_index = clamp(global_index, lower_index, upper_index)
        candidate[branch_count + node] =
            (local_index - lower_index) / local_count
    end
    return candidate
end

function _local_to_global_thresholds!(
    candidate,
    branch_count,
    global_splits,
    lower_indices,
    upper_indices,
)
    for node in 1:branch_count
        feature = Int(candidate[node])
        feature == 0 && continue
        lower_index = Int(lower_indices[feature])
        upper_index = Int(upper_indices[feature])
        local_count = upper_index - lower_index + 1
        encoded = clamp(candidate[branch_count + node], 0.0, 1.0)
        local_index = encoded >= 1.0 ?
                      local_count : floor(Int, encoded * local_count) + 1
        global_index = lower_index + local_index - 1
        global_count = length(global_splits[feature])
        candidate[branch_count + node] = (global_index - 1) / global_count
    end
    return candidate
end

function _regression_gpu_workspace(context, depth)
    return get!(context.workspaces, depth) do
        oct_gpu.regression_gpu_workspace(
            context.maximum_sample_count,
            depth,
            context.population_size,
            context.feature_count,
        )
    end
end

function _float32_regression_scale(y)
    target_offset = sum(Float32, y) / Float32(length(y))
    total_sse = 0.0f0
    for target in y
        centered_target = Float32(target) - target_offset
        total_sse += centered_target * centered_target
    end
    return target_offset, total_sse
end

function _regression_gpu_trial_randomness(rng, population_size, state_size)
    first_parents = Int32.(rand(rng, 1:population_size, population_size))
    second_parents = Int32.(rand(rng, 1:population_size, population_size))
    mutation_scales = Float32.(rand(rng, population_size))
    crossover = Matrix{Bool}(rand(rng, population_size, state_size) .< 0.1)
    for candidate in 1:population_size
        crossover[candidate, rand(rng, 1:state_size)] = true
    end
    return first_parents, second_parents, mutation_scales, crossover
end

function _regression_subtree_candidate(global_candidate, root, local_depth)
    global_branch_count = length(global_candidate) ÷ 2
    local_branch_count = 2^local_depth - 1
    local_candidate = zeros(Float64, 2 * local_branch_count)
    global_nodes = Vector{Int}(undef, local_branch_count)
    global_nodes[1] = root
    for local_node in 2:local_branch_count
        parent = global_nodes[fld(local_node, 2)]
        global_nodes[local_node] = iseven(local_node) ? 2 * parent : 2 * parent + 1
    end
    for local_node in 1:local_branch_count
        global_node = global_nodes[local_node]
        global_node <= global_branch_count || continue
        local_candidate[local_node] = global_candidate[global_node]
        local_candidate[local_branch_count + local_node] =
            global_candidate[global_branch_count + global_node]
    end
    return local_candidate
end

function _optimize_regression_gpu_tree(
    X,
    y,
    config,
    depth;
    context=nothing,
    row_indices_device=nothing,
    seed_candidate=nothing,
)
    local_context = context === nothing ?
                    _regression_gpu_context(X, y, config) : context
    local_rows_device = row_indices_device === nothing ?
                        local_context.all_rows_device : row_indices_device
    rng = MersenneTwister(config.seed)
    branch_count = 2^depth - 1
    feature_count = size(X, 2)
    splits = local_context.global_splits
    sorted_features = local_context.global_sorted_features
    lower_indices, upper_indices = _regression_local_split_bounds(X, splits)
    workspace = _regression_gpu_workspace(local_context, depth)
    population = _random_cpu_population(
        rng,
        config.population_size,
        branch_count,
        feature_count,
    )

    use_cart = config.initialization == :cart ||
               (config.initialization == :de && config.warm_start)
    if seed_candidate !== nothing
        length(seed_candidate) == 2 * branch_count || throw(DimensionMismatch(
            "GPU regression seed has length $(length(seed_candidate)); expected " *
            "$(2 * branch_count)",
        ))
        population[1, :] .= seed_candidate
        _global_to_local_thresholds!(
            view(population, 1, :),
            branch_count,
            splits,
            lower_indices,
            upper_indices,
        )
    elseif use_cart
        population[1, :] .= _regression_cart_candidate(X, y, config, depth, splits)
        _global_to_local_thresholds!(
            view(population, 1, :),
            branch_count,
            splits,
            lower_indices,
            upper_indices,
        )
    end
    copyto!(workspace.population, Float32.(population))

    lower_indices_device = CuArray(lower_indices)
    upper_indices_device = CuArray(upper_indices)
    target_offset, total_sse = _float32_regression_scale(y)
    oct_gpu.regression_gpu_fitness!(
        workspace,
        workspace.population,
        workspace.costs,
        local_context.X_device,
        local_context.y_device,
        local_rows_device,
        local_context.split_values_device,
        local_context.split_offsets_device,
        local_context.split_lengths_device,
        lower_indices_device,
        upper_indices_device,
        target_offset,
        total_sse,
        config.min_samples_leaf,
        config.alpha,
    )
    oct_gpu.regression_gpu_set_best!(workspace)

    for _ in 1:config.generations
        first_parents, second_parents, mutation_scales, crossover =
            _regression_gpu_trial_randomness(
                rng,
                config.population_size,
                workspace.state_size,
            )
        oct_gpu.regression_gpu_trial_population!(
            workspace,
            first_parents,
            second_parents,
            mutation_scales,
            crossover,
        )
        oct_gpu.regression_gpu_fitness!(
            workspace,
            workspace.trials,
            workspace.trial_costs,
            local_context.X_device,
            local_context.y_device,
            local_rows_device,
            local_context.split_values_device,
            local_context.split_offsets_device,
            local_context.split_lengths_device,
            lower_indices_device,
            upper_indices_device,
            target_offset,
            total_sse,
            config.min_samples_leaf,
            config.alpha,
        )
        oct_gpu.regression_gpu_select!(workspace)
    end

    best_index = Int(Array(workspace.best_index)[1])
    best_candidate = Float64.(Array(@view workspace.population[best_index, :]))
    _local_to_global_thresholds!(
        best_candidate,
        branch_count,
        splits,
        lower_indices,
        upper_indices,
    )
    config.verbose && println(
        "regression GPU Float32 workspace: depth=$(depth), " *
        "population_stride=$(workspace.population_stride), " *
        "partial_rows=$(workspace.maximum_partial_rows)",
    )
    return best_candidate, splits, sorted_features
end

function _fit_affine_model(X, y, regularization)
    feature_count = size(X, 2)
    target_mean = sum(y) / length(y)
    feature_means = vec(sum(X; dims=1)) / size(X, 1)
    if length(y) == 1 || feature_count == 0
        return target_mean, zeros(Float64, feature_count)
    end

    centered_X = Matrix{Float64}(X .- feature_means')
    centered_y = Float64.(y .- target_mean)
    gram = transpose(centered_X) * centered_X
    right_hand_side = transpose(centered_X) * centered_y
    scale = max(1.0, tr(gram) / feature_count)
    ridge = regularization * scale
    coefficients = if ridge > 0.0
        regularized_gram = Hermitian(
            gram + ridge * Matrix{Float64}(I, feature_count, feature_count),
        )
        cholesky(regularized_gram) \ right_hand_side
    else
        pinv(centered_X) * centered_y
    end
    intercept = target_mean - dot(feature_means, coefficients)
    return intercept, vec(coefficients)
end

function _fit_regression_leaf_models(X, y, assignments, counts, config)
    leaf_count = length(counts)
    feature_count = size(X, 2)
    fallback_intercept = sum(y) / length(y)
    fallback_coefficients = zeros(Float64, feature_count)
    if config.leaf_model == :linear
        fallback_intercept, fallback_coefficients = _fit_affine_model(
            X,
            y,
            config.linear_regularization,
        )
    end

    leaf_values = fill(fallback_intercept, leaf_count)
    leaf_coefficients = repeat(fallback_coefficients, 1, leaf_count)
    leaf_rows = [Int[] for _ in 1:leaf_count]
    for sample in eachindex(assignments)
        push!(leaf_rows[assignments[sample]], sample)
    end

    for leaf in 1:leaf_count
        rows = leaf_rows[leaf]
        isempty(rows) && continue
        if config.leaf_model == :mean
            leaf_values[leaf] = sum(@view y[rows]) / length(rows)
        else
            leaf_X = Matrix(@view X[rows, :])
            leaf_y = @view y[rows]
            intercept, coefficients = _fit_affine_model(
                leaf_X,
                leaf_y,
                config.linear_regularization,
            )
            leaf_values[leaf] = intercept
            leaf_coefficients[:, leaf] .= coefficients
        end
    end
    return leaf_values, leaf_coefficients
end

function _leaf_model_training_sse(X, y, assignments, leaf_values, leaf_coefficients)
    training_sse = 0.0
    for sample in axes(X, 1)
        leaf = assignments[sample]
        prediction = leaf_values[leaf] + dot(
            @view(X[sample, :]),
            @view(leaf_coefficients[:, leaf]),
        )
        training_sse += abs2(y[sample] - prediction)
    end
    return training_sse
end

function _finish_regression_result(candidate, X, y, config, splits)
    counts, _, _, _ = _regression_leaf_moments(
        candidate,
        X,
        y,
        config.depth,
        splits,
    )
    assignments = _regression_leaf_assignments(
        candidate,
        X,
        config.depth,
        splits,
    )
    leaf_values, leaf_coefficients = _fit_regression_leaf_models(
        X,
        y,
        assignments,
        counts,
        config,
    )
    training_sse = _leaf_model_training_sse(
        X,
        y,
        assignments,
        leaf_values,
        leaf_coefficients,
    )
    violations = count(
        count_value -> 0 < count_value < config.min_samples_leaf,
        counts,
    )
    return (
        candidate=vec(Float64.(candidate)),
        leaf_values=leaf_values,
        leaf_coefficients=leaf_coefficients,
        leaf_counts=counts,
        splits=[Float64.(split) for split in splits],
        training_sse=training_sse,
        min_leaf_violations=violations,
    )
end

function _regression_leaf_assignments(candidate, X, depth, splits)
    branch_count = 2^depth - 1
    features, thresholds, active = oct_gpu.trans_params_azd(
        copy(candidate),
        size(X, 2),
        branch_count,
        splits,
    )
    assignments = Vector{Int}(undef, size(X, 1))
    for sample in axes(X, 1)
        node = 1
        while node <= branch_count
            feature = findfirst(@view features[:, node])
            go_left =
                active[node] &&
                feature !== nothing &&
                X[sample, feature] < thresholds[node]
            node = go_left ? 2 * node : 2 * node + 1
        end
        assignments[sample] = node - branch_count
    end
    return assignments
end

function _fit_deoct_regression_cpu_backend(X, y, config)
    candidate, splits, _ =
        _optimize_regression_cpu_tree(X, y, config, config.depth)
    return _finish_regression_result(candidate, X, y, config, splits)
end

function _fit_cart_regression_cpu_backend(X, y, config)
    splits, _ = oct_gpu.split_x(X)
    candidate = _regression_cart_candidate(X, y, config, config.depth, splits)
    return _finish_regression_result(candidate, X, y, config, splits)
end

function _fit_deoct_regression_gpu_backend(X, y, config)
    candidate, splits, _ =
        _optimize_regression_gpu_tree(X, y, config, config.depth)
    return _finish_regression_result(candidate, X, y, config, splits)
end

function _regression_gpu_level_rows(
    context,
    X,
    candidate,
    splits,
    level,
    nodes,
)
    level == 0 && return [collect(axes(X, 1))]
    branch_count = length(candidate) ÷ 2
    features, thresholds, active = oct_gpu.trans_params_azd(
        copy(candidate),
        size(X, 2),
        branch_count,
        splits,
    )
    node_count = length(nodes)
    path_features = zeros(Int32, level, node_count)
    path_thresholds = zeros(Float32, level, node_count)
    path_directions = falses(level, node_count)

    for (local_node, node) in pairs(nodes)
        directions = Bool[]
        current = node
        while current > 1
            push!(directions, iseven(current))
            current = fld(current, 2)
        end
        reverse!(directions)

        current = 1
        for step in 1:level
            feature = findfirst(@view features[:, current])
            if active[current] && feature !== nothing
                path_features[step, local_node] = Int32(feature)
                path_thresholds[step, local_node] = Float32(thresholds[current])
            end
            path_directions[step, local_node] = directions[step]
            current = directions[step] ? 2 * current : 2 * current + 1
        end
    end

    selected = @view context.selected_device[:, 1:node_count]
    oct_gpu.regression_gpu_level_masks!(
        selected,
        context.X_device,
        CuArray(path_features),
        CuArray(path_thresholds),
        CuArray(path_directions),
        level,
        node_count,
    )
    selected_host = Array(selected)
    return [findall(@view selected_host[:, local_node]) for local_node in 1:node_count]
end

function _fit_mhdeoct_regression_gpu_backend_optimized(X, y, config)
    context = _regression_gpu_context(X, y, config)
    branch_count = 2^config.depth - 1
    feature_count = size(X, 2)
    global_splits = context.global_splits
    global_sorted_features = context.global_sorted_features
    candidate = zeros(Float64, 2 * branch_count)

    initial_candidate = if config.initialization == :cart
        _regression_cart_candidate(X, y, config, config.depth, global_splits)
    elseif config.initialization == :de
        first(_optimize_regression_gpu_tree(
            X,
            y,
            config,
            config.depth;
            context=context,
            row_indices_device=context.all_rows_device,
        ))
    else
        nothing
    end
    hybrid_candidate =
        initial_candidate === nothing ? nothing : copy(initial_candidate)

    for level in 0:(config.depth - 1)
        first_node = 2^level
        last_node = min(2^(level + 1) - 1, branch_count)
        nodes = collect(first_node:last_node)
        level_rows = _regression_gpu_level_rows(
            context,
            X,
            candidate,
            global_splits,
            level,
            nodes,
        )

        for (local_node, node) in pairs(nodes)
            rows = level_rows[local_node]
            local_y = @view y[rows]
            if length(rows) <= config.min_samples_leaf ||
               isempty(local_y) ||
               maximum(local_y) == minimum(local_y)
                continue
            end

            local_depth = min(config.horizon, config.depth - level)
            local_X = @view X[rows, :]
            local_rows_device = CuArray(Int32.(rows))
            local_seed = hybrid_candidate === nothing ? nothing :
                         _regression_subtree_candidate(
                hybrid_candidate,
                node,
                local_depth,
            )
            local_candidate, local_splits, _ = _optimize_regression_gpu_tree(
                local_X,
                local_y,
                config,
                local_depth;
                context=context,
                row_indices_device=local_rows_device,
                seed_candidate=local_seed,
            )
            _set_global_root!(
                candidate,
                node,
                local_candidate,
                local_splits,
                global_sorted_features,
                feature_count,
            )
            if hybrid_candidate !== nothing
                _set_global_root!(
                    hybrid_candidate,
                    node,
                    local_candidate,
                    local_splits,
                    global_sorted_features,
                    feature_count,
                )
            end
        end
    end

    candidates = initial_candidate === nothing ?
                 [candidate] : [candidate, hybrid_candidate, initial_candidate]
    centered_y = y .- sum(y) / length(y)
    total_sse = sum(abs2, centered_y)
    fitness = [
        _regression_cpu_fitness(
            item,
            X,
            centered_y,
            config,
            config.depth,
            global_splits,
            total_sse,
        ) for item in candidates
    ]
    best_candidate = candidates[argmin(fitness)]
    return _finish_regression_result(
        best_candidate,
        X,
        y,
        config,
        global_splits,
    )
end

function _fit_mhdeoct_regression_backend(X, y, config, backend)
    backend == :gpu &&
        return _fit_mhdeoct_regression_gpu_backend_optimized(X, y, config)
    optimizer = _optimize_regression_cpu_tree
    branch_count = 2^config.depth - 1
    feature_count = size(X, 2)
    global_splits, global_sorted_features = oct_gpu.split_x(X)
    candidate = zeros(Float64, 2 * branch_count)

    initial_candidate = if config.initialization == :cart
        _regression_cart_candidate(X, y, config, config.depth, global_splits)
    elseif config.initialization == :de
        first(optimizer(X, y, config, config.depth))
    else
        nothing
    end
    hybrid_candidate =
        initial_candidate === nothing ? nothing : copy(initial_candidate)

    for node in 1:branch_count
        selected = _cpu_rows_at_node(X, node, candidate, global_splits)
        local_y = y[selected]
        if count(selected) <= config.min_samples_leaf ||
           isempty(local_y) ||
           maximum(local_y) == minimum(local_y)
            continue
        end

        node_level = floor(Int, log2(node))
        local_depth = min(config.horizon, config.depth - node_level)
        local_X = X[selected, :]
        local_candidate, local_splits, _ =
            optimizer(local_X, local_y, config, local_depth)
        _set_global_root!(
            candidate,
            node,
            local_candidate,
            local_splits,
            global_sorted_features,
            feature_count,
        )
        if hybrid_candidate !== nothing
            _set_global_root!(
                hybrid_candidate,
                node,
                local_candidate,
                local_splits,
                global_sorted_features,
                feature_count,
            )
        end
    end

    candidates = initial_candidate === nothing ?
                 [candidate] : [candidate, hybrid_candidate, initial_candidate]
    centered_y = y .- sum(y) / length(y)
    total_sse = sum(abs2, centered_y)
    fitness = [
        _regression_cpu_fitness(
            item,
            X,
            centered_y,
            config,
            config.depth,
            global_splits,
            total_sse,
        ) for item in candidates
    ]
    best_candidate = candidates[argmin(fitness)]
    return _finish_regression_result(
        best_candidate,
        X,
        y,
        config,
        global_splits,
    )
end

function _fit_mhdeoct_regression_cpu_backend(X, y, config)
    return _fit_mhdeoct_regression_backend(X, y, config, :cpu)
end

function _fit_mhdeoct_regression_gpu_backend(X, y, config)
    return _fit_mhdeoct_regression_backend(X, y, config, :gpu)
end
