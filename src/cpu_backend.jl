using Random
using StatsBase

function _cpu_fitness(
    candidate,
    X,
    y_indicator,
    class_count,
    tree_size,
    min_samples_leaf,
    alpha,
    splits,
)
    return oct_gpu.OCT(
        3,
        copy(candidate),
        X,
        y_indicator,
        collect(1:class_count),
        tree_size,
        min_samples_leaf,
        0,
        alpha,
        splits,
    )
end

function _random_cpu_population(rng, population_size, branch_count, feature_count)
    population = zeros(Float64, population_size, 2 * branch_count)
    population[:, 1:branch_count] .= rand(
        rng,
        0:feature_count,
        population_size,
        branch_count,
    )
    population[:, branch_count + 1:end] .= rand(rng, population_size, branch_count)
    population[:, 1] .= rand(rng, 1:feature_count, population_size)
    return population
end

function _repair_cpu_candidate!(candidate, branch_count, feature_count)
    for index in 1:branch_count
        candidate[index] = floor(clamp(candidate[index], 0, feature_count))
    end
    candidate[1] = max(1, candidate[1])
    candidate[branch_count + 1:end] .=
        clamp.(candidate[branch_count + 1:end], 0.0, 1.0)
    return candidate
end

function _cart_cpu_candidate(X, y_encoded, config, depth, sorted_features)
    candidate, _ = warmstart_gpu.warm_start_DT(
        X,
        y_encoded,
        3,
        2,
        config.min_samples_leaf,
        depth,
        0.0,
        sorted_features,
    )
    return vec(Float64.(candidate))
end

function _optimize_cpu_tree(X, y_encoded, y_indicator, class_count, config, depth)
    rng = MersenneTwister(config.seed)
    tree_size = 2^(depth + 1) - 1
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
        population[1, :] .=
            _cart_cpu_candidate(X, y_encoded, config, depth, sorted_features)
    end

    fitness = [
        _cpu_fitness(
            view(population, index, :),
            X,
            y_indicator,
            class_count,
            tree_size,
            config.min_samples_leaf,
            config.alpha,
            splits,
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
                       rand(rng) .* (view(population, first_parent, :) .-
                                     view(population, second_parent, :))
            _repair_cpu_candidate!(mutation, branch_count, feature_count)

            crossover = rand(rng, length(best_candidate)) .< 0.1
            crossover[rand(rng, eachindex(crossover))] = true
            trial = copy(view(population, index, :))
            trial[crossover] .= mutation[crossover]
            _repair_cpu_candidate!(trial, branch_count, feature_count)

            trial_fitness = _cpu_fitness(
                trial,
                X,
                y_indicator,
                class_count,
                tree_size,
                config.min_samples_leaf,
                config.alpha,
                splits,
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

function _finish_cpu_result(candidate, X, y_indicator, class_count, config, splits)
    tree_size = 2^(config.depth + 1) - 1
    training_errors, leaf_classes, min_leaf_violations = oct_gpu.OCT(
        3,
        copy(candidate),
        X,
        y_indicator,
        collect(1:class_count),
        tree_size,
        config.min_samples_leaf,
        4,
        config.alpha,
        splits,
    )
    return (
        candidate=vec(Float64.(candidate)),
        leaf_classes=vec(Int.(leaf_classes)),
        splits=[Float64.(split) for split in splits],
        training_errors=Int(training_errors),
        min_leaf_violations=Int(min_leaf_violations),
    )
end

function _fit_deoct_cpu_backend(X, y_encoded, y_indicator, class_count, config)
    candidate, splits, _ =
        _optimize_cpu_tree(X, y_encoded, y_indicator, class_count, config, config.depth)
    return _finish_cpu_result(candidate, X, y_indicator, class_count, config, splits)
end

function _cpu_rows_at_node(X, node, candidate, splits)
    node == 1 && return trues(size(X, 1))

    branch_count = length(candidate) ÷ 2
    feature_count = size(X, 2)
    features, thresholds, active = oct_gpu.trans_params_azd(
        copy(candidate),
        feature_count,
        branch_count,
        splits,
    )

    directions = Bool[]
    current = node
    while current > 1
        push!(directions, iseven(current))
        current = fld(current, 2)
    end
    reverse!(directions)

    selected = trues(size(X, 1))
    current = 1
    for requires_left in directions
        feature = findfirst(@view features[:, current])
        for row in axes(X, 1)
            selected[row] || continue
            goes_left = active[current] && feature !== nothing &&
                        X[row, feature] < thresholds[current]
            selected[row] = goes_left == requires_left
        end
        current = requires_left ? 2 * current : 2 * current + 1
    end
    return selected
end

function _set_global_root!(
    global_candidate,
    node,
    local_candidate,
    local_splits,
    global_sorted_features,
    feature_count,
)
    global_branch_count = length(global_candidate) ÷ 2
    local_branch_count = length(local_candidate) ÷ 2
    features, thresholds, active = oct_gpu.trans_params_azd(
        copy(local_candidate),
        feature_count,
        local_branch_count,
        local_splits,
    )
    feature = findfirst(@view features[:, 1])

    if active[1] && feature !== nothing
        root_feature = falses(feature_count, 1)
        root_feature[feature, 1] = true
        encoded_threshold = warmstart_gpu.encode_b(
            root_feature,
            [thresholds[1]],
            global_sorted_features,
        )[1]
        global_candidate[node] = feature
        global_candidate[global_branch_count + node] = encoded_threshold
    else
        global_candidate[node] = 0.0
        global_candidate[global_branch_count + node] = 0.0
    end
    return global_candidate
end

function _fit_mhdeoct_cpu_backend(X, y_encoded, y_indicator, class_count, config)
    branch_count = 2^config.depth - 1
    feature_count = size(X, 2)
    global_splits, global_sorted_features = oct_gpu.split_x(X)
    candidate = zeros(Float64, 2 * branch_count)

    initial_candidate = if config.initialization == :cart
        _cart_cpu_candidate(
            X,
            y_encoded,
            config,
            config.depth,
            global_sorted_features,
        )
    elseif config.initialization == :de
        first(_optimize_cpu_tree(
            X,
            y_encoded,
            y_indicator,
            class_count,
            config,
            config.depth,
        ))
    else
        nothing
    end
    hybrid_candidate = initial_candidate === nothing ? nothing : copy(initial_candidate)

    for node in 1:branch_count
        selected = _cpu_rows_at_node(X, node, candidate, global_splits)
        local_y = y_encoded[selected]
        if count(selected) <= config.min_samples_leaf || length(unique(local_y)) <= 1
            continue
        end

        node_level = floor(Int, log2(node))
        local_depth = min(config.horizon, config.depth - node_level)
        local_X = X[selected, :]
        local_indicator = y_indicator[:, selected]
        local_candidate, local_splits, _ = _optimize_cpu_tree(
            local_X,
            local_y,
            local_indicator,
            class_count,
            config,
            local_depth,
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

    candidates = initial_candidate === nothing ?
                 [candidate] : [candidate, hybrid_candidate, initial_candidate]
    tree_size = 2^(config.depth + 1) - 1
    fitness = [
        _cpu_fitness(
            item,
            X,
            y_indicator,
            class_count,
            tree_size,
            config.min_samples_leaf,
            config.alpha,
            global_splits,
        ) for item in candidates
    ]
    best_candidate = candidates[argmin(fitness)]
    return _finish_cpu_result(
        best_candidate,
        X,
        y_indicator,
        class_count,
        config,
        global_splits,
    )
end
