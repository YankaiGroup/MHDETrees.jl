using CUDA
using DelimitedFiles
using Random
using StatsBase

"""
    MHDEOCTConfig(; kwargs...)

Configuration for [`fit`](@ref).

The defaults reproduce the main MH-DEOCT setup from the paper. Smaller values
for `population_size` and `generations` are useful for smoke tests. Choose
`algorithm=:deoct` for full-tree optimization or `algorithm=:mhdeoct` for
moving-horizon optimization. Both support `backend=:cpu` and `backend=:gpu`.
Regression also supports `algorithm=:cart` as a deterministic CPU baseline.
Set `task=:regression` to minimize within-leaf squared error and predict the
training-target mean in each leaf. Set `leaf_model=:linear` to replace each
leaf mean with an independently fitted affine model after the tree structure is
optimized. `linear_regularization` controls the dimensionless ridge penalty
used to stabilize those local fits. The default `task=:classification`
preserves the classification API. `horizon` is used only by MH-DEOCT. Set
`verbose=true` to print GPU training diagnostics. Regression GPU tree-search
fitness uses `Float32` target moments and costs (`Int32` leaf counts); final
leaf models, stored parameters, and prediction use `Float64`.
"""
Base.@kwdef struct MHDEOCTConfig
    depth::Int = 2
    horizon::Int = 2
    population_size::Int = 100
    generations::Int = 600
    min_samples_leaf::Int = 1
    alpha::Float64 = 0.0
    warm_start::Bool = true
    initialization::Symbol = :cart
    task::Symbol = :classification
    leaf_model::Symbol = :mean
    linear_regularization::Float64 = 1.0e-2
    algorithm::Symbol = :mhdeoct
    backend::Symbol = :auto
    seed::Int = 1
    scale_features::Bool = true
    verbose::Bool = false
end

abstract type AbstractMHDEOCTModel end

"""
    MHDEOCTModel

A fitted DEOCT or MH-DEOCT classifier. Use [`predict`](@ref) for class
predictions and [`accuracy`](@ref) for classification accuracy.
"""
struct MHDEOCTModel{T,L} <: AbstractMHDEOCTModel
    candidate::Vector{T}
    leaf_class_indices::Vector{Int}
    classes::Vector{L}
    splits::Vector{Vector{Float64}}
    feature_min::Vector{Float64}
    feature_scale::Vector{Float64}
    config::MHDEOCTConfig
    algorithm::Symbol
    backend::Symbol
    training_errors::Int
    min_leaf_violations::Int
end

"""
    MHDEOCTRegressorModel

A fitted CART, DEOCT, or MH-DEOCT regressor. With `leaf_model=:mean`, each
nonempty leaf predicts its training-target mean. With `leaf_model=:linear`,
each leaf predicts an affine function of all input features. Empty leaves use
the corresponding global mean or affine model as a safe fallback. Use
[`predict`](@ref), [`mean_squared_error`](@ref),
[`root_mean_squared_error`](@ref), or [`r2_score`](@ref) to evaluate it.
"""
struct MHDEOCTRegressorModel{T} <: AbstractMHDEOCTModel
    candidate::Vector{T}
    leaf_values::Vector{Float64}
    leaf_coefficients::Matrix{Float64}
    leaf_counts::Vector{Int}
    splits::Vector{Vector{Float64}}
    feature_min::Vector{Float64}
    feature_scale::Vector{Float64}
    config::MHDEOCTConfig
    algorithm::Symbol
    backend::Symbol
    training_sse::Float64
    min_leaf_violations::Int
end

# Preserve the mean-leaf constructor used by the original regression API.
function MHDEOCTRegressorModel(
    candidate::Vector{T},
    leaf_values::Vector{Float64},
    leaf_counts::Vector{Int},
    splits::Vector{Vector{Float64}},
    feature_min::Vector{Float64},
    feature_scale::Vector{Float64},
    config::MHDEOCTConfig,
    algorithm::Symbol,
    backend::Symbol,
    training_sse::Float64,
    min_leaf_violations::Int,
) where {T}
    coefficients = zeros(Float64, length(feature_min), length(leaf_values))
    return MHDEOCTRegressorModel(
        candidate,
        leaf_values,
        coefficients,
        leaf_counts,
        splits,
        feature_min,
        feature_scale,
        config,
        algorithm,
        backend,
        training_sse,
        min_leaf_violations,
    )
end

"""Return whether CUDA.jl can use a CUDA-capable device."""
gpu_available() = CUDA.functional()

"""Load the bundled Iris dataset as `(X, y)`."""
function load_iris()
    path = normpath(joinpath(@__DIR__, "..", "data", "iris.csv"))
    data = DelimitedFiles.readdlm(path, ',', Any)
    return Float64.(data[:, 1:4]), String.(data[:, 5])
end

"""
    load_yacht()

Load the bundled UCI Yacht Hydrodynamics regression dataset as `(X, y)`. It
contains 308 experiments, six real-valued input features, and residuary
resistance as the numeric target.
"""
function load_yacht()
    path = normpath(joinpath(@__DIR__, "..", "data", "yacht_hydrodynamics.data"))
    data = Float64.(DelimitedFiles.readdlm(path))
    return data[:, 1:6], vec(data[:, 7])
end

function _validate_config(config::MHDEOCTConfig)
    config.depth >= 1 || throw(ArgumentError("depth must be at least 1"))
    config.task in (:classification, :regression) ||
        throw(ArgumentError("task must be :classification or :regression"))
    config.leaf_model in (:mean, :linear) ||
        throw(ArgumentError("leaf_model must be :mean or :linear"))
    config.task == :classification && config.leaf_model != :mean && throw(ArgumentError(
        "leaf_model=:linear is available only for task=:regression",
    ))
    isfinite(config.linear_regularization) && config.linear_regularization >= 0 ||
        throw(ArgumentError("linear_regularization must be finite and nonnegative"))
    config.algorithm in (:cart, :deoct, :mhdeoct) ||
        throw(ArgumentError("algorithm must be :cart, :deoct, or :mhdeoct"))
    config.algorithm == :cart && config.task != :regression && throw(ArgumentError(
        "algorithm=:cart is available only for task=:regression",
    ))
    config.algorithm == :cart && config.backend == :gpu && throw(ArgumentError(
        "algorithm=:cart supports backend=:cpu only",
    ))
    if config.algorithm == :mhdeoct
        1 <= config.horizon <= config.depth ||
            throw(ArgumentError("horizon must be between 1 and depth for MH-DEOCT"))
    end
    config.population_size >= 4 ||
        throw(ArgumentError("population_size must be at least 4"))
    config.generations >= 0 || throw(ArgumentError("generations cannot be negative"))
    config.min_samples_leaf >= 1 ||
        throw(ArgumentError("min_samples_leaf must be at least 1"))
    config.alpha >= 0 || throw(ArgumentError("alpha cannot be negative"))
    config.initialization in (:cart, :de, :none) ||
        throw(ArgumentError("initialization must be :cart, :de, or :none"))
    config.backend in (:auto, :cpu, :gpu) ||
        throw(ArgumentError("backend must be :auto, :cpu, or :gpu"))
    return config
end

function _prepare_features(X::AbstractMatrix{<:Real}, scale_features::Bool)
    isempty(X) && throw(ArgumentError("X must contain at least one sample and feature"))
    X_float = Matrix{Float64}(X)
    all(isfinite, X_float) || throw(ArgumentError("X must contain only finite values"))

    feature_min = vec(minimum(X_float; dims=1))
    feature_max = vec(maximum(X_float; dims=1))
    feature_scale = feature_max .- feature_min
    feature_scale[feature_scale .== 0.0] .= 1.0

    if scale_features
        X_float = (X_float .- feature_min') ./ feature_scale'
    elseif any(X_float .< 0.0) || any(X_float .> 1.0)
        throw(ArgumentError("features must be in [0, 1] when scale_features=false"))
    else
        feature_min .= 0.0
        feature_scale .= 1.0
    end

    return X_float, feature_min, feature_scale
end

function _encode_labels(y::AbstractVector)
    isempty(y) && throw(ArgumentError("y must contain at least one label"))
    classes = collect(unique(y))
    try
        sort!(classes)
    catch
        # Preserve first-occurrence order for label types without an ordering.
    end
    length(classes) >= 2 || throw(ArgumentError("y must contain at least two classes"))
    class_to_index = Dict(label => index for (index, label) in pairs(classes))
    encoded = [class_to_index[label] for label in y]
    return classes, encoded
end

function _prepare_regression_targets(y::AbstractVector)
    isempty(y) && throw(ArgumentError("y must contain at least one target"))
    all(target -> target isa Real, y) ||
        throw(ArgumentError("regression targets must be real numbers"))
    targets = Float64.(y)
    all(isfinite, targets) ||
        throw(ArgumentError("regression targets must contain only finite values"))
    return targets
end

function _prepare_gpu_backend(X_train, y_encoded, class_count, config)
    CUDA.functional() ||
        error("The GPU backend requires a functional CUDA device; check CUDA.versioninfo()")
    Random.seed!(config.seed)
    tree_size = 2^(config.depth + 1) - 1
    tree_encoding = 3 # discrete feature/threshold encoding with inactive nodes
    variables_per_branch = 2
    population_size = Int32(config.population_size)
    X_device = CuArray(Float32.(X_train))

    kernels, threads = oct_gpu.gpu_init(config.verbose)
    arguments = oct_gpu.args_pre(
        class_count,
        tree_encoding,
        X_device,
        X_train,
        y_encoded,
        tree_size,
        config.min_samples_leaf,
        kernels,
        threads,
        population_size,
        config.alpha,
        config.verbose,
    )

    return (
        tree_size=tree_size,
        tree_encoding=tree_encoding,
        variables_per_branch=variables_per_branch,
        population_size=population_size,
        X_device=X_device,
        arguments=arguments,
    )
end

function _finish_gpu_result(
    candidate,
    X_train,
    y_indicator,
    class_count,
    config,
    context,
)
    splits = [Float64.(split) for split in context.arguments[13]]
    training_errors, leaf_classes, min_leaf_violations = oct_gpu.OCT(
        context.tree_encoding,
        copy(candidate),
        X_train,
        y_indicator,
        collect(1:class_count),
        context.tree_size,
        config.min_samples_leaf,
        4,
        config.alpha,
        splits,
    )

    return (
        candidate=vec(Float64.(candidate)),
        leaf_classes=vec(Int.(leaf_classes)),
        splits=splits,
        training_errors=Int(training_errors),
        min_leaf_violations=Int(min_leaf_violations),
    )
end

function _fit_deoct_gpu_backend(X_train, y_encoded, y_indicator, class_count, config)
    context = _prepare_gpu_backend(X_train, y_encoded, class_count, config)
    use_cart = config.initialization == :cart ||
               (config.initialization == :de && config.warm_start)
    candidate, _, _, _, _ = de_gpu.DEb1b_warmStart(
        context.arguments,
        context.tree_encoding,
        config.generations,
        X_train,
        y_encoded,
        context.tree_size,
        context.variables_per_branch,
        config.min_samples_leaf,
        context.population_size,
        0.5,
        0.1,
        0,
        nothing,
        config.seed,
        use_cart,
        0,
        config.verbose,
    )
    return _finish_gpu_result(
        candidate,
        X_train,
        y_indicator,
        class_count,
        config,
        context,
    )
end

function _fit_mhdeoct_gpu_backend(X_train, y_encoded, y_indicator, class_count, config)
    context = _prepare_gpu_backend(X_train, y_encoded, class_count, config)

    initial_candidate = nothing
    if config.initialization != :none
        initialization_flag = config.initialization == :cart ? 1 : 0
        candidate, _, cart_candidate, _, _ = de_gpu.DEb1b_warmStart(
            context.arguments,
            context.tree_encoding,
            config.generations,
            X_train,
            y_encoded,
            context.tree_size,
            context.variables_per_branch,
            config.min_samples_leaf,
            context.population_size,
            0.5,
            0.1,
            0,
            nothing,
            config.seed,
            config.warm_start,
            initialization_flag,
            config.verbose,
        )
        initial_candidate = config.initialization == :cart ? cart_candidate : candidate
    end

    candidate = de_gpu.layer_by_layer_original_warmStart(
        context.arguments,
        context.tree_encoding,
        config.horizon,
        config.generations,
        context.X_device,
        X_train,
        y_encoded,
        y_indicator,
        context.tree_size,
        context.variables_per_branch,
        config.min_samples_leaf,
        context.population_size,
        0.5,
        0.1,
        0,
        initial_candidate,
        config.warm_start && config.initialization != :none,
        config.verbose,
    )
    return _finish_gpu_result(
        candidate,
        X_train,
        y_indicator,
        class_count,
        config,
        context,
    )
end

"""
    fit(X, y; config=MHDEOCTConfig())

Train a classification or regression tree on rows of `X`. Select the task with
`task=:classification` (the default) or `task=:regression`, full-tree DEOCT with
`algorithm=:deoct`, or moving-horizon DEOCT with `algorithm=:mhdeoct`. Both
DE algorithms support `backend=:cpu` and `backend=:gpu`. Regression additionally
supports a deterministic `algorithm=:cart`, `backend=:cpu` baseline. The default
`backend=:auto` selects the GPU when CUDA is functional and otherwise uses the
CPU, except that CART always selects the CPU.

Features are min-max scaled by default and the fitted scaling parameters are
stored in the returned model. Regression tree structures minimize the sum of
squared deviations from leaf means. Each leaf then predicts either that mean
or, with `leaf_model=:linear`, an independently fitted affine model. Set
`config.verbose=true` to print GPU training diagnostics; the default is quiet.
The regression GPU search uses `Int32` leaf counts and `Float32` moments and
fitness, then returns the selected tree for `Float64` final leaf fitting and
CPU prediction.
"""
function fit(
    X::AbstractMatrix{<:Real},
    y::AbstractVector;
    config::MHDEOCTConfig=MHDEOCTConfig(),
)
    _validate_config(config)
    size(X, 1) == length(y) ||
        throw(DimensionMismatch("X has $(size(X, 1)) rows but y has $(length(y)) values"))

    X_train, feature_min, feature_scale = _prepare_features(X, config.scale_features)
    backend = if config.algorithm == :cart
        :cpu
    elseif config.backend == :auto
        CUDA.functional() ? :gpu : :cpu
    else
        config.backend
    end

    if config.task == :classification
        classes, y_encoded = _encode_labels(y)
        class_count = length(classes)
        y_indicator = StatsBase.indicatormat(y_encoded, class_count)
        result = if config.algorithm == :deoct && backend == :cpu
            _fit_deoct_cpu_backend(
                X_train,
                y_encoded,
                y_indicator,
                class_count,
                config,
            )
        elseif config.algorithm == :deoct && backend == :gpu
            _fit_deoct_gpu_backend(
                X_train,
                y_encoded,
                y_indicator,
                class_count,
                config,
            )
        elseif config.algorithm == :mhdeoct && backend == :cpu
            _fit_mhdeoct_cpu_backend(
                X_train,
                y_encoded,
                y_indicator,
                class_count,
                config,
            )
        else
            _fit_mhdeoct_gpu_backend(
                X_train,
                y_encoded,
                y_indicator,
                class_count,
                config,
            )
        end

        return MHDEOCTModel(
            result.candidate,
            result.leaf_classes,
            classes,
            result.splits,
            feature_min,
            feature_scale,
            config,
            config.algorithm,
            backend,
            result.training_errors,
            result.min_leaf_violations,
        )
    end

    targets = _prepare_regression_targets(y)
    result = if config.algorithm == :cart
        _fit_cart_regression_cpu_backend(X_train, targets, config)
    elseif config.algorithm == :deoct && backend == :cpu
        _fit_deoct_regression_cpu_backend(X_train, targets, config)
    elseif config.algorithm == :deoct && backend == :gpu
        _fit_deoct_regression_gpu_backend(X_train, targets, config)
    elseif config.algorithm == :mhdeoct && backend == :cpu
        _fit_mhdeoct_regression_cpu_backend(X_train, targets, config)
    else
        _fit_mhdeoct_regression_gpu_backend(X_train, targets, config)
    end
    return MHDEOCTRegressorModel(
        result.candidate,
        result.leaf_values,
        result.leaf_coefficients,
        result.leaf_counts,
        result.splits,
        feature_min,
        feature_scale,
        config,
        config.algorithm,
        backend,
        result.training_sse,
        result.min_leaf_violations,
    )
end

function _transform_features(model::AbstractMHDEOCTModel, X::AbstractMatrix{<:Real})
    size(X, 2) == length(model.feature_min) || throw(DimensionMismatch(
        "X has $(size(X, 2)) features but the model expects $(length(model.feature_min))",
    ))
    X_float = Matrix{Float64}(X)
    all(isfinite, X_float) || throw(ArgumentError("X must contain only finite values"))
    return (X_float .- model.feature_min') ./ model.feature_scale'
end

"""Predict class labels for the rows of `X`."""
function predict(model::MHDEOCTModel, X::AbstractMatrix{<:Real})
    X_scaled = _transform_features(model, X)
    branch_count = 2^model.config.depth - 1
    feature_count = size(X_scaled, 2)
    a, b, active = oct_gpu.trans_params_azd(
        copy(model.candidate),
        feature_count,
        branch_count,
        model.splits,
    )

    predictions = Vector{eltype(model.classes)}(undef, size(X_scaled, 1))
    for sample in axes(X_scaled, 1)
        node = 1
        while node <= branch_count
            feature = findfirst(@view a[:, node])
            go_left = active[node] && feature !== nothing && X_scaled[sample, feature] < b[node]
            node = go_left ? 2 * node : 2 * node + 1
        end
        class_index = model.leaf_class_indices[node - branch_count]
        predictions[sample] = model.classes[class_index]
    end
    return predictions
end

"""Predict numeric responses using the fitted mean or affine leaf models."""
function predict(model::MHDEOCTRegressorModel, X::AbstractMatrix{<:Real})
    X_scaled = _transform_features(model, X)
    assignments = _regression_leaf_assignments(
        model.candidate,
        X_scaled,
        model.config.depth,
        model.splits,
    )
    predictions = model.leaf_values[assignments]
    if model.config.leaf_model == :linear
        for sample in axes(X_scaled, 1)
            leaf = assignments[sample]
            predictions[sample] += dot(
                @view(X_scaled[sample, :]),
                @view(model.leaf_coefficients[:, leaf]),
            )
        end
    end
    return predictions
end

"""Return the fraction of labels correctly predicted by `model`."""
function accuracy(model::MHDEOCTModel, X::AbstractMatrix{<:Real}, y::AbstractVector)
    size(X, 1) == length(y) ||
        throw(DimensionMismatch("X has $(size(X, 1)) rows but y has $(length(y)) labels"))
    return sum(predict(model, X) .== y) / length(y)
end

"""Return the mean squared prediction error for a fitted regressor."""
function mean_squared_error(
    model::MHDEOCTRegressorModel,
    X::AbstractMatrix{<:Real},
    y::AbstractVector,
)
    size(X, 1) == length(y) || throw(DimensionMismatch(
        "X has $(size(X, 1)) rows but y has $(length(y)) targets",
    ))
    targets = _prepare_regression_targets(y)
    predictions = predict(model, X)
    return sum(abs2, targets .- predictions) / length(targets)
end

"""Return the root mean squared prediction error for a fitted regressor."""
function root_mean_squared_error(
    model::MHDEOCTRegressorModel,
    X::AbstractMatrix{<:Real},
    y::AbstractVector,
)
    return sqrt(mean_squared_error(model, X, y))
end

"""
    r2_score(model, X, y)

Return the coefficient of determination. For a constant target, return `1.0`
for perfect predictions and `0.0` otherwise.
"""
function r2_score(
    model::MHDEOCTRegressorModel,
    X::AbstractMatrix{<:Real},
    y::AbstractVector,
)
    size(X, 1) == length(y) || throw(DimensionMismatch(
        "X has $(size(X, 1)) rows but y has $(length(y)) targets",
    ))
    targets = _prepare_regression_targets(y)
    predictions = predict(model, X)
    residual_sse = sum(abs2, targets .- predictions)
    target_mean = sum(targets) / length(targets)
    total_sse = sum(abs2, targets .- target_mean)
    return total_sse == 0.0 ? Float64(residual_sse == 0.0) : 1.0 - residual_sse / total_sse
end

function Base.show(io::IO, model::MHDEOCTModel)
    print(
        io,
        "MHDEOCTModel(algorithm=$(model.algorithm), backend=$(model.backend), ",
        "depth=$(model.config.depth), ",
        "classes=$(length(model.classes)), ",
        "features=$(length(model.feature_min)), training_errors=$(model.training_errors), ",
        "leaf_violations=$(model.min_leaf_violations))",
    )
end


function Base.show(io::IO, model::MHDEOCTRegressorModel)
    sample_count = sum(model.leaf_counts)
    training_mse = model.training_sse / sample_count
    print(
        io,
        "MHDEOCTRegressorModel(algorithm=$(model.algorithm), backend=$(model.backend), ",
        "depth=$(model.config.depth), ",
        "leaf_model=$(model.config.leaf_model), ",
        "features=$(length(model.feature_min)), training_mse=$(training_mse), ",
        "leaf_violations=$(model.min_leaf_violations))",
    )
end
