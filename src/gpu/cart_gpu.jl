# GPU version of the CART warm start: DecisionTree_modified.build_tree(y, X, 0,
# max_depth, min_samples_leaf) with the entropy loss. The split search runs on the
# GPU; the node order, the feature permutation and every draw from the random number
# generator are replayed on the CPU exactly as in treeclassifier._split!, so the tree
# and the generator state afterwards match the CPU version. The one difference is
# near-ties within a feature: the GPU takes the first threshold whose purity is within
# isapprox tolerance of the feature's best, while the CPU keeps the first threshold
# that beat the running best by more than that tolerance.
module cart_gpu
using CUDA
using Random
using ..DecisionTree_modified: DecisionTree_modified, Leaf, Node, Root

const util = DecisionTree_modified.treeclassifier.util

export SortedColumns, build_tree_gpu

"""
    SortedColumns(X_d)

Training data on the GPU together with each column's sort order (the sample indices
that sort the column with `isless`). The CART warm start and `oct_gpu.split_x` use the
orders, so each column of a node's data is sorted once.
"""
struct SortedColumns{T,M<:CuMatrix{T}}
    X::M
    order::CuMatrix{Int32}
end

function SortedColumns(X::CuMatrix)
    order = CuArray{Int32}(undef, size(X))
    for f in 1:size(X, 2)
        order[:, f] .= Int32.(sortperm(view(X, :, f)))
    end
    return SortedColumns(X, order)
end

Base.size(S::SortedColumns, dims...) = size(S.X, dims...)
Base.eltype(::Type{<:SortedColumns{T}}) where {T} = T

# Sorted values and labels of the node's samples for each feature in `feats`:
# vs[i, c] = X[order[i, feats[c]], feats[c]], ys[i, c] = Y[order[i, feats[c]]].
function gather_sorted!(vs, ys, X, Y, order, feats)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    m, F = size(vs)
    if k <= m * F
        c, i = divrem(k - 1, m)
        f = feats[c + 1]
        @inbounds s = order[i + 1, f]
        @inbounds vs[i + 1, c + 1] = X[s, f]
        @inbounds ys[i + 1, c + 1] = Y[s]
    end
    return
end

# Purity of the split after sorted position i of feature column c, as computed by
# treeclassifier._split!: -(nl*entropy(ncl, nl) + nr*entropy(ncr, nr)); -Inf where
# values i and i+1 are equal or a side would have fewer than min_samples_leaf samples.
function split_purity!(pur, vs, pref, totals, min_samples_leaf)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    m, F = size(vs)
    if k <= (m - 1) * F
        c, i = divrem(k - 1, m - 1)
        i += 1
        c += 1
        purity = -Inf
        @inbounds if vs[i, c] != vs[i + 1, c] && i >= min_samples_leaf && m - i >= min_samples_leaf
            nl, nr = i, m - i
            sl, sr = 0.0, 0.0
            for class in 1:length(totals)
                cl = pref[i, c, class]
                cr = totals[class] - cl
                cl > 0 && (sl += cl * log(Float64(cl)))
                cr > 0 && (sr += cr * log(Float64(cr)))
            end
            purity = -(nl * (log(Float64(nl)) - sl / nl) + nr * (log(Float64(nr)) - sr / nr))
        end
        @inbounds pur[i, c] = purity
    end
    return
end

function launch1d!(kernel, count, args...)
    count == 0 && return
    k = @cuda launch=false kernel(args...)
    threads = min(count, launch_configuration(k.fun).threads)
    k(args...; threads, blocks=cld(count, threads))
end

# Best split of each feature in `feats` for the node whose samples are the columns of
# `order` (sample ids sorted by each feature). Returns host vectors: whether the
# feature is constant in the node, whether any split honours min_samples_leaf, and the
# first near-best split (left size, left class counts, threshold_lo, threshold_hi).
function best_splits(X_d, Y_d, order, feats, totals_d, n_classes, min_samples_leaf)
    m = size(order, 1)
    F = length(feats)
    constant = trues(F)
    valid = falses(F)
    left_size = zeros(Int, F)
    left_counts = zeros(Int, n_classes, F)
    lo = zeros(eltype(X_d), F)
    hi = zeros(eltype(X_d), F)
    chunk = max(1, min(F, 32_000_000 ÷ m))
    for first in 1:chunk:F
        cols = first:min(first + chunk - 1, F)
        Fc = length(cols)
        feats_d = CuArray(Int32.(feats[cols]))
        vs = CuArray{eltype(X_d)}(undef, m, Fc)
        ys = CuArray{Int32}(undef, m, Fc)
        launch1d!(gather_sorted!, m * Fc, vs, ys, X_d, Y_d, order, feats_d)
        onehot = CuArray{Int32}(undef, m, Fc, n_classes)
        for class in 1:n_classes
            onehot[:, :, class] .= ys .== Int32(class)
        end
        pref = cumsum(onehot; dims=1)
        onehot = nothing
        ends = Array(vcat(view(vs, 1:1, :), view(vs, m:m, :)))
        constant[cols] .= ends[1, :] .== ends[2, :]
        m == 1 && continue
        pur = CuArray{Float64}(undef, m - 1, Fc)
        launch1d!(split_purity!, (m - 1) * Fc, pur, vs, pref, totals_d, min_samples_leaf)
        best = maximum(pur; dims=1)
        cutoff = best .- sqrt(eps(Float64)) .* abs.(best)
        rows = CuArray(Int32(1):Int32(m - 1))
        pos = vec(Array(minimum(ifelse.(pur .>= cutoff, rows, typemax(Int32)); dims=1)))
        best = vec(Array(best))
        valid[cols] .= best .> -Inf
        pos[best .== -Inf] .= 1
        lin = CuArray(pos .+ (0:Fc-1) .* m)
        lo[cols] = Array(vs[lin])
        hi[cols] = Array(vs[lin .+ 1])
        for class in 1:n_classes
            left_counts[class, cols] = Array(pref[lin .+ (class - 1) * m * Fc])
        end
        left_size[cols] = pos
    end
    return constant, valid, left_size, left_counts, lo, hi
end

# Stable partition of each column of `order` into the samples that go left and right:
# one column at a time, so the extra memory is a few vectors of the node's size.
function partition_column!(left, right, order, c, goes_left, left_rank)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(order, 1)
        @inbounds s = order[i, c]
        @inbounds rank = left_rank[i]
        @inbounds if goes_left[s]
            left[rank, c] = s
        else
            right[i - rank, c] = s
        end
    end
    return
end

function partition(order, goes_left)
    m, p = size(order)
    flags = CuArray{Int32}(undef, m)
    left_rank = CuArray{Int32}(undef, m)
    flags .= goes_left[view(order, :, 1)]
    nl = Int(sum(flags))
    left = CuArray{Int32}(undef, nl, p)
    right = CuArray{Int32}(undef, m - nl, p)
    for c in 1:p
        flags .= goes_left[view(order, :, c)]
        cumsum!(left_rank, flags)
        launch1d!(partition_column!, m, left, right, order, c, goes_left, left_rank)
    end
    return left, right
end

function node_counts(Y_d, order, n_classes)
    labels = view(order, :, 1)
    return [Int(count(==(Int32(class)), view(Y_d, labels))) for class in 1:n_classes]
end

# Build the tree below a node whose samples are the columns of `order`; `features`
# is shared with the sibling and permuted in place, as in the CPU version.
function grow(X_d, Y_d, order, features, depth, list, max_depth, min_samples_leaf, rng)
    n_classes = length(list)
    nc = node_counts(Y_d, order, n_classes)
    nt = sum(nc)
    label = argmax(nc)
    node_impurity = nt * util.entropy(nc, nt)
    leaf = Leaf{eltype(list)}(list[label], eltype(list)[])
    if min_samples_leaf * 2 > nt || 2 > nt || max_depth <= depth || nc[label] == nt
        return leaf
    end

    n_features = length(features)
    total_features = size(X_d, 2)
    non_consts_used = util.hypergeometric(n_features, total_features - n_features, total_features, rng)
    constant, valid, left_size, left_counts, lo, hi = best_splits(
        X_d, Y_d, order, features, CuArray(Int32.(nc)), n_classes, min_samples_leaf)
    column = Dict(f => c for (c, f) in enumerate(features))

    best_purity = typemin(Int)
    best_feature = -1
    threshold_lo = threshold_hi = zero(eltype(X_d))
    indf = 1
    n_const = 0
    unsplittable = true
    while (unsplittable || indf <= non_consts_used) && indf <= n_features
        indr = rand(rng, indf:n_features)
        features[indf], features[indr] = features[indr], features[indf]
        feature = features[indf]
        c = column[feature]
        if valid[c]
            unsplittable = false
            nl = left_size[c]
            ncl = left_counts[:, c]
            ncr = nc .- ncl
            purity = -(nl * util.entropy(ncl, nl) + (nt - nl) * util.entropy(ncr, nt - nl))
            if purity > best_purity && !isapprox(purity, best_purity)
                threshold_lo, threshold_hi = lo[c], hi[c]
                best_purity = purity
                best_feature = feature
            end
        end
        if constant[c]
            n_const += 1
            features[indf], features[n_const] = features[n_const], features[indf]
        end
        indf += 1
    end
    if unsplittable || best_purity + node_impurity < 0.0
        return leaf
    end

    # Samples with value <= threshold_lo go left.
    goes_left = CUDA.zeros(Bool, size(X_d, 1))
    samples = view(order, :, best_feature)
    goes_left[samples] .= view(X_d, :, best_feature)[samples] .<= threshold_lo
    left, right = partition(order, goes_left)
    goes_left = samples = nothing
    child_features = features[(n_const + 1):n_features]
    left_node = grow(X_d, Y_d, left, child_features, depth + 1, list, max_depth, min_samples_leaf, rng)
    left = nothing
    right_node = grow(X_d, Y_d, right, child_features, depth + 1, list, max_depth, min_samples_leaf, rng)
    threshold = (threshold_lo + threshold_hi) / 2.0
    return Node{typeof(threshold),eltype(list)}(best_feature, threshold, left_node, right_node)
end

"""
    build_tree_gpu(labels, features, max_depth, min_samples_leaf; rng)

`features` is a host matrix or a `SortedColumns`. Same tree as `DecisionTree_modified.build_tree(labels, features, 0, max_depth,
min_samples_leaf)` (entropy loss, all features), with the split search on the GPU.
Leaves carry their majority label but no sample labels.
"""
function build_tree_gpu(labels::AbstractVector, features::SortedColumns, max_depth, min_samples_leaf;
                        rng=Random.GLOBAL_RNG)
    p = size(features, 2)
    list, Y = util.assign(labels)
    Y_d = CuArray(Int32.(Y))
    max_depth == -1 && (max_depth = typemax(Int))
    node = grow(features.X, Y_d, features.order, collect(1:p), 0, list, max_depth, min_samples_leaf,
                DecisionTree_modified.mk_rng(rng))
    return Root{eltype(features),eltype(list)}(node, p, Float64[])
end

build_tree_gpu(labels::AbstractVector, features::AbstractMatrix, max_depth, min_samples_leaf; kwargs...) =
    build_tree_gpu(labels, SortedColumns(CuArray(features)), max_depth, min_samples_leaf; kwargs...)

end # module cart_gpu
