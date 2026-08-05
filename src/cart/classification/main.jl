# Classification-tree code adapted from DecisionTree.jl 0.12.3.

include("tree.jl")

function _convert(
    node::treeclassifier.NodeMeta{S}, list::AbstractVector{T}, labels::AbstractVector{T}
) where {S,T}
    if node.is_leaf
        return Leaf{T}(list[node.label], labels[node.region])
    end

    left = _convert(node.l, list, labels)
    right = _convert(node.r, list, labels)
    return Node{S,T}(node.feature, node.threshold, left, right)
end

function _update_impurity!(
    feature_importance::Vector{Float64}, node::treeclassifier.NodeMeta
)
    if !node.is_leaf
        _update_impurity!(feature_importance, node.l)
        _update_impurity!(feature_importance, node.r)
        feature_importance[node.feature] +=
            node.node_impurity - node.l.node_impurity - node.r.node_impurity
    end
    return nothing
end

function build_tree(
    labels::AbstractVector{T},
    features::AbstractMatrix{S},
    n_subfeatures=0,
    max_depth=-1,
    min_samples_leaf=1,
    min_samples_split=2,
    min_purity_increase=0.0;
    loss=util.entropy::Function,
    rng=Random.GLOBAL_RNG,
    impurity_importance::Bool=true,
) where {S,T}
    max_depth == -1 && (max_depth = typemax(Int))
    n_subfeatures == 0 && (n_subfeatures = size(features, 2))

    tree = treeclassifier.fit(;
        X=features,
        Y=labels,
        W=nothing,
        loss,
        max_features=Int(n_subfeatures),
        max_depth=Int(max_depth),
        min_samples_leaf=Int(min_samples_leaf),
        min_samples_split=Int(min_samples_split),
        min_purity_increase=Float64(min_purity_increase),
        rng=mk_rng(rng),
    )

    node = _convert(tree.root, tree.list, labels[tree.labels])
    if !impurity_importance
        return Root{S,T}(node, size(features, 2), Float64[])
    end

    feature_importance = zeros(Float64, size(features, 2))
    _update_impurity!(feature_importance, tree.root)
    feature_importance ./= size(features, 1)
    return Root{S,T}(node, size(features, 2), feature_importance)
end
