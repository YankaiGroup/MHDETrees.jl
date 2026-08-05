# Internal CART implementation adapted from DecisionTree.jl 0.12.3.
# Only the classification-tree functionality required for MH-DEOCT warm starts is
# retained. See THIRD_PARTY_NOTICE.md for attribution and license details.
module DecisionTree_modified

using Random

export Leaf, Node, Root, build_tree

struct Leaf{T}
    majority::T
    values::Vector{T}
end

struct Node{S,T}
    featid::Int
    featval::S
    left::Union{Leaf{T},Node{S,T}}
    right::Union{Leaf{T},Node{S,T}}
end

const LeafOrNode{S,T} = Union{Leaf{T},Node{S,T}}

struct Root{S,T}
    node::LeafOrNode{S,T}
    n_feat::Int
    featim::Vector{Float64}

    function Root{S,T}(
        node::LeafOrNode{S,T}, n_feat::Int, featim::Vector{Float64}
    ) where {S,T}
        return new{S,T}(node, n_feat, featim)
    end
end

mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::Integer) = Random.MersenneTwister(seed)

include("util.jl")
include("classification/main.jl")

end
