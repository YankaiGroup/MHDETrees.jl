module warmstart_gpu
using ..DecisionTree_modified
using StatsBase
using Random

export warm_start_DT, warm_start_DT_params, encode_b

# warm start of Decision Tree (CART)
function warm_start_DT(X, y,tree_da, var_number, Nmin, D = 4, prune_val=0.0, sorted_X=nothing, loss_flag=0)
    n,p = size(X)
    Tb = 2^D-1
    T = 2^(D+1)-1
    if loss_flag == 0
        cart_model = DecisionTree_modified.build_tree(y, X, 0, D, Nmin)
    else
        cart_model = DecisionTree_modified.build_tree(y, X, 0, D, Nmin, loss=DecisionTree_modified.util.normal_loss)
    end
    a,b,c,d = warm_start_DT_params(zeros(p,Tb), zeros(Tb), zeros(T), zeros(Tb), 1, cart_model.node, 2^D:T)

    # change binary a to 1-d float number, which is used in Differential evolution
    if tree_da !=3
        aFloat =  zeros(Float64, Tb)
        for j in 1:Tb
            if iszero(a[:,j])
                aFloat[j] = 0.0
            else
                idx_one = findall(!iszero, a[:,j])
                oneToFloat =  (idx_one[1]-1)/p .+ rand(Float64,1)*((idx_one[1])/p-(idx_one[1]-1)/p)
                aFloat[j] = oneToFloat[1]
            end
        end
    else
        aFloat =  zeros(Float64, Tb)
        for j in 1:Tb
            if iszero(a[:,j])
                aFloat[j] = 0.0
            else
                idx_one = findall(!iszero, a[:,j])
                aFloat[j] = idx_one[1]
            end
        end
    end 

    # find the index of b in splits
    b = encode_b(a, b, sorted_X)
    if var_number == 3
        DT_adb = vcat(aFloat,d,b)                         # vector (9,)
        DT_adb_mat =  reshape(DT_adb,1,length(DT_adb))    #（1， 9）
        
    else var_number == 2 
        DT_adb = vcat(aFloat,b)                         # vector (9,)
        DT_adb_mat =  reshape(DT_adb,1,length(DT_adb))    #（1， 9）
    end

    return DT_adb_mat, cart_model
end

function warm_start_DT_params(a,b,c,d,t,node,Tl)
    if node isa Leaf
        t_leaf = t
        while !(t_leaf in Tl)
            t_leaf = 2*t_leaf+1
        end
        #println("$t_leaf from $t")
        c[t_leaf] = node.majority
    else
        a[node.featid, t] = 1
        b[t] = node.featval
        d[t] = 1
        a,b,c,d = warm_start_DT_params(a,b,c,d,2*t, node.left, Tl)
        a,b,c,d = warm_start_DT_params(a,b,c,d,2*t+1, node.right, Tl)
    end
    return a,b,c,d
end

function encode_b(a, b, sorted_X)
    # find the index of b in splits
    p = length(sorted_X)
    a_number = Array(Array(1:p)') * a # 1*(SB)
    size_branch = length(b)
    new_b = zeros(size_branch)
    len_splits = [length(i)-1 for i in sorted_X]
    for i in 1:size_branch
        if a_number[i] == 0
            new_b[i] = 0.0
        else
            p_X = sorted_X[Int(a_number[i])]
            index = searchsortedfirst(p_X, b[i]) - 2
            if index == -1
                index = 0
            end
            new_b[i] = index/len_splits[Int(a_number[i])]
        end
    end
    return new_b
end

end 
