module oct_gpu
using LinearAlgebra, Random, StatsBase
using TimerOutputs: @timeit, get_timer
using CUDA


export OCT_gpu, OCT_test, get_UB, gpu_init, args_pre, OCT, split_x, onehot, trans_params_azd

# tree_a_with_zeros_discrete
function trans_params_azd(candidate, p, size_branch, splits)
    size_branch = Int(size_branch)
    can_as = Int.(candidate[1:size_branch])
    a = onehot(can_as, p).>0 # P*(SB)
    a_number = reshape(can_as, 1, size_branch) # Array(Array(1:p)') * a # 1*(SB)
    len_splits = [length(i) for i in splits] 
    b = candidate[size_branch+1:size_branch*2] # SB
    # println(a_number)
    for i in 1:size_branch
        try
            if a_number[i] == 0
                b[i] = 0
            elseif b[i] == 1 
                b[i] = splits[a_number[i]][len_splits[a_number[i]]]
            else
                b[i] = splits[a_number[i]][floor(Int, b[i]*len_splits[a_number[i]])+1]
            end
        catch
            println("a_number[i]: ", a_number[i])
            println("original_b[i]: ", b[i])
            b[i] = 0
        end
    end
    d = sum(a, dims=1) .> 0
    for t in 1:size_branch
        # (5) enforce the hierarchical structure of the tree:
        if t>1 &&  d[t]>d[floor.(Int, t/2)]
            d[t]=d[floor.(Int, t/2)]
        end
    end
    return a, b, d
end

# tree_a_with_zeros
function trans_params_azs(candidates, p, size_branch, NP, splits)
    size_branch = Int(size_branch)
    can_as = Int.(candidates[1:size_branch, :]) # SB*NP
    as = onehot(can_as, p).>0 # P*(SB*NP)
    as_number = can_as
    # as_number = reshape(as_number, size_branch, NP)
    len_splits = [length(i) for i in splits]
    # println("len_splits: ", len_splits)
    # println("as_number: ", as_number)
    bs = Float32.(candidates[size_branch+1:size_branch*2, :]) # SB*NP
    bs_number = zeros(Int64, size_branch*NP)
    # bs[i]: [0,1] => [1:len_splits[as_number[i]]]
    for i in 1:size_branch*NP
        if as_number[i] == 0
            bs[i] = 0
        elseif bs[i] == 1 
            bs_number[i] = len_splits[as_number[i]]
            bs[i] = splits[as_number[i]][bs_number[i]]
        else
            bs_number[i] = floor(Int, bs[i]*len_splits[as_number[i]])+1
            bs[i] = splits[as_number[i]][bs_number[i]]
        end
    end
    bs = reshape(bs, 1, size_branch * NP) # 1*(SB*NP)
    bs_number = reshape(bs_number, size_branch, NP) # SB*NP
    ds = sum(as, dims=1) .> 0 # 1*(SB*NP)
    sum_ds = zeros(NP)
    for i in 1:NP
        d = view(ds, :, (i-1)*size_branch+1:i*size_branch)
        for t in 1:size_branch
            # (5) enforce the hierarchical structure of the tree:
            if t>1 && d[t] > d[floor.(Int, t/2)]
                d[t] = d[floor.(Int, t/2)]
            end
        end
        sum_ds[i] = sum(d)
    end
    return CuArray(as), CuArray(bs), CuArray(ds), sum_ds, vcat(as_number, bs_number)
end

# obtain all the possible splits on all the features
function split_x(X)
    n, p = size(X)
    splits = []
    sorted_X = []
    for i in 1:p
        cur_splits = [0; sort(unique(X[:, i]))] # assuming X in [0, 1]
        push!(sorted_X, cur_splits)
        # split = item1 + item2 / 2
        cur_splits = [(cur_splits[i] + cur_splits[i+1]) / 2 for i in 1:length(cur_splits)-1]
        push!(splits, cur_splits)
    end

    return splits, sorted_X
end

# julia test_gpu/test_warmstart_LayerOriginal.jl 63 63 2 OCT_az 0 1 2 0 0 0 50
function args_pre(num_classes, tree_da, X, X_cpu, Y, tree_size, Nmin, kernels, threadss, NP=1, alpha=0.0, verbose=false)
    fun_args = []
    push!(fun_args, tree_da) # 1
    if X isa CuArray{Float32,2}
        push!(fun_args, X) # 2
    else
        push!(fun_args, CuArray(Float32.(X))) # 2
    end
    push!(fun_args, CuArray(Int32.(Y))) # 3
    push!(fun_args, num_classes) # 4
    push!(fun_args, tree_size) # 5
    push!(fun_args, Nmin) # 6
    push!(fun_args, kernels) # 7, [gpu_matrix_3!, _get_data_gpu!]
    push!(fun_args, threadss) # 8
    verbose && println("threadss: ", threadss)
    push!(fun_args, alpha) # 9
    verbose && println("alpha: ", alpha)
    n = size(X, 1)
    threads = threadss[1]
    size_leaves = ceil(Int32, tree_size/2)
    stride = floor(Int32, n/(threads*108)) # A100 has 108 SMs
    if stride < 1
        stride = 1
    elseif stride > 100
        stride = 100
    end
    verbose && println("stride: ", stride)
    blocks = Int(cld(n/stride, threads))
    # 20GB memory limit -> 20*1024*1024*1024bits
    if size_leaves > 128
        z_d_size_limit = 10*8000000000 # 5*8000000000  # 20*8000000000 
    else
        z_d_size_limit = 20*8000000000 # 5*8000000000  # 20*8000000000 
    end
    NP_stride = floor(Int32, z_d_size_limit/(threads*blocks*size_leaves*num_classes*32*2))*2
    verbose && println("block size: ", blocks)
    verbose && println("dataset size n:", n)
    verbose && println("NP_stride before check: ", NP_stride)
    if NP_stride < 1
        NP_stride = 1
    elseif NP_stride >= NP
        NP_stride = NP
    else 
        # NPs: all the integers that can divide NP
        NPs = [i for i in 1:NP if NP%i==0]
        # NP_stride -> nearest integer smaller than NP_stride in NPs
        NP_stride = findfirst(NP_stride.<NPs)
        NP_stride = NPs[NP_stride-1]
    end
    verbose && println("NP_stride after check: ", NP_stride)
    if size_leaves > 128
        verbose && CUDA.memory_status()
        z_d = nothing # D8P7,P8,  huge
        @timeit get_timer("Shared") "gc" GC.gc(true) # D8P7,P8, huge
        verbose && println(" ")
        verbose && CUDA.memory_status()
        verbose && println(" ")
    end
    z_d = CUDA.zeros(Float32, threads*blocks, size_leaves*NP_stride, num_classes)
    push!(fun_args, z_d) # 10
    push!(fun_args, stride) # 11
    push!(fun_args, NP_stride) # 12
    @timeit get_timer("Shared") "split_x" splits, sorted_X = split_x(X_cpu)
    push!(fun_args, splits) # 13
    push!(fun_args, sorted_X) # 14

    return fun_args
end

function gpu_init(verbose=false)
    start = time()
    n, p, num_classes, tree_size, Nmin, NP = 4215, 100, 10, 31, 1, 100
    stride = 1
    X_d = CUDA.rand(n, p)
    size_branch = floor(Int, tree_size/2)
    size_leaves = ceil(Int32, tree_size/2)
    a = falses(p, size_branch*NP)
    for i in 1:size_branch*NP
        a[ceil(Int, rand()*p),i] = true
    end
    b = zeros(Float32, 1, size_branch*NP)
    a_d = CuArray(a)
    b_d = CuArray(b)
    z_d = CUDA.zeros(n, size_leaves*NP, num_classes).<0
    Y = CuArray(ceil.(Int32, rand(n)*num_classes))
    
    # gpu_matrix_3!(z_d, X_d, a_d, b_d, d_d, NP, Y, stride)
    z_d = CUDA.zeros(n, size_leaves*NP, num_classes)
    d_d = sum(a_d, dims=1) .> 0 # 1*(SB*NP)
    shmem = p*size_branch*sizeof(Bool)+1*size_branch*sizeof(Bool)#+1*size_branch*sizeof(Float32)
    kernel4 = @cuda launch=false gpu_matrix_3!(z_d, X_d, a_d, b_d, d_d, NP, Y, stride)
    config4 = launch_configuration(kernel4.fun)
    threads = min(n, config4.threads)
    blocks = Int(cld(n/stride, threads))
    threads4 = copy(threads)
    verbose && println("shmem_size: ", shmem, " Bits", ", ", shmem/(1024), " KB", ", config4.threads: ", config4.threads, ", config4.blocks: ", config4.blocks)
    kernel4(z_d, X_d, a_d, b_d, d_d, NP, Y, stride; threads, blocks)

    # _get_data_gpu!(selected, X_d, a_d, vec(b_d), d_d, ancesters, n, size_ancs)
    ancesters = CuArray(Int32.([1, 2, 5]))
    size_ancs = length(ancesters)
    d = sum(a, dims=1) .> 0
    d_d = CuArray(d)
    selected = CUDA.zeros(n).>0
    kernel5 = @cuda launch=false _get_data_gpu!(selected, X_d, a_d, vec(b_d), d_d, ancesters, n, size_ancs)
    config5 = launch_configuration(kernel5.fun)
    threads = min(n, config5.threads)
    blocks = cld(n, threads)
    threads5 = copy(threads)
    kernel5(selected, X_d, a_d, vec(b_d), d_d, ancesters, n, size_ancs; threads, blocks)
    verbose && println("config5.threads: ", config5.threads, ", config5.blocks: ", config5.blocks)

    kernels = [kernel4, kernel5]
    threadss = [threads4, threads5]

    verbose && println("gpu_init time: ", time()-start)
    return kernels, threadss
end

function _get_data_gpu!(selected, X, a, b, d, ancesters, n, size_ancs)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ~, P = size(X)
    if index < n
        for i in 1:(size_ancs-1) # 1,2
            if ancesters[i+1] == 2*ancesters[i] # left_node, ax<b
                if d[ancesters[i]] == false
                    selected[index] = false
                else
                    if selected[index]
                        for p in 1:P
                            if a[p, ancesters[i]] == true
                                if X[index, p] >= b[ancesters[i]]
                                    selected[index] = false
                                    break
                                end # end if
                            end # end if 
                        end # end for 
                    end # end if 
                end # end if
            else # right_node, ax>=b
                if d[ancesters[i]] == true
                    if selected[index]
                        for p in 1:P
                            if a[p, ancesters[i]] == true
                                if X[index, p] < b[ancesters[i]]
                                    selected[index] = false
                                    break
                                end # end if
                            end # end if
                        end # end for
                    end # end if 
                end # end if
            end  # end if         
        end # end for
    end  # end if
end

function gpu_matrix_3!(z_d, X_d, a_d, b_d, d_d, NP, Y, stride)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_d, size_leaves_d, ~ = size(z_d)
    size_leaves_d = Int32(size_leaves_d/NP)
    size_branch_d = size_leaves_d-1
    n_d, p_d = size(X_d)
    p_d_1 = p_d+1
    # considering NP
    if index <= ceil(n_d/stride)
        strides = (index-1)*stride+1:min(index*stride, n_d)
        z_d[index, :, :] .= 0.0
        for np = 1:NP
            np_offset = (np-1)*size_branch_d
            np_offset_leaves = (np-1)*size_leaves_d - size_branch_d
            for s in strides
                t = Int32(1)
                while t<=size_branch_d
                    np_offset_t = np_offset + t
                    if d_d[np_offset_t] == false
                        t = t*2+1
                    else
                        @inbounds for i = 1:p_d_1
                            if i == p_d_1 # sum(a) == 0, to right, t = t*2+1
                                t = t*2+1
                                break
                            end
                            @inbounds if a_d[i, np_offset_t] # a==1, branch on ith feature
                                @inbounds t = X_d[s, i].<b_d[1, np_offset_t] ? t*2 : t*2+1
                                break
                            end # end if a_d[i, (np-1)*size_branch_d+t]
                        end # end for i
                    end # end if d_d[(np-1)*size_branch_d+t] == false
                end # end while true
                @inbounds z_d[index, np_offset_leaves+t, Y[s]] += 1.0
            end # end for s
        end # end for np
    end # end if index <= n_d
end

# gpu fitness function
function OCT_gpu(candidates, tree_da, X_d, Y_d, num_classes, tree_size, Nmin, kernels, threadss, alpha=0.0, z_d=nothing, stride=1, NP_stride=1, splits=nothing)
    # TREE STRUCTURE: ################################################################
    size_branch = floor(Int32, tree_size/2)
    size_leaves = ceil(Int32, tree_size/2)
    n, p = size(X_d)
    NP, state_size = size(candidates)
    @timeit get_timer("Shared") "tree structure" begin
        candidates = Array(candidates') # state_size * NP
        as_d, bs_d, ds_d, sum_ds, decoded_ab = trans_params_azs(candidates, p, size_branch, NP, splits)
    end # end of @timeit get_timer("Shared") "tree structure"

    @timeit get_timer("Shared") "class cost" begin
        NCT = []
        threads = min(n, threadss[1])
        blocks = Int(cld(n/stride, threads))
        for np in 1:ceil(Int32, NP/NP_stride) # assume NP is multiple of NP_stride
            a_d = as_d[:, (np-1)*size_branch*NP_stride+1:min(np*size_branch*NP_stride, size(as_d,2))] # P * SB*NP_stride
            b_d = bs_d[:, (np-1)*size_branch*NP_stride+1:min(np*size_branch*NP_stride, size(bs_d,2))] # 1 * SB*NP_stride
            d_d = ds_d[:, (np-1)*size_branch*NP_stride+1:min(np*size_branch*NP_stride, size(ds_d,2))] # 1 * SB*NP_stride
            @timeit get_timer("Shared") "gpu" begin
                @timeit get_timer("Shared") "matmul+sample" begin
                    kernels[1](z_d, X_d, a_d, b_d, d_d, NP_stride, Y_d, stride; threads, blocks)
                    CUDA.synchronize()
                end
                @timeit get_timer("Shared") "sum" begin
                    Nct = sum(z_d, dims=1)[1,:,:] # 1*(SL*NP_stride)*k, Int64 -> SL*K
                    CUDA.synchronize()
                end
                @timeit get_timer("Shared") "NCT" begin
                    NCT = np == 1 ? Nct : vcat(NCT, Nct)
                    CUDA.synchronize()
                end 
            end # end of @timeit get_timer("Shared") "gpu"
        end # end of for np in 1:NP

        @timeit get_timer("Shared") "cpu" begin
            Nct = Array(NCT) # (SL*NP)*K
            Nt = sum(Nct, dims=2) # (SL*NP)*1, Int64
            Nmin_flag = (Nt .< Nmin) .& (Nt .> 0) # (SL*NP)*1
            Nct_max = maximum(Nct, dims=2) # (SL*NP)*1
            octCosts_d = zeros(Float64, NP)
            for np in 1:NP
                octCosts_d[np] = sum(view(Nt, (np-1)*size_leaves+1:np*size_leaves) - view(Nct_max, (np-1)*size_leaves+1:np*size_leaves)) + alpha * sum_ds[np] + Float64(sum(view(Nmin_flag, (np-1)*size_leaves+1:np*size_leaves)).>0)
            end
            results = findmin(octCosts_d)
        end # end of @timeit get_timer("Shared") "cpu"
    end # end of @timeit get_timer("Shared") "class cost"
    return octCosts_d, results, decoded_ab
end

# cpu fitness function
function OCT(tree_da, candidate, X, Y_K, classes, tree_size, Nmim, whatReturn, alpha=0.05, splits=nothing)
    # TREE STRUCTURE: ################################################################
    tree_depth = floor(Int64, log2(tree_size+1)-1)#tree depth
    branch_nodes = [trunc(Int64, x) for x in 1:floor(tree_size/2)]     #branch nodes
    leaf_nodes = [trunc(Int64, x) for x in floor(tree_size/2)+1:tree_size]   #leaf nodes
    size_branch = size(branch_nodes,1)
    size_leaf = size(leaf_nodes,1)
    n = size(X,1)
    p = size(X,2)

    K=size(classes,1)    #number of classes
    a, b, d = trans_params_azd(candidate, p, size_branch, splits)
    # Class Cost: ################################################################
    zs = X*a.<(b')
        z = falses(n,tree_size)   
        for i in 1:n                                   
            t=1                     #always starts in the root node
            while true
                @inbounds if d[t] && zs[i, t] # a1*x1+a2*x2+...+ap*xp < b
                    t = t*2
                    if t>size_branch
                        break
                    end
                else
                    t = t*2 + 1
                    if t>size_branch
                        break
                    end
                end
            end
            @inbounds z[i,t]=true
        end

    Nt = sum(z[:, size_branch+1:tree_size], dims=1) # 1*SL
    Nmin_flag = sum((Nt.<Nmim) .& (Nt.>0))
    Nct = Y_K*z[:, size_branch+1:tree_size] # K*N*N*SL = K * SL
    Nct, ct = findmax(Nct, dims=1)
    sumLt = sum(Nt - Nct) # sum of the cost of the leaf nodes
    octCost = sumLt

    if whatReturn==1 # used for debug case
        return d, a, [x[1] for x in ct]
    elseif whatReturn==2 
        return octCost
    elseif whatReturn==4 # used for training set in the test case
        return octCost, [x[1] for x in ct], Nmin_flag
    end

    # complexity of the tree:
    sumdt = sum(d) * alpha

    # optimal classification tree cost:
    octCost = octCost + sumdt
    if whatReturn==3 # used for validating minleafsize for local_search
        if Nmin_flag > 0 
            Nmin_flag = true
        else
            Nmin_flag = false
        end
        return octCost, Nmin_flag, z[:, size_branch+1:tree_size] 
    end

    return octCost
end

# cpu fitness function for test cases, without the update of c
function OCT_test(c, tree_da, candidate, X, Y, classes, tree_size, splits=nothing)
        # TREE STRUCTURE: ################################################################
        branch_nodes = [trunc(Int64, x) for x in 1:floor(tree_size/2)]     #branch nodes
        leaf_nodes = [trunc(Int64, x) for x in floor(tree_size/2)+1:tree_size]   #leaf nodes
        size_branch = size(branch_nodes,1)
        size_leaf = size(leaf_nodes,1)
        n = size(X,1)
        p = size(X,2)

        K=size(classes,1)    #number of classes
        a, b, d = trans_params_azd(candidate, p, size_branch, splits)
  
        z = falses(n,size_leaf)                      
        for i in 1:n                                   
            t=1                     #always starts in the root node
            while true
                if sum(a[:, t])>0 && d[t] && X[i, :][a[:, t]][1] < b[t]
                    t = t*2
                    if t>size_branch
                        break
                    end
                else
                    t = t*2 + 1
                    if t>size_branch
                        break
                    end
                end
            end
            z[i,t-size_branch]=true
        end
        octCost = 0.0
        for t in 1:size_leaf
            Nt = sum(z[:,t]) # Number of points at leaf node t
            Yt = Y[z[:,t]] 
            Nct = count(isequal(c[t]), Yt) # Number of points at leaf node t with label c[t]
            octCost = octCost + Nt - Nct # Cost of leaf node t
        end
    return octCost
end

function get_UB(cost_train, L_hat, candidate, tree_da, tree_size, p, alpha=0.05, splits=nothing)
    @timeit get_timer("Shared") "tree structure" begin
        # TREE STRUCTURE: ################################################################
        branch_nodes = [trunc(Int64, x) for x in 1:floor(tree_size/2)]     #branch nodes
        leaf_nodes = [trunc(Int64, x) for x in floor(tree_size/2)+1:tree_size]   #leaf nodes
        size_branch = size(branch_nodes,1)
        size_leaf = size(leaf_nodes,1)

        # classes=unique(Y)      #get a dataframe with the classes in the data
        a, b, d = trans_params_azd(candidate, p, size_branch, splits)
    end # end of @timeit get_timer("Shared") "tree structure"

    @timeit get_timer("Shared") "UB" begin
        UB = cost_train * (1/L_hat) + alpha * sum(d)
    end # end of @timeit get_timer("Shared") "UB"

    return UB
end

function onehot(s::AbstractMatrix, n_dims)
    x = zeros(eltype(s), n_dims, length(s))
    for j in eachindex(s)
        1 <= s[j] <= n_dims && (x[s[j], j] = one(eltype(s)))
    end
    return x
end

function onehot(s::AbstractVector, n_dims)
    x = zeros(eltype(s), n_dims, length(s))
    for j in eachindex(s)
        1 <= s[j] <= n_dims && (x[s[j], j] = one(eltype(s)))
    end
    return x
end

end # end of module
