module oct_gpu
using LinearAlgebra, Random, StatsBase
using TimerOutputs: @timeit, get_timer
using CUDA


export OCT_gpu,
       OCT_regression_gpu,
       OCT_test,
       get_UB,
       gpu_init,
       regression_gpu_init,
       args_pre,
       regression_args_pre,
       OCT,
       split_x,
       onehot,
       trans_params_azd

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

function gpu_regression_stats!(
    counts_d,
    sums_d,
    squared_sums_d,
    X_d,
    a_d,
    b_d,
    d_d,
    NP,
    Y_d,
    stride,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n, p = size(X_d)
    combined_leaf_count = size(counts_d, 2)
    leaf_count = combined_leaf_count ÷ NP
    branch_count = leaf_count - 1
    sample_group_count = (n + stride - 1) ÷ stride

    if index <= sample_group_count
        for column in 1:combined_leaf_count
            @inbounds counts_d[index, column] = 0.0
            @inbounds sums_d[index, column] = 0.0
            @inbounds squared_sums_d[index, column] = 0.0
        end

        first_sample = (index - 1) * stride + 1
        last_sample = min(index * stride, n)
        for candidate in 1:NP
            branch_offset = (candidate - 1) * branch_count
            leaf_offset = (candidate - 1) * leaf_count
            for sample in first_sample:last_sample
                node = 1
                while node <= branch_count
                    branch_position = branch_offset + node
                    if !d_d[branch_position]
                        node = 2 * node + 1
                    else
                        found_feature = false
                        for feature in 1:p
                            if a_d[feature, branch_position]
                                node = if X_d[sample, feature] < b_d[branch_position]
                                    2 * node
                                else
                                    2 * node + 1
                                end
                                found_feature = true
                                break
                            end
                        end
                        if !found_feature
                            node = 2 * node + 1
                        end
                    end
                end

                leaf_position = leaf_offset + node - branch_count
                target = Y_d[sample]
                @inbounds counts_d[index, leaf_position] += 1.0
                @inbounds sums_d[index, leaf_position] += target
                @inbounds squared_sums_d[index, leaf_position] += target * target
            end
        end
    end
    return
end

function regression_gpu_init(verbose=false)
    start = time()
    n, p, leaf_count, population_size = 64, 4, 4, 4
    branch_count = leaf_count - 1
    X_d = CUDA.rand(Float32, n, p)
    Y_d = CUDA.rand(Float32, n)
    a = falses(p, branch_count * population_size)
    for column in axes(a, 2)
        a[mod1(column, p), column] = true
    end
    a_d = CuArray(a)
    b_d = CUDA.rand(Float32, 1, branch_count * population_size)
    d_d = reshape(sum(a_d; dims=1) .> 0, 1, :)
    counts_d = CUDA.zeros(Float32, n, leaf_count * population_size)
    sums_d = similar(counts_d)
    squared_sums_d = similar(counts_d)
    kernel = @cuda launch=false gpu_regression_stats!(
        counts_d,
        sums_d,
        squared_sums_d,
        X_d,
        a_d,
        b_d,
        d_d,
        population_size,
        Y_d,
        1,
    )
    configuration = launch_configuration(kernel.fun)
    threads = min(n, configuration.threads)
    blocks = cld(n, threads)
    kernel(
        counts_d,
        sums_d,
        squared_sums_d,
        X_d,
        a_d,
        b_d,
        d_d,
        population_size,
        Y_d,
        1;
        threads,
        blocks,
    )
    CUDA.synchronize()
    verbose && println("regression_gpu_init time: ", time() - start)
    return kernel, threads
end

function regression_args_pre(
    tree_da,
    X_d,
    X_cpu,
    Y,
    tree_size,
    Nmin,
    kernel,
    threads,
    population_size,
    alpha=0.0,
    verbose=false,
)
    n = size(X_cpu, 1)
    leaf_count = cld(tree_size, 2)
    branch_count = fld(tree_size, 2)
    stride = clamp(fld(n, threads * 108), 1, 100)
    maximum_buffer_bytes = 2 * 1024^3

    partial_row_count = cld(n, stride)
    bytes_per_candidate =
        partial_row_count * leaf_count * 3 * sizeof(Float32)
    while bytes_per_candidate > maximum_buffer_bytes && stride < n
        stride = min(2 * stride, n)
        partial_row_count = cld(n, stride)
        bytes_per_candidate =
            partial_row_count * leaf_count * 3 * sizeof(Float32)
    end

    maximum_population_stride = max(1, fld(maximum_buffer_bytes, bytes_per_candidate))
    divisors = [
        value for value in 1:population_size if
        population_size % value == 0 && value <= maximum_population_stride
    ]
    population_stride = isempty(divisors) ? 1 : maximum(divisors)
    counts_d = CUDA.zeros(
        Float32,
        partial_row_count,
        leaf_count * population_stride,
    )
    sums_d = similar(counts_d)
    squared_sums_d = similar(counts_d)
    splits, sorted_X = split_x(X_cpu)
    targets = Float32.(Y)
    target_offset = sum(targets) / Float32(length(targets))
    centered_Y = targets .- target_offset
    total_sse = sum(abs2, centered_Y)
    invalid_leaf_penalty =
        total_sse + Float32(alpha) * Float32(branch_count) + 1.0f0

    verbose && println("regression stride: ", stride)
    verbose && println("regression population stride: ", population_stride)
    return (
        tree_da=tree_da,
        X_d=X_d,
        Y_d=CuArray(centered_Y),
        tree_size=tree_size,
        min_samples_leaf=Nmin,
        kernel=kernel,
        threads=threads,
        alpha=alpha,
        counts_d=counts_d,
        sums_d=sums_d,
        squared_sums_d=squared_sums_d,
        stride=stride,
        population_stride=population_stride,
        splits=splits,
        sorted_X=sorted_X,
        total_sse=total_sse,
        invalid_leaf_penalty=invalid_leaf_penalty,
    )
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

function OCT_regression_gpu(candidates, arguments)
    population_size = size(candidates, 1)
    branch_count = fld(arguments.tree_size, 2)
    leaf_count = cld(arguments.tree_size, 2)
    population_stride = arguments.population_stride
    population_size % population_stride == 0 || throw(ArgumentError(
        "population size must be divisible by the GPU population stride",
    ))

    candidate_matrix = Array(candidates')
    a_d, b_d, d_d, active_counts, decoded_ab = trans_params_azs(
        candidate_matrix,
        size(arguments.X_d, 2),
        branch_count,
        population_size,
        arguments.splits,
    )
    costs = zeros(Float32, population_size)
    blocks = cld(size(arguments.counts_d, 1), arguments.threads)

    for chunk in 1:(population_size ÷ population_stride)
        first_candidate = (chunk - 1) * population_stride + 1
        last_candidate = chunk * population_stride
        first_branch = (first_candidate - 1) * branch_count + 1
        last_branch = last_candidate * branch_count
        chunk_a_d = a_d[:, first_branch:last_branch]
        chunk_b_d = b_d[:, first_branch:last_branch]
        chunk_d_d = d_d[:, first_branch:last_branch]

        arguments.kernel(
            arguments.counts_d,
            arguments.sums_d,
            arguments.squared_sums_d,
            arguments.X_d,
            chunk_a_d,
            chunk_b_d,
            chunk_d_d,
            population_stride,
            arguments.Y_d,
            arguments.stride;
            threads=arguments.threads,
            blocks=blocks,
        )
        CUDA.synchronize()

        counts = vec(Array(sum(arguments.counts_d; dims=1)))
        sums = vec(Array(sum(arguments.sums_d; dims=1)))
        squared_sums = vec(Array(sum(arguments.squared_sums_d; dims=1)))
        for local_candidate in 1:population_stride
            candidate = first_candidate + local_candidate - 1
            first_leaf = (local_candidate - 1) * leaf_count + 1
            last_leaf = local_candidate * leaf_count
            sse = 0.0f0
            violations = 0
            for leaf in first_leaf:last_leaf
                count = counts[leaf]
                if count > 0.0f0
                    sse += max(
                        0.0f0,
                        squared_sums[leaf] - sums[leaf]^2 / count,
                    )
                    violations += count < arguments.min_samples_leaf
                end
            end
            costs[candidate] =
                sse +
                Float32(arguments.alpha) * Float32(active_counts[candidate]) +
                arguments.invalid_leaf_penalty * violations
        end
    end

    return costs, findmin(costs), decoded_ab
end

# Optimized Float32 regression pipeline. The legacy entry point above remains
# available for compatibility while the package backends use these routines.

function gpu_decode_regression_candidates!(
    feature_ids,
    thresholds,
    candidates,
    split_values,
    split_offsets,
    split_lengths,
    split_lower_indices,
    split_upper_indices,
    branch_count,
    population_size,
)
    candidate = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if candidate <= population_size
        feature_count = length(split_lengths)
        for node in 1:branch_count
            feature = Int32(floor(candidates[candidate, node]))
            if feature < 1 || feature > feature_count ||
               (node > 1 && feature_ids[fld(node, 2), candidate] == 0)
                feature_ids[node, candidate] = 0
                thresholds[node, candidate] = 0.0f0
                continue
            end

            feature_ids[node, candidate] = feature
            lower_index = split_lower_indices[feature]
            upper_index = split_upper_indices[feature]
            split_count = upper_index - lower_index + 1
            encoded_threshold = clamp(
                candidates[candidate, branch_count + node],
                0.0f0,
                1.0f0,
            )
            local_split_index = if encoded_threshold >= 1.0f0
                split_count
            else
                Int32(floor(encoded_threshold * Float32(split_count))) + 1
            end
            local_split_index = clamp(
                local_split_index,
                Int32(1),
                split_count,
            )
            split_index = lower_index + local_split_index - 1
            thresholds[node, candidate] = split_values[
                split_offsets[feature] + split_index - 1
            ]
        end
    end
    return
end

function gpu_regression_partial_stats_direct!(
    partial_counts,
    partial_moments,
    X,
    Y,
    row_indices,
    feature_ids,
    thresholds,
    sample_count,
    group_count,
    stride,
    branch_count,
    leaf_count,
    population_stride,
    chunk_start,
    target_offset,
)
    group = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    local_candidate = blockIdx().y
    if group <= group_count && local_candidate <= population_stride
        candidate = chunk_start + local_candidate - 1
        leaf_offset = (local_candidate - 1) * leaf_count
        moment_span = leaf_count * population_stride

        for leaf in 1:leaf_count
            position = leaf_offset + leaf
            @inbounds partial_counts[group, position] = Int32(0)
            @inbounds partial_moments[group, position] = 0.0f0
            @inbounds partial_moments[group, moment_span + position] = 0.0f0
        end

        first_local_sample = (group - 1) * stride + 1
        last_local_sample = min(group * stride, sample_count)
        for local_sample in first_local_sample:last_local_sample
            sample = row_indices[local_sample]
            node = 1
            while node <= branch_count
                feature = feature_ids[node, candidate]
                if feature > 0 && X[sample, feature] < thresholds[node, candidate]
                    node = 2 * node
                else
                    node = 2 * node + 1
                end
            end

            leaf = node - branch_count
            position = leaf_offset + leaf
            target = Y[sample] - target_offset
            @inbounds partial_counts[group, position] += Int32(1)
            @inbounds partial_moments[group, position] += target
            @inbounds partial_moments[group, moment_span + position] += target * target
        end
    end
    return
end

function gpu_reduce_regression_stats!(
    leaf_counts,
    leaf_moments,
    partial_counts,
    partial_moments,
    group_count,
    leaf_count,
    population_stride,
    chunk_start,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    chunk_leaf_count = leaf_count * population_stride
    if index <= chunk_leaf_count
        local_candidate = fld(index - 1, leaf_count) + 1
        leaf = mod(index - 1, leaf_count) + 1
        moment_span = chunk_leaf_count
        count_value = Int32(0)
        sum_value = 0.0f0
        squared_sum_value = 0.0f0
        for group in 1:group_count
            @inbounds count_value += partial_counts[group, index]
            @inbounds sum_value += partial_moments[group, index]
            @inbounds squared_sum_value +=
                partial_moments[group, moment_span + index]
        end

        candidate = chunk_start + local_candidate - 1
        output_position = (candidate - 1) * leaf_count + leaf
        @inbounds leaf_counts[output_position] = count_value
        @inbounds leaf_moments[1, output_position] = sum_value
        @inbounds leaf_moments[2, output_position] = squared_sum_value
    end
    return
end

function gpu_regression_costs!(
    costs,
    leaf_counts,
    leaf_moments,
    feature_ids,
    leaf_count,
    branch_count,
    population_stride,
    chunk_start,
    min_samples_leaf,
    alpha,
    invalid_leaf_penalty,
)
    local_candidate = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if local_candidate <= population_stride
        candidate = chunk_start + local_candidate - 1
        sse = 0.0f0
        violations = Int32(0)
        for leaf in 1:leaf_count
            position = (candidate - 1) * leaf_count + leaf
            count_value = leaf_counts[position]
            if count_value > 0
                sum_value = leaf_moments[1, position]
                squared_sum_value = leaf_moments[2, position]
                leaf_sse = squared_sum_value -
                           sum_value * sum_value / Float32(count_value)
                sse += max(0.0f0, leaf_sse)
                violations += count_value < min_samples_leaf
            end
        end

        active_count = Int32(0)
        for node in 1:branch_count
            active_count += feature_ids[node, candidate] > 0
        end
        costs[candidate] =
            sse +
            alpha * Float32(active_count) +
            invalid_leaf_penalty * Float32(violations)
    end
    return
end

function gpu_regression_best_index!(best_index, costs, population_size)
    if blockIdx().x == 1 && threadIdx().x == 1
        selected = Int32(1)
        selected_cost = costs[1]
        for candidate in 2:population_size
            cost = costs[candidate]
            if cost < selected_cost
                selected = Int32(candidate)
                selected_cost = cost
            end
        end
        best_index[1] = selected
    end
    return
end

function gpu_regression_trials!(
    trials,
    population,
    best_index,
    first_parents,
    second_parents,
    mutation_scales,
    crossover,
    population_size,
    state_size,
    branch_count,
    feature_count,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= population_size * state_size
        candidate = mod(index - 1, population_size) + 1
        variable = fld(index - 1, population_size) + 1
        value = population[candidate, variable]
        if crossover[candidate, variable]
            value =
                population[best_index[1], variable] +
                mutation_scales[candidate] * (
                    population[first_parents[candidate], variable] -
                    population[second_parents[candidate], variable]
                )
        end

        if variable <= branch_count
            value = floor(clamp(value, 0.0f0, Float32(feature_count)))
            if variable == 1
                value = max(1.0f0, value)
            end
        else
            value = clamp(value, 0.0f0, 1.0f0)
        end
        trials[candidate, variable] = value
    end
    return
end

function gpu_regression_select_population!(
    population,
    trials,
    costs,
    trial_costs,
    population_size,
    state_size,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= population_size * state_size
        candidate = mod(index - 1, population_size) + 1
        variable = fld(index - 1, population_size) + 1
        if trial_costs[candidate] <= costs[candidate]
            population[candidate, variable] = trials[candidate, variable]
        end
    end
    return
end

function gpu_regression_select_costs!(costs, trial_costs, population_size)
    candidate = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if candidate <= population_size && trial_costs[candidate] <= costs[candidate]
        costs[candidate] = trial_costs[candidate]
    end
    return
end

function gpu_regression_level_masks!(
    selected,
    X,
    path_features,
    path_thresholds,
    path_directions,
    level,
    node_count,
)
    sample = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    local_node = blockIdx().y
    if sample <= size(X, 1) && local_node <= node_count
        keep = true
        for step in 1:level
            feature = path_features[step, local_node]
            goes_left = feature > 0 &&
                        X[sample, feature] < path_thresholds[step, local_node]
            if goes_left != path_directions[step, local_node]
                keep = false
                break
            end
        end
        selected[sample, local_node] = keep
    end
    return
end

function regression_gpu_workspace(
    maximum_sample_count,
    depth,
    population_size,
    feature_count;
    maximum_buffer_bytes=2 * 1024^3,
)
    branch_count = 2^depth - 1
    leaf_count = 2^depth
    state_size = 2 * branch_count
    threads = 256
    multiprocessor_count = attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
    maximum_partial_rows = min(
        maximum_sample_count,
        2 * threads * multiprocessor_count,
    )
    bytes_per_candidate = maximum_partial_rows * leaf_count * (
        sizeof(Int32) + 2 * sizeof(Float32)
    )
    maximum_population_stride = max(
        1,
        fld(maximum_buffer_bytes, max(1, bytes_per_candidate)),
    )
    divisors = [
        value for value in 1:population_size if
        population_size % value == 0 && value <= maximum_population_stride
    ]
    population_stride = isempty(divisors) ? 1 : maximum(divisors)

    return (
        depth=depth,
        branch_count=branch_count,
        leaf_count=leaf_count,
        state_size=state_size,
        population_size=population_size,
        feature_count=feature_count,
        threads=threads,
        multiprocessor_count=multiprocessor_count,
        maximum_partial_rows=maximum_partial_rows,
        population_stride=population_stride,
        partial_counts=CUDA.zeros(
            Int32,
            maximum_partial_rows,
            leaf_count * population_stride,
        ),
        partial_moments=CUDA.zeros(
            Float32,
            maximum_partial_rows,
            2 * leaf_count * population_stride,
        ),
        leaf_counts=CUDA.zeros(Int32, leaf_count * population_size),
        leaf_moments=CUDA.zeros(Float32, 2, leaf_count * population_size),
        feature_ids=CUDA.zeros(Int32, branch_count, population_size),
        thresholds=CUDA.zeros(Float32, branch_count, population_size),
        population=CUDA.zeros(Float32, population_size, state_size),
        trials=CUDA.zeros(Float32, population_size, state_size),
        costs=CUDA.zeros(Float32, population_size),
        trial_costs=CUDA.zeros(Float32, population_size),
        best_index=CUDA.ones(Int32, 1),
        first_parents=CUDA.ones(Int32, population_size),
        second_parents=CUDA.ones(Int32, population_size),
        mutation_scales=CUDA.zeros(Float32, population_size),
        crossover=CUDA.zeros(Bool, population_size, state_size),
    )
end

function regression_gpu_split_arrays(splits)
    split_values = Float32[]
    split_offsets = Vector{Int32}(undef, length(splits))
    split_lengths = Vector{Int32}(undef, length(splits))
    for feature in eachindex(splits)
        split_offsets[feature] = Int32(length(split_values) + 1)
        feature_splits = Float32.(splits[feature])
        split_lengths[feature] = Int32(length(feature_splits))
        append!(split_values, feature_splits)
    end
    return CuArray(split_values), CuArray(split_offsets), CuArray(split_lengths)
end

function regression_gpu_fitness!(
    workspace,
    candidates,
    costs,
    X,
    Y,
    row_indices,
    split_values,
    split_offsets,
    split_lengths,
    split_lower_indices,
    split_upper_indices,
    target_offset,
    total_sse,
    min_samples_leaf,
    alpha,
)
    sample_count = length(row_indices)
    stride = clamp(
        fld(
            sample_count,
            workspace.threads * workspace.multiprocessor_count,
        ),
        1,
        100,
    )
    group_count = cld(sample_count, stride)
    group_count <= workspace.maximum_partial_rows || error(
        "regression GPU workspace requires $(group_count) partial rows but has " *
        "$(workspace.maximum_partial_rows)",
    )

    @cuda threads=workspace.threads blocks=cld(
        workspace.population_size,
        workspace.threads,
    ) gpu_decode_regression_candidates!(
        workspace.feature_ids,
        workspace.thresholds,
        candidates,
        split_values,
        split_offsets,
        split_lengths,
        split_lower_indices,
        split_upper_indices,
        workspace.branch_count,
        workspace.population_size,
    )

    invalid_leaf_penalty =
        Float32(total_sse) +
        Float32(alpha) * Float32(workspace.branch_count) +
        1.0f0
    for chunk_start in 1:workspace.population_stride:workspace.population_size
        @cuda threads=workspace.threads blocks=(
            cld(group_count, workspace.threads),
            workspace.population_stride,
        ) gpu_regression_partial_stats_direct!(
            workspace.partial_counts,
            workspace.partial_moments,
            X,
            Y,
            row_indices,
            workspace.feature_ids,
            workspace.thresholds,
            sample_count,
            group_count,
            stride,
            workspace.branch_count,
            workspace.leaf_count,
            workspace.population_stride,
            chunk_start,
            Float32(target_offset),
        )
        @cuda threads=workspace.threads blocks=cld(
            workspace.leaf_count * workspace.population_stride,
            workspace.threads,
        ) gpu_reduce_regression_stats!(
            workspace.leaf_counts,
            workspace.leaf_moments,
            workspace.partial_counts,
            workspace.partial_moments,
            group_count,
            workspace.leaf_count,
            workspace.population_stride,
            chunk_start,
        )
        @cuda threads=workspace.threads blocks=cld(
            workspace.population_stride,
            workspace.threads,
        ) gpu_regression_costs!(
            costs,
            workspace.leaf_counts,
            workspace.leaf_moments,
            workspace.feature_ids,
            workspace.leaf_count,
            workspace.branch_count,
            workspace.population_stride,
            chunk_start,
            min_samples_leaf,
            Float32(alpha),
            invalid_leaf_penalty,
        )
    end
    return costs
end

function regression_gpu_set_best!(workspace)
    @cuda threads=1 blocks=1 gpu_regression_best_index!(
        workspace.best_index,
        workspace.costs,
        workspace.population_size,
    )
    return workspace.best_index
end

function regression_gpu_trial_population!(
    workspace,
    first_parents,
    second_parents,
    mutation_scales,
    crossover,
)
    copyto!(workspace.first_parents, first_parents)
    copyto!(workspace.second_parents, second_parents)
    copyto!(workspace.mutation_scales, mutation_scales)
    copyto!(workspace.crossover, crossover)
    @cuda threads=workspace.threads blocks=cld(
        workspace.population_size * workspace.state_size,
        workspace.threads,
    ) gpu_regression_trials!(
        workspace.trials,
        workspace.population,
        workspace.best_index,
        workspace.first_parents,
        workspace.second_parents,
        workspace.mutation_scales,
        workspace.crossover,
        workspace.population_size,
        workspace.state_size,
        workspace.branch_count,
        workspace.feature_count,
    )
    return workspace.trials
end

function regression_gpu_select!(workspace)
    @cuda threads=workspace.threads blocks=cld(
        workspace.population_size * workspace.state_size,
        workspace.threads,
    ) gpu_regression_select_population!(
        workspace.population,
        workspace.trials,
        workspace.costs,
        workspace.trial_costs,
        workspace.population_size,
        workspace.state_size,
    )
    @cuda threads=workspace.threads blocks=cld(
        workspace.population_size,
        workspace.threads,
    ) gpu_regression_select_costs!(
        workspace.costs,
        workspace.trial_costs,
        workspace.population_size,
    )
    regression_gpu_set_best!(workspace)
    return workspace.population
end

function regression_gpu_level_masks!(
    selected,
    X,
    path_features,
    path_thresholds,
    path_directions,
    level,
    node_count,
)
    threads = 256
    @cuda threads=threads blocks=(cld(size(X, 1), threads), node_count) gpu_regression_level_masks!(
        selected,
        X,
        path_features,
        path_thresholds,
        path_directions,
        level,
        node_count,
    )
    return selected
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
