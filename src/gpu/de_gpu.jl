module de_gpu
using LinearAlgebra, Random, StatsBase
using TimerOutputs: @timeit, get_timer
using CUDA

# Internal sibling modules provided by the MHDETrees package.
using ..warmstart_gpu, ..oct_gpu

export DEb1b_warmStart, layer_by_layer_original_warmStart


function DEb1b_warmStart(fun_args, tree_da, DE_iters, X, Y, tree_size, var_number, Nmin, Np::Int32=20,F=0.8,Cr=0.7, imprimir=0, xinits=nothing, seed=nothing, ws_flag=true, init_flag=0, verbose=false)
    if seed === nothing
        Random.seed!(42)
    else
        Random.seed!(seed)
    end
    @timeit get_timer("Shared") "part 1" begin
        tree_depth = floor(Int32, log2(tree_size+1)-1)
        size_leaves = ceil(Int32, tree_size/2)
        n, p = size(X)
        L = zeros(1,floor(Int32,tree_size/2)*var_number)
        U = ones(1,floor(Int32,tree_size/2)*var_number)
        U[1, 1:floor(Int32,tree_size/2)] .= float(p+1)
        if tree_da == 3 # tree_az, ensure that the first node is a split
            L[1] = 1.0
        end
        L_d = CuArray(L)
        U_d = CuArray(U)
        state_size = size(L, 2);             #dimensiones del problema
        cart_start = time()
        # Initializing population
        if ws_flag
            @timeit get_timer("Shared") "warm_start_DT"  DT_warmstart, cart_model = warmstart_gpu.warm_start_DT(X, Y, tree_da,var_number, Nmin, tree_depth, 0.0, fun_args[14])
            cart_time = time()-cart_start
        else # warm start disabled
            DT_warmstart = zeros(0, state_size)
            cart_model = nothing
            cart_time = 0
        end
        if init_flag == 1 # CART-LA return DT_warmstart
            return Array(DT_warmstart), cart_model, DT_warmstart, 1, cart_time
        end

        DT_warmstart_num = size(DT_warmstart, 1)
        verbose && println("DT_warmstart_num: ", DT_warmstart_num)
        RF_warmstart_num = 0 # size(RF_warmstart, 1)
        N_xinits = xinits === nothing ? 0 : length(xinits)
        states = rand(Float32,(Np-DT_warmstart_num-RF_warmstart_num-N_xinits,state_size));     # poblacion matrix
        states = vcat(states, DT_warmstart) #, RF_warmstart) 
        for i in 1:N_xinits
            states = vcat(states, reshape(xinits[i], 1, length(xinits[i])))
        end
        for ind=1:Np
            if ind <= Np-DT_warmstart_num-RF_warmstart_num-N_xinits
                states[[ind],:].= L.+states[[ind],:].*(U-L)
                for ind2 = 1:Int(state_size/2)
                    states[ind, ind2] = floor(states[ind, ind2])
                    if states[ind, ind2] == p+1
                        states[ind, ind2] = float(p)
                    end
                end
            end 
        end
        Np = size(states, 1)
        fitnesses, results, decoded_abs = OCT_gpu(states, fun_args[1], fun_args[2], fun_args[3], fun_args[4], fun_args[5], fun_args[6], fun_args[7], fun_args[8], fun_args[9], fun_args[10], fun_args[11], fun_args[12], fun_args[13])
        best_fitness = results[1]
        best_idx = Int32.(results[2])
        best_state = states[best_idx,:]
    end #part 1
    
    @timeit get_timer("Shared") "part 2" begin
        for iter=1:DE_iters
            @timeit get_timer("Shared") "update_states" begin
                if tree_size <= 127
                    new_states = update_states_cpu_matrix(states, best_state, L, U, Np, state_size, Cr, p)
                else
                    new_states = update_states_gpu(states, best_state, L_d, U_d, Np, state_size, Cr, p)
                end
            end

            @timeit get_timer("Shared") "OCT_gpu" new_fitnesses, results, decoded_ab = OCT_gpu(new_states, fun_args[1], fun_args[2], fun_args[3], fun_args[4], fun_args[5], fun_args[6], fun_args[7], fun_args[8], fun_args[9], fun_args[10], fun_args[11], fun_args[12], fun_args[13])
            
            @timeit get_timer("Shared") "update_best" begin
                replace_idx = new_fitnesses.<fitnesses
                fitnesses[replace_idx] = new_fitnesses[replace_idx]
                states[replace_idx,:] = new_states[replace_idx,:]
                new_best_fitness = results[1]
                new_best_idx = Int32.(results[2])
                if new_best_fitness < best_fitness
                    best_fitness = new_best_fitness
                    best_state = new_states[new_best_idx,:]
                end
            end
        end
    end #part 2

    return Array(best_state), cart_model, DT_warmstart, 1, cart_time
end # DEb1b

function update_states_cpu_matrix(states, best_state, L, U, Np, state_size, Cr, p)
    # Random.seed!(42)
    branch_number = Int(state_size/2)
    new_states = zeros(Float32, Np, state_size) # NP x state_size
    idx1 = ceil.(Int32, rand(Np)*Np) # NP x 1
    idx2 = ceil.(Int32, rand(Np)*Np)  # NP x 1
    states1 = states[idx1, :]  # NP x state_size
    states2 = states[idx2, :] # NP x state_size
    F = rand(Np) # F is a scalar
    va = best_state'.+F.*(states1-states2) # NP x state_size
    LL = va .< L # NP x state_size
    va = LL .* L .+ va .* (1 .- LL) # NP x state_size
    UU = va .> U # NP x state_size
    va = UU .* U .+ va .* (1 .- UU) # NP x state_size
    va[:, 1:branch_number] = floor.(va[:, 1:branch_number]) .+ (va[:, 1:branch_number] .== p+1) .* -1
    
    cross_point = rand(Float32, Np, state_size) .< Cr # NP x state_size
    # new_states = cross_point .* va .+ (1 .- cross_point) .* states # NP x state_size
    new_states[cross_point] = va[cross_point]
    new_states[.!(cross_point)] = states[.!(cross_point)]
    
    return new_states
end

function update_states_gpu(states, best_state, L, U, Np, state_size, Cr, p)
    branch_number = Int(state_size/2)
    states = CuArray(states)
    best_state = CuArray(best_state)
    new_states = CUDA.zeros(Float32, Np, state_size) # NP x state_size
    idx1 = ceil.(Int32, CUDA.rand(Np)*Np) # NP x 1
    idx2 = ceil.(Int32, CUDA.rand(Np)*Np)  # NP x 1
    states1 = states[idx1, :]  # NP x state_size
    states2 = states[idx2, :] # NP x state_size
    F = CUDA.rand(Np) # F is a scalar
    va = best_state'.+F.*(states1-states2) # NP x state_size
    LL = va .< L # NP x state_size
    va = LL .* L .+ va .* (1 .- LL) # NP x state_size
    UU = va .> U # NP x state_size
    va = UU .* U .+ va .* (1 .- UU) # NP x state_size
    va[:, 1:branch_number] = floor.(va[:, 1:branch_number]) .+ (va[:, 1:branch_number] .== p+1) .* -1

    cross_point = CUDA.rand(Float32, Np, state_size).<Cr # not considering all 0 situation, NP x state_size
    new_states = va .* cross_point .+ states .* (1 .- cross_point) # NP x state_size

    return Array(new_states)
end

function layer_by_layer_original_warmStart(fun_args, tree_da, P, K, X_d, X, Y, Y_K, tree_size, var_number, Nmin, Np::Int32=Int32(20), F=0.8,Cr=0.7, imprimir=0, xinit=nothing, ws_flag=true, verbose=false)
    tree_depth = floor(Int32, log2(tree_size+1)-1)
    size_branch = floor(Int32, tree_size/2)
    xinits = nothing
    classes=sort(unique(Y))      #get a dataframe with the classes in the data
    class_labels=size(classes,1)               #number of classes 
    original_splits = deepcopy(fun_args[13])
    original_sorted_X = deepcopy(fun_args[14])

    xbest_1 = zeros(1, size_branch*var_number) # puerly layerbylayer, [size_branch*a, size_branch*b, ...]
    if xinit !== nothing
        xbest_3 = deepcopy(xinit) # substitude xinit, [size_branch*a, size_branch*b, ...]
        xbest_3_best = deepcopy(xinit) # substitude xinit, [size_branch*a, size_branch*b, ...]
        fitness_3_best = OCT(tree_da, xbest_3_best, X, Y_K, classes, tree_size, Nmin, 2, 0.05, original_splits)
        verbose && println("fitness_xinit: ", fitness_3_best)
    end
    xbest = zeros(1, size_branch*var_number)
    seed = nothing

    for i in 1:ceil(Int, size_branch)
        P_i = tree_depth - floor(Int32, log2(i)) >= P ? P : tree_depth - floor(Int32, log2(i))
        @timeit get_timer("Shared") "get_data" X_i, Y_i = get_data_gpu(X, Y, X_d, i, xbest, size_branch, fun_args[7][2], fun_args[8][2], tree_da, original_splits)
        tree_size_i = 2^(P_i+1) - 1

        if xinit !== nothing
            xinit_i = get_sub_tree(xinit, i, var_number, P_i)
            xinits = [xinit_i]
        end
        if size(Y_i,1) > Nmin && length(unique(Y_i)) > 1
            verbose && println("### node number: ", i, ", Predicted depth: ", P_i, ", Dataset size: ", length(Y_i), " ###")
            @timeit get_timer("Shared") "args_pre" fun_args = args_pre(class_labels, fun_args[1], X_i, X_i, Y_i, tree_size_i, fun_args[6], fun_args[7], fun_args[8], Np, fun_args[9], verbose)
            xbest_i, cart_model, DT_warmstart,  = DEb1b_warmStart(fun_args, tree_da, K, X_i, Y_i, tree_size_i, var_number, Nmin, Np, F, Cr, imprimir, xinits, seed, ws_flag, 0, verbose)
            xbest_i = encode_xbest_i(xbest_i, length(original_sorted_X), floor(Int, tree_size_i/2), fun_args[13], original_sorted_X)
            @timeit get_timer("Shared") "selection" begin
                for j in 1:var_number
                    xbest_1[i+(j-1)*size_branch] = xbest_i[floor(Int32,tree_size_i/2)*(j-1)+1]
                end
                if xinit !== nothing
                    for j in 1:var_number
                        xbest_3[i+(j-1)*size_branch] = xbest_i[floor(Int32,tree_size_i/2)*(j-1)+1]
                    end
                    fitness_3 = OCT(tree_da, xbest_3, X, Y_K, classes, tree_size, Nmin, 2, 0.05, original_splits)
                    if fitness_3 < fitness_3_best
                        xbest_3_best = deepcopy(xbest_3)
                        verbose && println("fitness3: ", fitness_3, ", fitness3_best: ", fitness_3_best, ", select xbest_3")
                        fitness_3_best = fitness_3
                    else
                        verbose && println("fitness3: ", fitness_3, ", fitness3_best: ", fitness_3_best, ", select xbest_3_best")
                    end
                end
                xbest = xbest_1
            end # selection
        else
            verbose && println("### node number: ", i, ", P: ", P_i, ", size: ", length(Y_i), "<= Nmin", ", classes: ", length(unique(Y_i)), " ###")
            xbest_i = []
        end
    end
    @timeit get_timer("Shared") "final selection" begin
        # select best x from xbest_1, xbest_2, xbest_3 and xinit
        if xinit !== nothing
            fitness_1 = OCT(tree_da, xbest_1, X, Y_K, classes, tree_size, Nmin, 2, 0.05, original_splits)
            fitness_3 = OCT(tree_da, xbest_3_best, X, Y_K, classes, tree_size, Nmin, 2, 0.05, original_splits)
            fitness_init = OCT(tree_da, xinit, X, Y_K, classes, tree_size, Nmin, 2, 0.05, original_splits)
            fitnesses = [fitness_1, fitness_3, fitness_init]
            verbose && println("fitnesses: ", fitnesses)
            xbests = [xbest_1, xbest_3_best, xinit]
            xbest = xbests[argmin(fitnesses)]
        else
            xbest = xbest_1
        end
    end # end of @timeit

    return xbest
end

# encode xbest_i from the cur_splits to the original splits
function encode_xbest_i(xbest_i, p, size_branch, cur_splits, original_sorted_X)
    # transfer xbest_i from index to absolute value
    a, b, = trans_params_azd(xbest_i, p, size_branch, cur_splits) 
    b = encode_b(a, b, original_sorted_X)

    new_xbest_i = deepcopy(xbest_i)
    new_xbest_i[size_branch+1:2*size_branch] = b

    return new_xbest_i
end

# tree_da - 1: tree_d; 2: tree_a; 3: tree_a_with_zero
function get_data_gpu(X, Y, X_d, i, xbest, size_branch, kernel, threads, tree_da=1, splits=nothing) #i=5
    # [floor(i/2), floor(i/4), ..., 1]
    ancesters = [floor(Int32, i/2^j) for j in floor(Int32, log2(i)):-1:0] # [1,2,5]
    # println("node ", i, ":", ancesters)
    size_ancs = length(ancesters) # 3
    n, p = size(X) 
    a, b, d = trans_params_azd(xbest, p, size_branch, splits)
    selected = cu(trues(n))
    threads = min(n, threads)
    blocks = ceil(Int32, n/threads)
    kernel(selected, X_d, CuArray(a), CuArray(Float32.(b)), CuArray(d), CuArray(ancesters), n, size_ancs; threads, blocks) # _get_data_gpu
    selected = Array(selected)

    return X[selected, :], Y[selected]
end

function get_child_index(branch_index, p_i)
    childs = Int32[branch_index]
    for i in 1:p_i-1
        childs_i = Int32[2*x for x in childs]
        childs_i = union(childs_i, Int32[2*x+1 for x in childs])
        childs = sort(union(childs, childs_i))
    end
    return childs
end

function get_sub_tree(initial_tree, branch_index, var_number, p_i)
    branch_size = floor(Int, length(initial_tree)/var_number)
    childs = get_child_index(branch_index, p_i)
    # println("branch_index: ", branch_index, ",P_i: ", p_i,", childs: ", childs)
    # println("initial_tree: ", initial_tree)
    sub_tree = Float32[]
    for j in 1:var_number
        sub_tree = [sub_tree; Float32[initial_tree[(j-1)*branch_size + x] for x in childs]]
    end
    return sub_tree
end

end # module
