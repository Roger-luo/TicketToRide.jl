using LinearAlgebra
export prune, prune_parent, rm_redundant_rotation, find_smallest, prune_train

function prune(x::RotationGate; threshold=1e-1)
    if abs(x.theta) < threshold
        return NoGate()
    else
        return x
    end
end

prune(x::AbstractBlock; threshold=1e-1) =
    chsubblocks(x, prune.(subblocks(x); threshold=threshold))

prune_parent(c::NoGate, p::RotationGate) = NoGate()

function prune_parent(c::ChainBlock, p::ChainBlock)
    c[1] isa ControlBlock && return copy(c)
    chsubblocks(p, prune_parent.(subblocks(c), subblocks(p)))
end

prune_parent(c::AbstractBlock, p::AbstractBlock) =
    chsubblocks(p, prune_parent.(subblocks(c), subblocks(p)))

function rm_cnot(x::ChainBlock{N}) where N
    x[1] isa ControlBlock || return x
    deleteat!(x.blocks, rand(1:length(x)))
    return x
end

rm_cnot(x::AbstractBlock) = chsubblocks(x, rm_cnot.(subblocks(x)))

function rm_redundant_rotation(x::ChainBlock{1})
    length(x) === 3 || return x
    # this is an identity
    if tr((mat(x) - IMatrix(2))' * (mat(x) - IMatrix(2))) < 1e-2
        return NoGate()
    end

    if x[2] isa NoGate
        return Rz(x[1].theta + x[3].theta)
    end
    return x
end

rm_redundant_rotation(x::AbstractBlock) =
    chsubblocks(x, rm_redundant_rotation.(subblocks(x)))

"""
    find_smallest(ngates, circuit)

find the first `ngates` smallest absolute value of parameters
of gates in the given circuit.
"""
function find_smallest(ngates, circuit)
    return sort!(parameters(circuit); by=abs)[ngates]
end

"""
    prune_train(opt, circuit, ham; verbose=false, nprune=10, nepochs=10, niteration=100)

# Parameters
- `opt`: optimizer from `Flux.Optimise`.
- `circuit`: the circuit to prune
- `ham`: the Hamiltonian

# Keywords

- `verbose`: print the verbose info in each training process if `true`
- `nprune`: number of gates to prune
- `nepochs`: the number of epochs of pruning procedure
- `niteration`: number of iterations for training
"""
function prune_train(opt, circuit, ham; verbose=false, nprune=10, nepochs=10, niteration=100, relative_error=true, least_prune=10, cnot_prune=true)
    total_history = Float64[]
    n = nqubits(ham)
    original = copy(circuit)
    if relative_error
        E0 = eigmin(Matrix(mat(ham)))/4n
    end

    try
        for k in 1:nepochs
            @info "epoch = $k"
            @info "nparameters = $(nparameters(circuit))"
            history = train!(opt, niteration, ham, circuit; verbose=verbose)
            append!(total_history, history)
            @info "E/n = $(history[end])"

            if relative_error
                re = (history[end] - E0)/E0
                @info "(E - E0)/E0 = $re"

                if k > least_prune && re > 1e-2
                    break
                end    
            end

            circuit = prune(circuit; threshold=abs(find_smallest(nprune, circuit)));
            circuit = rm_redundant_rotation(circuit);
            circuit = copy(prune_parent(circuit, original))

            if cnot_prune
                circuit = chsubblocks(circuit, rm_cnot.(subblocks(circuit)))
            end
        end
    catch e
        if e isa InterruptException
            return circuit, total_history
        else
            rethrow(e)
        end
    end
    return circuit, total_history
end
