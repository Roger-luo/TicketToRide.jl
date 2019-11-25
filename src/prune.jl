using LinearAlgebra
export prune, rm_redundant_rotation, find_smallest, prune_train

function prune(x::RotationGate; threshold=1e-1)
    if abs(x.theta) < threshold
        return NoGate()
    else
        return x
    end
end

prune(x::AbstractBlock; threshold=1e-1) =
    chsubblocks(x, prune.(subblocks(x); threshold=threshold))

function rm_redundant_rotation(x::ChainBlock{1})
    length(x) === 3 || return x
    # this is an identity
    if tr((mat(x) - IMatrix(2))' * (mat(x) - IMatrix(2))) < 1e-2
        return NoGate()
    end
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
function prune_train(opt, circuit, ham; verbose=false, nprune=10, nepochs=10, niteration=100)
    total_history = Float64[]
    n = nqubits(ham)
    E0 = eigmin(Matrix(mat(ham)))/4n
    try
        for k in 1:nepochs
            @info "epoch = $k"
            @info "nparameters = $(nparameters(circuit))"
            history = train!(opt, niteration, ham, circuit; verbose=verbose)
            append!(history, total_history)
            @info "E/n = $(history[end])"
            @info "(E - E0)/E0 = $((history[end] - E0)/E0)"
            circuit = prune(circuit; threshold=abs(find_smallest(nprune, circuit)));
            circuit = rm_redundant_rotation(circuit);
        end
    catch e
        if e isa InterruptException
            return circuit
        else
            rethrow(e)
        end
    end
    return circuit, total_history
end
