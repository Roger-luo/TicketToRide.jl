using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!
using LuxurySparse

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

function find_smallest(ngates, circuit)
    return sort!(parameters(circuit); by=abs)[ngates]
end

function prune_train(opt, circuit, ham; verbose=false, nprune=10, nepochs=10, niteration=100)
    total_history = Float64[]
    try
        for k in 1:nepochs
            @info "epoch = $k"
            @info "nparameters = $(nparameters(circuit))"
            history = train!(opt, niteration, ham, circuit; verbose=verbose)
            append!(history, total_history)
            @info "E/n = $(history[end])"
            circuit = prune(circuit; threshold=find_smallest(nprune, circuit));
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

opt = Optimise.ADAM()
n = 10
nlayers = 100
circuit = variational_circuit(n, nlayers);
ham = heisenberg1D(n);
dispatch!(circuit, :random);
history = train!(opt, 100, ham, circuit; verbose=true)
circuit = prune_train(opt, circuit, ham; nprune=100, nepochs=50, niteration=100);

# if the training is not enough we do more in the end to get better
opt = Optimise.ADAM(0.01)
train!(opt, 10000, heisenberg1D(10), circuit)
nparameters(circuit)
