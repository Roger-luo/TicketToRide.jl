using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!

function prune(x::RotationGate; threshold=1e-1)
    if abs(x.theta) < threshold
        return NoGate()
    else
        return x
    end
end

prune(x::AbstractBlock; threshold=1e-1) =
    chsubblocks(x, prune.(subblocks(x); threshold=threshold))


function find_smallest(ngates, circuit)
    return sort!(parameters(circuit); by=abs)[ngates]
end

function prune_train(n, nlayers, nepochs, niteration; verbose=false, nprune=10)
    opt = Optimise.ADAM()
    circuit = variational_circuit(n, nlayers);
    ham = heisenberg1D(n)
    dispatch!(circuit, :random);

    for k in 1:nepochs
        @info "epoch = $k"
        @info "nparameters = $(nparameters(circuit))"
        history = train!(opt, niteration, ham, circuit; verbose=verbose)
        @info "E/n = $(history[end])"
        circuit = prune(circuit; threshold=find_smallest(nprune, circuit));
    end
    return circuit
end

circuit = prune_train(10, 100, 10, 100; nprune=100);
