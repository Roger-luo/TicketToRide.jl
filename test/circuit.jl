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



function prune_train(n, nlayers, nepochs, niteration)
    opt = Optimise.ADAM()
    circuit = variational_circuit(n, nlayers);
    ham = heisenberg1D(n)
    dispatch!(circuit, :random);

    for k in 1:nepochs
        @info "epoch = $k"
        @info "nparameters = $(nparameters(circuit))"
        train!(opt, niteration, ham, circuit)
        circuit = prune(circuit);
    end
    return circuit
end

prune_train(10, 100, 100, 10)