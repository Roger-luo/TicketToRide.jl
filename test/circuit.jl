using Distributed
addprocs(2; exeflags="--project")

@everywhere begin

using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!
using LuxurySparse
using LinearAlgebra

function task(n, nlayers; verbose=false, nprune=10, nepochs=10, niteration=100, relative_error=true, least_prune=10)
    opt = Optimise.ADAM()
    circuit = variational_circuit(n, nlayers);
    ham = heisenberg1D(n);
    dispatch!(circuit, :random);
    circuit, history = prune_train(opt, circuit, ham;
        nprune=nprune, nepochs=nepochs,
        niteration=niteration,
        relative_error=relative_error,
        least_prune=least_prune);

    return circuit, history
end

function run_task(total, n, nlayers; verbose=false, nprune=10, nepochs=10, niteration=100, relative_error=true, least_prune=10)
    results = pmap(1:total) do k
        task(n, nlayers;
        nprune=nprune, nepochs=nepochs,
        niteration=niteration,
        relative_error=relative_error,
        least_prune=least_prune)
    end

    min_energy = 0
    circuit, history = nothing
    for each in results        
        if each[2][end] < min_energy
            circuit = each[1]
            history = each[2]
        end
    end
    return circuit, history
end

end

run_task(2, 10, 100)

n = 10
nlayers = 100

# if the training is not enough we do more in the end to get better
opt = Optimise.ADAM(0.01)
train!(opt, 10000, heisenberg1D(n), circuit)
nparameters(circuit)
