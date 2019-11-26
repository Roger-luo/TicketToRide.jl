using Distributed
addprocs(30; exeflags="--project")

@everywhere begin

using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!
using LuxurySparse
using LinearAlgebra

function task(n, nlayers; verbose=false, nprune=10, nepochs=10, niteration=100, relative_error=true, least_prune=10)
    opt = Optimise.ADAMW()
    circuit = variational_circuit(n, nlayers);
    ham = heisenberg1D(n);
    dispatch!(circuit, :random);
    circuit, history = prune_train(opt, circuit, ham;
        nprune=nprune, nepochs=nepochs,
        niteration=niteration,
        relative_error=relative_error,
        least_prune=least_prune, cnot_prune=false);

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
    circuit, history = nothing, nothing
    for each in results        
        if each[2][end] < min_energy
            circuit = each[1]
            history = each[2]
        end
    end
    return circuit, history
end

end

using JLD2
c, h = run_task(30, 10, 100; nepochs=20, nprune=50, niteration=1000, least_prune=5)

jldopen("data-10-30.jld", "w+") do f
    f["circuit"] = c
    f["history"] = h
end
