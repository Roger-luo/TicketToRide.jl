using Distributed
if length(ARGS) < 2
    error("expect number of qubits and number of random initialized circuits")
end

n = parse(Int, ARGS[1])
nrandom_initialize = parse(Int, ARGS[2])
addprocs(nrandom_initialize; exeflags="--project")

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
c, h = run_task(nrandom_initialize, n, 100; nepochs=10, nprune=50, niteration=1000, least_prune=5)

jldopen("data-$n-$nrandom_initialize.jld", "w+") do f
    f["circuit"] = c
    f["history"] = h
end
