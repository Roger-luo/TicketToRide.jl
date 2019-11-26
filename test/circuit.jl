@info "loading packages"
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
    opt = Optimise.Descent(1e-2)
    circuit = variational_circuit(n, nlayers);
    ham = heisenberg1D(n);
    dispatch!(circuit, :random);
    circuit, history = prune_train(opt, circuit, ham;
        nprune=nprune, nepochs=nepochs,
        niteration=niteration,
        relative_error=relative_error,
        least_prune=least_prune);

    opt = Optimise.Descent(1e-3)
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
# @time c, h = run_task(2, 2, 2; nepochs=2, nprune=5, niteration=2, least_prune=5)

@time c, h = run_task(nprocs(), 10, 2; nepochs=2, nprune=5, niteration=2, least_prune=5)
c, h = run_task(nprocs(), 10, 100; nepochs=20, nprune=50, niteration=2000, least_prune=5)

jldopen("data-10-10.jld", "w+") do f
    f["circuit"] = c
    f["history"] = h
end

# opt = Optimise.Descent(1e-3)
# circuit = variational_circuit(10, 100);
# nparameters(circuit)
# dispatch!(circuit, :random);
# history = train!(opt, 2000, heisenberg1D(10), circuit; verbose=true)
# prune_train(opt, circuit, heisenberg1D(10);
#     nprune=50, nepochs=20,
#     niteration=500,
#     relative_error=true,
#     least_prune=5);


# nparameters(circuit)
