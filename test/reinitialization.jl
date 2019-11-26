using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!
using LuxurySparse
using LinearAlgebra

opt = Optimise.ADAMW()
circuit = variational_circuit(10, 100);
ham = heisenberg1D(10);
dispatch!(circuit, :random);

original = copy(circuit);

circuit, history = prune_train(opt, circuit, ham;
    nprune=50, nepochs=30,
    niteration=500,
    relative_error=true,
    least_prune=5);
