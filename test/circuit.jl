using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!
using LuxurySparse
using LinearAlgebra


opt = Optimise.ADAM()
n = 10
nlayers = 100
circuit = variational_circuit(n, nlayers);
ham = heisenberg1D(n);
dispatch!(circuit, :random);
circuit = prune_train(opt, circuit, ham; nprune=100, nepochs=50, niteration=100);

# if the training is not enough we do more in the end to get better
opt = Optimise.ADAM(0.01)
train!(opt, 10000, heisenberg1D(10), circuit)
nparameters(circuit)
