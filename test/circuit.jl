using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!

function prune(x::RotationGate; threshold=1e-1)
    if x.theta < threshold
        return NoGate()
    else
        return x
    end
end

prune(x::AbstractBlock; threshold=1e-1) =
    chsubblocks(x, prune.(subblocks(x); threshold=threshold))

# TODO: how to prune?
opt = Optimise.ADAM()
circuit = variational_circuit(4, 100);
dispatch!(circuit, :random);
train!(opt, 1000, heisenberg1D(4), circuit)

length(filter(x->abs(x)<0.1, parameters(circuit)))/nparameters(circuit)

t = Depolar(0.1, 0.2, 0.3, 0.8)
rand_state(10) |> repeat(10, Depolar(0.1, 0.2, 0.3, 0.8))


using LinearAlgebra


M = mat(heisenberg1D(4))/4

eigmin(Matrix(M))/4