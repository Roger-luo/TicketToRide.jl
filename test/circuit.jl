using TicketToRide, Flux
using Flux.Optimise

circuit = variational_circuit(10, 20);
dispatch!(circuit, :random);

h(n, i) = sum(put(n, i=>σ) * put(n, i+1=>σ) for σ in (X, Y, Z))
heisenberg(n) = simplify(sum(h(n, i) for i in 1:n-1))

function train(opt, epochs)
    history = Float64[]
    for k in 1:epochs
        # the expectation is calculated on a complex matrix
        # thus we just use the real part here
        E = expect(heisenberg(10), zero_state(10)=>circuit) |> real
        @info "step=$k"
        @info "E=$E"
        push!(history, E)
        _, grad = expect'(heisenberg(10), zero_state(10)=>circuit)
        ps = parameters(circuit)
        Optimise.update!(opt, ps, grad)
        popdispatch!(circuit, ps)
    end
    return history
end

# TODO: how to prune?
opt = Optimise.ADAM()
train(opt, 10)