using Yao.YaoBlocks.Optimise: simplify
using ProgressMeter
export simplify
export neighboring_pairs, heisenberg1D, train!

neighboring_pairs(n) = [i=>mod1(i+1, n) for i in 1:n]

YaoExtensions.variational_circuit(n, nlayers) = variational_circuit(n, nlayers, neighboring_pairs(n))

_heisenberg_S(n, i) = sum(put(n, i=>σ) * put(n, i+1=>σ) for σ in (X, Y, Z))
heisenberg1D(n) = simplify(sum(_heisenberg_S(n, i) for i in 1:n-1))


using Flux.Optimise

function train!(opt, epochs, ham, circuit; verbose=true)
    history = Float64[]
    n = nqubits(ham)
    for k in 1:epochs
        # the expectation is calculated on a complex matrix
        # thus we just use the real part here
        E = expect(ham, zero_state(n)=>circuit) |> real
        
        if verbose
            @info "step=$k"
            @info "E/n=$(E/4n)"
        end
        push!(history, E/4n)
        _, grad = expect'(ham, zero_state(n)=>circuit)
        ps = parameters(circuit)
        Optimise.update!(opt, ps, grad)
        popdispatch!(circuit, ps)
    end
    return history
end
