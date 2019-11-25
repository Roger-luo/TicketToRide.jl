module TicketToRide
using Reexport
@reexport using Yao, YaoExtensions
using Yao.YaoBlocks.Optimise
export simplify
export neighboring_pairs

neighboring_pairs(n) = [i=>mod1(i+1, n) for i in 1:n]

YaoExtensions.variational_circuit(n, nlayers) = variational_circuit(n, nlayers, neighboring_pairs(n))

end # module
