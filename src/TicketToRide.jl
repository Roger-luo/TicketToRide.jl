module TicketToRide

using Yao, YaoExtensions
export neighboring_pairs

neighboring_pairs(n) = [i=>mod1(i+1, n) for i in 1:n]


end # module
