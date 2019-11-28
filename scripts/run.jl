println("starting!!!")
using DelimitedFiles
using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!

Exs = [-0.425803520728287]

Ls = [10]
n = 10

# nlayers = [50, 60, 70, 80, 90, 100]

nlayers = parse(Int, ARGS[1])

println("\nL is $n")
println("nlayers is $nlayers")

# pre-compile once 
opt = Optimise.Descent(1e-3)
circuit = variational_circuit(n, nlayers);
ham = heisenberg1D(n)
dispatch!(circuit, :random);

# make directory if none exists
dir = "testing_tickets/L-$n/"
mkpath(dir)
file = dir * "layers-$nlayers.txt" 

@info file

ind = findall(x -> x == n, Ls) 
println(ind)

opt = Optimise.Descent(1e-2)
@time history = train!(opt, 1, ham, circuit; verbose=true)
history = train!(opt, 2000, ham, circuit; verbose=true, Eex = Exs[ind][1])
writedlm(file, history)

opt = Optimise.Descent(1e-3)
history = train!(opt, 2000, ham, circuit; verbose=true, Eex = Exs[ind][1])
writedlm(file, history)

