println("starting!!!")
using DelimitedFiles
using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!

# Exs = [-0.4040063509461097,
       # -0.4155961889813218,
       # -0.42186657483598655,
        # -0.4258035207282875,
        # -0.4285075527367124]

Ls = [4, 6, 8]
nlayers = [1:50]
repeats = [i for i in 1:5]

arg = parse(Int, ARGS[1])

list = Iterators.product(Ls, nlayers, repeats) |> collect
println("length of array jobs is:\t", length(list))

n, nlayers, r = list[arg]
println("\nL is $n")
println("nlayers is $nlayers")
println("r is $r")

# pre-compile once 
opt = Optimise.Descent(1e-3)
# opt = Optimise.ADAMW()
circuit = variational_circuit(n, nlayers);
ham = heisenberg1D(n)
dispatch!(circuit, :random);

# make directory if none exists
dir = "/scratch/mbeach/new_new_tickettoride/L-$n/"
mkpath(dir)
file = dir * "layers-$nlayers-r-$r.txt" 

@info file

ind = findall(x -> x == n, Ls) 
println(ind)

opt = Optimise.ADAM()
@time history = train!(opt, 1, ham, circuit; verbose=true)
history = train!(opt, 2000, ham, circuit; verbose=true, Eex = Exs[ind][1])
writedlm(file, history)

opt = Optimise.Descent(1e-3)
history = train!(opt, 2000, ham, circuit; verbose=true, Eex = Exs[ind][1])
writedlm(file, history)

opt = Optimise.Descent(1e-4)
history = train!(opt, 2000, ham, circuit; verbose=true, Eex = Exs[ind][1])
writedlm(file, history)

