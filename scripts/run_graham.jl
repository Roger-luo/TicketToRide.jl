println("starting!!!")
using DelimitedFiles
using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!

Ls = [4,6,8,10,12,14,16]
# nlayers = [n for n in 100:100:1000]
nlayers = [10, 50, 100, 200, 500, 1000, 2000, 5000]
repeats = [i for i in 1:4]

arg = parse(Int, ARGS[1])

list = Iterators.product(Ls, nlayers, repeats) |> collect
println("length of array jobs is:\t", length(list))

n, nlayers, r = list[arg]
println("\nL is $n")
println("nlayers is $nlayers")
println("r is $r")

# pre-compile once 
opt = Optimise.Descent(1e-2)
circuit = variational_circuit(n, nlayers);
ham = heisenberg1D(n)
dispatch!(circuit, :random);

# make directory if none exists
dir = "/scratch/mbeach/new_tickettoride/L-$n/"
mkpath(dir)
file = dir * "layers-$nlayers-r-$r.txt" 

@info file

@time history = train!(opt, 1, ham, circuit; verbose=true)
history = train!(opt, 500, ham, circuit; verbose=true)
writedlm(file, history)
