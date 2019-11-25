using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!

Ls = [4,6,8,10,12,14,16,18,20]
nlayers = [n for n in 100:100:1000]
repeats = [i for i in 1:10]

arg = parse(Int, ARGS[1])

list = Iterators.product(Ls, nlayers, repeats) |> collect
println("length of array jobs is:\t", length(list))

n, nlayers, r = list[arg]
println("\nL is $n")
println("nlayers is $nlayers")
println("r is $r")

# pre-compile once 
opt = Optimise.ADAM()
circuit = variational_circuit(n, nlayers);
ham = heisenberg1D(n)
dispatch!(circuit, :random);
@time history = train!(opt, 1, ham, circuit; verbose=false)

# make directory if none exists
dir = "/scratch/mbeach/tickettoride/L-$n/"
mkpath(dir)
file = dir * "layers-$nlayers-r-$r.txt" 

history = train!(opt, 200, ham, circuit; verbose=true, filename=file)
