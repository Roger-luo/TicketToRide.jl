println("starting!!!")
using DelimitedFiles
using TicketToRide
using Flux
using Flux.Optimise
using TicketToRide: train!


Ls = [4, 6, 8]
Nlayers = collect(1:50)
repeats = [i for i in 1:5]

arg = 1
 #parse(Int, ARGS[1])

list = Iterators.product(Ls, Nlayers, repeats) |> collect
println("length of array jobs is:\t", length(list))

n, nlayers, r = list[arg]

for L in Ls
    for nlayers in Nlayers
        for repeats in 1:10
            @info "\nL is $n"
            @info "nlayers is $nlayers"
            @info "r is $r"
            # make directory if none exists
            dir = "/scratch/mbeach/new_new_tickettoride/L-$n/"
            mkpath(dir)
            file = dir * "layers-$nlayers-r-$r.txt" 
            @info file

            # pre-compile once 
            opt = Optimise.ADAMW()
            circuit = variational_circuit(n, nlayers);
            ham = heisenberg1D(n)
            dispatch!(circuit, :random);

            ind = findall(x -> x == n, Ls) 
            println(ind)

            opt = Optimise.ADAM()
            # @time history = train!(opt, 1, ham, circuit; verbose=true)
            history = train!(opt, 2000, ham, circuit; verbose=false)
            writedlm(file, history)

            opt = Optimise.Descent(1e-3)
            history = train!(opt, 2000, ham, circuit; verbose=false)
            writedlm(file, history)

            opt = Optimise.Descent(1e-4)
            history = train!(opt, 20000, ham, circuit; verbose=false)
            writedlm(file, history)

        end
    end
end
