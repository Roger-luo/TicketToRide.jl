using DelimitedFiles

Ls = [4,6,8,10,12,14,16]                                                                                
nlayers = collect(100:100:1000)
repeats = [i for i in 1:5]

function read()
    Es = zeros(length(Ls), length(nlayers), length(repeats))
    i = 0
    for (k, L) in enumerate(Ls)
        for (i, nlayer) in enumerate(nlayers)
            for (j, r) in enumerate(repeats)
                E = 0
                try
                    E =
                    readdlm("/scratch/mbeach/tickettoride/L-$L/layers-$nlayer-r-$r.txt")[end]
                catch
                end
                # println(E)
                # println(k, "  ", i, "  ",  j)
                Es[k, i, j] = E
            end
        end
    end
    return Es
end

Eex = [-0.4040063509461097,
       -0.4155961889813218,
       -0.42186657483598655,
        -0.4258035207282875,
        -0.4285075527367124]
# Eex = [Eex[i] for i in 1:5, for j in 1:10]
Es = read()
Emins = dropdims(minimum(Es, dims=3), dims=3)
