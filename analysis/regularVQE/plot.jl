using Plots
using DelimitedFiles

a = readdlm("vqe.txt")

using GR
heatmap(a, xlog=true)

