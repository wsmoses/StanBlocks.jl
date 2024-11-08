module StanBlocks

using LinearAlgebra, Statistics, Distributions, LogExpFunctions

include("macros.jl")
include("functions.jl")

"Initializes a (PosteriorDB) model."
julia_implementation(key; kwargs...) = missing

end # module StanBlocks
