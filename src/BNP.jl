module BNP

import HDF5
import Distributions

export
Gaussian1D, Dirichlet,
Gaussian1DGaussian1D, MultinomialDirichlet,
BMM, LDA, DPM,
sample, collapsed_gibbs_sampler, truncated_gibbs_sampler, posterior

include("common.jl")
include("distributions.jl")
include("conjugates.jl")
include("BMM.jl")
include("LDA.jl")
include("DPM.jl")

end # module
