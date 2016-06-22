#=
DPM_demo_Gaussian1DGaussian1D_collapsed

A demo for Dirichlet Process Mixture Model with Gaussian1DGaussian1D Bayesian
components.

08/07/2015
Adham Beyki, odinay@gmail.com
=#

using BNP


## --- synthesizing the data ---
# synthesized corpus properties
true_KK = 5           # number of components
NN = 500              # number of data points

vv = 0.02            # fixed variance
ss = 2

true_atoms = [Gaussian1D(ss*kk, vv) for kk = 1:true_KK]

# constructing the observations and labels
mix = ones(Float64, true_KK)/true_KK
xx = ones(Float64, NN)
true_zz = ones(Int64, NN)
true_nn = zeros(Int64, true_KK)

for n=1:NN
    kk = sample(mix)
    true_zz[n] = kk
    xx[n] = sample(true_atoms[kk])
    true_nn[kk] += 1
end


## ------- inference --------
# constructing the conjugate component
m0= mean(xx)
v0 = 2
q0 = Gaussian1DGaussian1D(m0, v0, vv)

# constructing the model
init_KK = 1
KK_truncation = 10
dpm_aa  = 1
dpm_aa1 = 1
dpm_aa2 = 1
dpm = DPM(q0, init_KK, dpm_aa, dpm_aa1, dpm_aa2)

# sampling
zz = zeros(Int64, length(xx))
zz = rand(1:dpm.KK, NN)

n_burnins   = 100
n_lags      = 1
n_samples   = 300
sample_hyperparam = true
n_internals = 10
store_every = 10000
results_path = ""
filename    = "demo_DPM_Gaussian1DGaussian1D_"
KK_list, KK_dict = truncated_gibbs_sampler(dpm, KK_truncation, xx, zz,
	n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, results_path, filename)

KK_hist = hist(KK_list, 0.5:maximum(KK_list)+0.5)[2]
candidate_K = indmax(KK_hist)
posterior_components, nn = posterior(dpm, xx, KK_dict, candidate_K)
