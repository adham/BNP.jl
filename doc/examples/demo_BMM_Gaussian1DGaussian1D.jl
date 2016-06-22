#=
demo_BMM_Gaussian1DGaussian1D.jl

Demo for Bayesian Finite Mixture Model with Gaussian1DGaussian1D conjugates.

08/07/2015
Adham Beyki, odinay@gmail.com
=#

using BNP
srand(123)


## --- synthesizing the data --- ##
true_KK = 5                             # number of components
NN      = 500                           # number of data points

vv = 0.01                               # fixed variance
ss = 2
true_atoms = [Gaussian1D(ss*kk, vv) for kk = 1:true_KK]

mix = ones(Float64, true_KK)/true_KK
xx  = ones(Float64, NN)
true_zz = ones(Int, NN)
true_nn = zeros(Int, true_KK)

for n=1:NN
    kk = sample(mix)
    true_zz[n] = kk
    xx[n] = sample(true_atoms[kk])
    true_nn[kk] += 1
end



## ------- inference -------- ##
# constructing the Gaussian1DGaussian1D conjugate
m0 = mean(xx)
v0 = 2.0
q0 = Gaussian1DGaussian1D(m0, v0, vv)


# constructing the Bayesian Mixture Model
KK = true_KK
bmm_aa = 1.0
bmm = BMM(q0, KK, bmm_aa)

# initializing the cluster labels
zz = zeros(Int, length(xx))
zz = rand(1:bmm.KK, NN)


# sampling
n_burnins    = 100
n_lags       = 2
n_samples    = 200
store_every  = 10000
results_path = ""
filename     = "demo_BMM_Gaussian1DGaussian1D"

collapsed_gibbs_sampler(bmm, xx, zz, n_burnins, n_lags, n_samples, store_every, results_path, filename)


# posterior distributions
posterior_components, nn = posterior(bmm, xx, zz)
