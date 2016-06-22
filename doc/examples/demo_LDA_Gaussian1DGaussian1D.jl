#=
demo_LDA_Gaussian1DGaussian1D

A demo for LDA with Gaussian1DGaussian1D Bayesian components.

28/07/2015
Adham Beyki, odinay@gmail.com
=#

using BNP
srand(123)

## --- synthesizing the data --- ##
true_KK = 5
vv      = 0.01
ss      = 1
true_atoms = [Gaussian1D(ss*kk, vv) for kk = 1:true_KK]

n_groups  = 1000
n_group_j = 100 * ones(Int, n_groups)

# constructing the observations and labels
alpha = 0.1
xx = Array(Vector{Float64}, n_groups)
true_zz = Array(Vector{Int}, n_groups)
true_nn = zeros(Int, n_groups, true_KK)
for jj = 1:n_groups
    xx[jj] = zeros(Float64, n_group_j[jj])
    true_zz[jj] = ones(Int, n_group_j[jj])
    theta = BNP.rand_Dirichlet(alpha .* ones(Float64, true_KK))
    for ii = 1:n_group_j[jj]
        kk = sample(theta)
        true_zz[jj][ii] = kk
        true_nn[jj, kk] += 1
        xx[jj][ii] = sample(true_atoms[kk])
    end
end


## ------- inference -------- ##
# constructing the Gaussian1DGaussian1D conjugate
m0 = mean(mean(xx))
v0 = 2.0
q0 = Gaussian1DGaussian1D(m0, v0, vv)

# constructing the LDA model
KK = true_KK
lda_aa = 1.0
lda = LDA(q0, KK, lda_aa)

# initializing the cluster labels
zz = Array(Vector{Int}, n_groups)
for jj = 1:n_groups
    zz[jj] = ones(Int, n_group_j[jj])
    zz[jj] = rand(1:lda.KK, n_group_j[jj])
end

# sampling
n_burnins   = 100
n_lags      = 0
n_samples   = 200
store_every = 1000
results_path = ""
filename    = "demo_LDA_Gaussian1DGaussian1D_"

collapsed_gibbs_sampler(lda, xx, zz, n_burnins, n_lags, n_samples, store_every, results_path, filename)

# posterior distributions
posterior_components, nn = posterior(lda, xx, zz)
