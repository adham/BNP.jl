#=
demo_HDP_Gaussian1DGaussian1D

A demo for Hierarchical Dirichlet Process mixture models with Gaussian1DGaussian1D
Bayesian components. 

28/07/2015
Adham Beyki, odinay@gmail.com
=#


using BNP


## --- synthesizing the data --- ##
true_gg = 2.0
true_aa = 0.5
n_groups = 100
n_group_j   = 100 * ones(Int, n_groups)
KK_truncation = 25


beta_tilde = rand(Distributions.Beta(1, true_gg), KK_truncation)
beta_tilde[KK_truncation] = 1.0
my_beta = BNP.stick_breaking(beta_tilde)

pi_tilde = zeros(Float64, n_groups, KK_truncation)
my_pi = zeros(Float64, n_groups, KK_truncation)
for jj = 1:n_groups
    for kk = 1:KK_truncation-1
        gamma1 = true_aa * my_beta[kk]
        gamma2 = true_aa * (1 - sum(my_beta[1:kk])) 
        pi_tilde[jj, kk] = rand(Distributions.Beta(gamma1, gamma2))
    end
    pi_tilde[jj, KK_truncation] = 1.0
    my_pi[jj, :] = BNP.stick_breaking(pi_tilde[jj, :][:])
end

true_zz = Array(Vector{Int}, n_groups)
for jj = 1:n_groups
    true_zz[jj] = zeros(Int, n_group_j[jj])
    for ii = 1:n_group_j[jj]
        kk = sample(my_pi[jj, :][:])
        true_zz[jj][ii] = kk
    end
end

zz_flat = Int[]
for jj = 1:n_groups
	append!(zz_flat, true_zz[jj])
end

zz_unq = sort(unique(zz_flat))
true_KK = length(zz_unq)
for jj = 1:n_groups
	for kk = 1:true_KK
		idx = find(x -> x==zz_unq[kk], true_zz[jj])
		true_zz[jj][idx] = kk
	end
end

vv = 0.001          # fixed variance
ss = 2
true_atoms = [Gaussian1D(ss*kk, vv) for kk = 1:true_KK]

xx = Array(Vector{Float64}, n_groups)
true_nn = zeros(Int, n_groups, true_KK)
for jj = 1:n_groups
    xx[jj] = zeros(Float64, n_group_j[jj])
    for ii = 1:n_group_j[jj]
        kk = true_zz[jj][ii]
        xx[jj][ii] = sample(true_atoms[kk])
        true_nn[jj, kk] += 1
    end
end




m0 = mean(mean(xx))
v0 = 10.0
q0 = Gaussian1DGaussian1D(m0, v0, vv)

# constructing the HDP model
hdp_KK_init = 1
KK_truncation = 15
hdp_gg = 1.0
hdp_g1 = 0.1
hdp_g2 = 0.1
hdp_aa = 1.0
hdp_a1 = 0.1
hdp_a2 = 0.1
hdp = HDP(q0, hdp_KK_init, hdp_gg, hdp_g1, hdp_g2, hdp_aa, hdp_a1, hdp_a2)

# sampling
zz = Array(Vector{Int}, n_groups)
for jj = 1:n_groups
    zz[jj] = ones(Int, n_group_j[jj])
    zz[jj] = rand(1:hdp.KK, n_group_j[jj])
end

n_burnins   = 0
n_lags      = 0
n_samples   = 500
sample_hyperparam = true
n_internals = 10
store_every = 1000
results_path = ""
filename    = "demo_HDP_Gaussian1DGaussian1D_"


KK_list, KK_dict = truncated_gibbs_sampler(hdp, KK_truncation, xx, zz, 
    n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, results_path, filename)

KK_hist = hist(KK_list, 0.5:maximum(KK_list)+0.5)[2]
candidate_KK = indmax(KK_hist)
posterior_components, nn = posterior(hdp, xx, KK_dict[candidate_KK], candidate_KK)
