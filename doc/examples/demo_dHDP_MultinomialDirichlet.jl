#=
demo_HDP_MultinomialDirichlet.jl

A demo for Hierarchical Dirichlet Process mixture models with MultinomialDirichlet 
Bayesian components. This demo uses auxiliary variable method for for inference.

28/07/2015
Adham Beyki, odinay@gmail.com
=#

using BNP
srand(123)


true_gg = 1.0
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


n_groups   = 50
n_group_j  = 100 * ones(Int, n_groups)
vocab_size = 25

# constructing the topics
true_topics = BNP.gen_bars(true_KK, vocab_size, 0)

# constructing the observations and labels
xx = Array(Vector{Int}, n_groups)
true_nn = zeros(Int, n_groups, true_KK)

for jj = 1:n_groups
    xx[jj] = zeros(Int, n_group_j[jj])
    for ii = 1:n_group_j[jj]
        kk = true_zz[jj][ii]
        xx[jj][ii] = sample(true_topics[kk, :][:])
        true_nn[jj, kk] += 1
    end
end


## ------- inference -------- ##
# constructing the Bayesian component
dd = vocab_size
aa = 0.5*dd
q0 = MultinomialDirichlet(dd, aa)


# constructing the  model
dhdp_KK_init = 1
KK_truncation = 15
dhdp_gg = 1.0
dhdp_g1 = 0.1
dhdp_g2 = 0.1
dhdp_aa = 0.5
dhdp_a1 = 0.1
dhdp_a2 = 0.1
dhdp_aw = 2.0
dhdp_bw = 3.0
dhdp = dHDP(q0, dhdp_KK_init, dhdp_gg, dhdp_g1, dhdp_g2, dhdp_aa, dhdp_a1, dhdp_a2, dhdp_aw, dhdp_bw)

# sampling
zz = Array(Vector{Int}, n_groups)
for jj = 1:n_groups
    zz[jj] = ones(Int, n_group_j[jj])
    zz[jj] = rand(1:dhdp.KK, n_group_j[jj])
end

n_burnins   = 500
n_lags      = 1
n_samples   = 500
sample_hyperparam = false
n_internals = 10
store_every = 1000
results_path = ""
filename    = "demo_dHDP_MultinomialDirichlet"

KK_list, KK_dict, w_tilde = BNP.truncated_gibbs_sampler(dhdp, KK_truncation, xx, zz, 
    n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, results_path, filename)

KK_hist = hist(KK_list, 0.5:maximum(KK_list)+0.5)[2]
candidate_KK = indmax(KK_hist)
posterior_components, nn = posterior(dhdp, xx, KK_dict[candidate_KK], candidate_KK)

