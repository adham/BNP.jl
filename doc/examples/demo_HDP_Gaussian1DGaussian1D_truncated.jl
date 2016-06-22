#=
demo_HDP_Gaussian1DGaussian1D

A demo for Hierarchical Dirichlet Process mixture models with Gaussian1DGaussian1D
Bayesian components. This demo uses auxiliary variable method for for inference.

28/07/2015
Adham Beyki, odinay@gmail.com
=#
using Debug
using BNP
srand(123)

additem! = BNP.additem!
loglikelihood = BNP.loglikelihood
logpredictive = BNP.logpredictive
delitem! = BNP.delitem!
sample = BNP.sample
lognormalize! = BNP.lognormalize!
sample_hyperparam! = BNP.sample_hyperparam!

count_active_clusters(nn::Matrix{Int}) = length(find(x -> x>0, sum(nn, 1)))
function stick_breaking(vv::Vector{Float64})
    KK = length(vv)
    pp = ones(Float64, KK)

    pp[2:KK] = 1-vv[1:KK-1]
    pp = vv .* cumprod(pp)
    pp
end


@debug function storesample{T}(
    hdp::HDP{T},
    KK_list::Vector{Int},
    KK_dict::Dict{Int, Vector{Vector{Int}}},
    sample_n::Int,
    filename::ASCIIString)

    if endswith(filename, "_")
        dummy_filename = string(filename, sample_n, ".h5")
    else
        dummy_filename = string(filename, "_", sample_n, ".h5")
    end

    KK_dict_keys = collect(keys(KK_dict))
    n_groups = length(KK_dict[KK_dict_keys[1]])
    n_group_j = Array(Int, n_groups)
    for jj = 1:n_groups
        n_group_j[jj] = length(KK_dict[KK_dict_keys[1]][jj])
    end

    println("storing on disk...")
    HDF5.h5open(dummy_filename, "w") do file
        for kk in KK_dict_keys
            zz_flat = Int[]
            for jj=1:n_groups
                append!(zz_flat, KK_dict[kk][jj])
            end
            HDF5.write(file, "zz/$(kk)", zz_flat)
        end
        HDF5.write(file, "sample_n", sample_n)
        HDF5.write(file, "n_groups", n_groups)
        HDF5.write(file, "n_group_j", n_group_j)        
    end


end


@debug function BNP.truncated_gibbs_sampler{T1, T2}(
    hdp::HDP{T1},
    KK_truncation::Int,
    xx::Vector{Vector{T2}},
    zz::Vector{Vector{Int}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true, n_internals::Int=10,
    store_every::Int=250, results_path="", filename::ASCIIString="HDP_results_")


    n_iterations    = n_burnins + (n_samples)*(n_lags+1)
    n_groups        = length(xx)
    n_group_j       = zeros(Int, n_groups)
    nn              = zeros(Int, n_groups, KK_truncation)
    pp              = zeros(Float64, KK_truncation)
    log_likelihood  = 0.0

    for jj = 1:n_groups
        n_group_j[jj] = length(zz[jj])
    end


    components = Array(typeof(hdp.component), KK_truncation)
    for kk = 1:KK_truncation
        components[kk] = deepcopy(hdp.component)
    end
    
    KK_list = Int[]
    KK_dict = Dict{Int, Vector{Vector{Int}}}()

    beta_tilde = zeros(Float64, KK_truncation)
    pi_tilde   = zeros(Float64, n_groups, KK_truncation)

    snumbers_file = string(Pkg.dir(), "\\BNP\\src\\StirlingNums_10K.mat")
    snumbers_data = MAT.matread(snumbers_file)
    snumbers = snumbers_data["snumbersNormalizedSparse"]
    
    ################################
    #    Initializing the model    #
    ################################    
    tic()
    print_with_color(:red, "\nInitializing the model\n")
    

    # 1
    # adding the data points
    for jj = 1:n_groups
        for ii = 1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            nn[jj, kk] += 1
            log_likelihood += loglikelihood(components[kk], xx[jj][ii])
        end
    end
    push!(KK_list, count_active_clusters(nn))


    # 2
    # constructing my_beta and its beta variables
    for kk = 1:KK_truncation-1
        gamma1 = 1 + sum(nn[:, kk])
        gamma2 = hdp.gg + sum(nn[:, kk+1:KK_truncation])
        beta_tilde[kk] = gamma1 / (gamma1 + gamma2)
    end
    beta_tilde[KK_truncation] = 1.0

    my_beta = stick_breaking(beta_tilde)

    # 3
    # constructing beta variables for pi_j
    for jj = 1:n_groups
        for kk = 1:KK_truncation-1
            gamma1 = hdp.aa * my_beta[kk] + nn[jj, kk]
            gamma2 = hdp.aa * (1 - sum(my_beta[1:kk])) + sum(nn[jj, kk+1:KK_truncation])
            pi_tilde[jj, kk] = gamma1 / (gamma1 + gamma2)
        end
        pi_tilde[jj, KK_truncation] = 1.0
    end

    elapsed_time = toq()


    ###################################
    #     starting the MCMC chain     #
    ###################################
    for iteration = 1:n_iterations

        # Verbose
        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, gg=%.2f, aa=%.2f, time=%.2f, likelihood=%.2f", 
            iteration, hdp.KK, indmax(hist(KK_list, .5:maximum(KK_list)+0.5)[2]), hdp.gg, hdp.aa, 
            elapsed_time, log_likelihood))
        log_likelihood = 0.0

        tic()

        for jj = 1:n_groups
            for ii = 1:n_group_j[jj]

                # 1
                # remove the data point
                kk = zz[jj][ii]
                delitem!(components[kk], xx[jj][ii])
                nn[jj, kk] -= 1

                # 2
                # resample for zz[jj][ii]
                pp = zeros(Float64, KK_truncation)
                for kk = 1:KK_truncation
                    pp[kk] = log(pi_tilde[jj, kk]) + sum(log(1-pi_tilde[jj, 1:kk-1])) + logpredictive(components[kk], xx[jj][ii])
                end
                lognormalize!(pp)
                kk = sample(pp)

                # 3
                # add the data point to the newly sampled cluster
                zz[jj][ii] = kk
                nn[jj, kk] += 1
                additem!(components[kk], xx[jj][ii])
                log_likelihood += loglikelihood(components[kk], xx[jj][ii])

            end # ii
        end # jj
        hdp.KK = count_active_clusters(nn)

        
        if sample_hyperparam
            M = zeros(Int, n_groups, KK_truncation)
            for hh in 1:n_internals
                for jj = 1:n_groups
                    for kk = 1:KK_truncation
                        if sum(nn[:, kk]) != 0
                            if nn[jj, kk] == 0
                                M[jj, kk] = 0
                            else
                                rr = zeros(Float64, nn[jj, kk])
                                for mm = 1:nn[jj, kk]
                                    rr[mm] = log(snumbers[nn[jj, kk], mm]) + mm*log(hdp.aa * my_beta[kk])
                                end
                                lognormalize!(rr)
                                M[jj, kk] = sample(rr)
                            end
                        end
                    end # kk
                end # n_groups

                m = sum(M)
                sample_hyperparam!(hdp, n_group_j, m)
            end # n_internals
        end # sample_hyperparam       


        # 4
        # resample my_beta
        for kk = 1:KK_truncation-1
            gamma1 = 1 + sum(nn[:, kk])
            gamma2 = hdp.gg + sum(nn[:, kk+1:KK_truncation])
            beta_tilde[kk] = rand(Distributions.Beta(gamma1, gamma2))
        end
        beta_tilde[KK_truncation] = 1.0
        my_beta = stick_breaking(beta_tilde)

        # 5
        # resample pi_tilde
        for jj = 1:n_groups
            for kk = 1:KK_truncation-1
                gamma1 = hdp.aa * my_beta[kk] + nn[jj, kk]
                gamma2 = hdp.aa * (1 - sum(my_beta[1:kk])) + sum(nn[jj, kk+1:KK_truncation])
                pi_tilde[jj, kk] = rand(Distributions.Beta(gamma1, gamma2))
            end
            pi_tilde[jj, KK_truncation] = 1.0
        end


        elapsed_time = toq()

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))

            push!(KK_list, hdp.KK)
            KK_dict[hdp.KK] = deepcopy(zz)


            if sample_n % store_every == 0
                normalized_filename = normpath(joinpath(results_path, filename))
                storesample(hdp, KK_list, KK_dict, sample_n, normalized_filename)
            end
        end
    end # iteration

        sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))
        normalized_filename = normpath(joinpath(results_path, filename))
        storesample(hdp, KK_list, KK_dict, sample_n, normalized_filename)

    KK_list, KK_dict
end


function posterior{T1, T2}(
    hdp::HDP{T1},
    xx::Vector{Vector{T2}},
    KK_dict::Dict{Int, Vector{Vector{Int}}},
    KK::Int)

end


## --- synthesizing the data --- ##
true_gg     = 1.0
true_aa     = 0.5
n_groups    = 10
n_group_j   = 100 * ones(Int, n_groups)
join_tables = true

true_tji, true_njt, true_kjt, true_nn, true_mm, true_zz, true_KK = BNP.gen_CRF_data(n_group_j, true_gg, true_aa, join_tables)

vv = 0.001          # fixed variance
ss = 2
true_atoms = [Gaussian1D(ss*kk, vv) for kk = 1:true_KK]

xx = Array(Vector{Float64}, n_groups)
for jj = 1:n_groups
    xx[jj] = zeros(Float64, n_group_j[jj])
    for ii = 1:n_group_j[jj]
        kk = true_zz[jj][ii]
        xx[jj][ii] = sample(true_atoms[kk])
    end
end


## ------- inference -------- ##
# constructing the Bayesian component of LDA model
m0 = mean(mean(xx))
v0 = 10.0
q0 = Gaussian1DGaussian1D(m0, v0, vv)

# constructing the HDP model
hdp_KK_init = 5
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
store_every = 10
results_path = ""
filename    = "demo_HDP_Gaussian1DGaussian1D_"


KK_list, KK_dict = truncated_gibbs_sampler(hdp, KK_truncation, xx, zz, 
    n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, results_path, filename)