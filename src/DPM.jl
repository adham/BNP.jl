#=
DPM.jl

10/27/2015
Adham Beyki, odinay@gmail.com
=#

#############################################
###### Dirichlet Process Mixture Model ######
#############################################
type DPM{T} <: MixtureModel
    component::T
    KK::Int
    aa::Float64
    aa1::Float64
    aa2::Float64

    DPM{T}(c::T, KK::Int, aa::Float64, aa1::Float64, aa2::Float64) = new(c, KK, aa, aa1, aa2)
end
DPM{T}(c::T, KK::Int, aa::Real, aa1::Real, aa2::Real) = DPM{typeof(c)}(c, KK, convert(Float64, aa),
    convert(Float64, aa1), convert(Float64, aa2))


function Base.show(io::IO, dpm::DPM)
    println(io, "Dirichlet Process Mixture Model with $(dpm.KK) $(typeof(dpm.component)) components")
end


"""
resampling concentration parameter based on Escobar and West 1995
"""
function sample_hyperparam!(dpm::DPM, NN::Int, iters::Int)


    for n = 1:iters
        eta = rand(Distributions.Beta(dpm.aa+1, NN))
        rr = (dpm.aa1+dpm.KK-1) / (n*(dpm.aa2-log(eta)))
        pi_eta = rr / (1.0+rr)

        if rand() < pi_eta
            dpm.aa = rand(Distributions.Gamma(dpm.aa1+dpm.KK)) / (dpm.aa2-log(eta))
        else
            dpm.aa = rand(Distributions.Gamma(dpm.aa1+dpm.KK-1)) / (dpm.aa2-log(eta))
        end
    end
end

count_active_clusters(nn::Vector{Int}) = length(find(x -> x>0, nn))




function storesample{T}(
    dpm::DPM{T},
    KK_list::Vector{Int},
    KK_dict::Dict{Int, Vector{Int}},
    sample_n::Int,
    filename::ASCIIString)

    if endswith(filename, "_")
        dummy_filename = string(filename, sample_n, ".h5")
    else
        dummy_filename = string(filename, "_", sample_n, ".h5")
    end

    KK_dict_keys = keys(KK_dict)

    println("storing on disk...")
    HDF5.h5open(dummy_filename, "w") do file
        for kk in KK_dict_keys
            HDF5.write(file, "zz/$(kk)", KK_dict[kk])
        end
        HDF5.write(file, "sample_n", sample_n)
    end
end





function collapsed_gibbs_sampler{T1, T2}(
    dpm::DPM{T1},
    xx::Vector{T2},
    zz::Vector{Int},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true, n_internals::Int=10,
    store_every::Int=250, results_path="", filename::ASCIIString="DPM_results")



    n_iterations = n_burnins + (n_samples)*(n_lags+1)
    NN = length(xx)
    nn = zeros(Int, dpm.KK)
    log_likelihood = 0.0


    components = Array(typeof(dpm.component), dpm.KK)
    for kk = 1:dpm.KK
        components[kk] = deepcopy(dpm.component)
    end


    KK_list = Int[]
    KK_dict = Dict{Int, Vector{Int}}()

    # initializing the model
    tic()
    for ii = 1:NN
        kk = zz[ii]
        additem!(components[kk], xx[ii])
        nn[kk] += 1
        log_likelihood += loglikelihood(components[kk], xx[ii])
    end
    push!(KK_list, dpm.KK)
    elapsed_time = toq()


    #=
    starting the MCMC chain
    1. remote the data point
    2. sample for zz
    3. add the datapoint back
    =#

    for iteration = 1:n_iterations

        # verbose
        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, aa=%.2f, time=%.2f, likelihood=%.2f",
            iteration, dpm.KK, indmax(hist(KK_list, .5:maximum(KK_list)+0.5)[2]),
            dpm.aa, elapsed_time, log_likelihood))
        log_likelihood = 0.0

        tic()
        @inbounds for ii = randperm(NN)

            # 1
            kk = zz[ii]
            delitem!(components[kk], xx[ii])
            nn[kk] -= 1

            # remove the cluster if it is empty
            if nn[kk] == 0
                println("\tcomponent $kk has become inactive")
                splice!(nn, kk)
                splice!(components, kk)
                dpm.KK -= 1

                # shifting the labels one cluster back
                idx = find(x -> x>kk, zz)
                zz[idx] -= 1
            end

            # 2
            pp = zeros(Float64, dpm.KK+1)
            @inbounds for kk = 1:dpm.KK
                pp[kk] = log(nn[kk]) + logpredictive(components[kk], xx[ii])
            end
            pp[dpm.KK+1] = log(dpm.aa) + logpredictive(dpm.component, xx[ii])
            lognormalize!(pp)
            kk = sample(pp)

            # instanciate a new component if needed
            if kk == dpm.KK+1
                println("\tcomponent $(kk) activated.")
                push!(components, deepcopy(dpm.component))
                push!(nn, 0)
                dpm.KK += 1
            end

            # 3
            zz[ii] = kk
            nn[kk] += 1
            additem!(components[kk], xx[ii])
            log_likelihood += loglikelihood(components[kk], xx[ii])
        end # ii

        # sample the hyperparameter
        if sample_hyperparam
            sample_hyperparam!(dpm, NN, n_internals)
        end
        elapsed_time = toq()

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))
            push!(KK_list, dpm.KK)
            KK_dict[dpm.KK] = deepcopy(zz)

            if sample_n % store_every == 0
                normalized_filename = normpath(joinpath(results_path, filename))
                storesample(dpm, KK_list, KK_dict, sample_n, normalized_filename)
            end
        end
    end # iteration

    sample_n = convert(Int, (n_iterations-n_burnins)/(n_lags+1))
    normalized_filename = normpath(joinpath(results_path, filename))
    storesample(dpm, KK_list, KK_dict, sample_n, normalized_filename)

    KK_list, KK_dict
end



function truncated_gibbs_sampler{T1, T2}(
    dpm::DPM{T1},
    KK_truncation::Int,
    xx::Vector{T2},
    zz::Vector{Int},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true, n_internals::Int=10,
    store_every::Int=100, results_path="", filename::ASCIIString="DPM_results_")

    n_iterations = n_burnins + (n_samples)*(n_lags+1)
    NN = length(xx)
    nn = zeros(Int, KK_truncation)
    pp = zeros(Float64, KK_truncation)
    log_likelihood = 0.0


    components = Array(typeof(dpm.component), KK_truncation)
    for kk = 1:KK_truncation
        components[kk] = deepcopy(dpm.component)
    end

    KK_list = Int[]
    KK_dict = Dict{Int, Vector{Int}}()

    tic()
    for ii = 1:NN
        kk = zz[ii]
        additem!(components[kk], xx[ii])
        nn[kk] += 1
        log_likelihood += loglikelihood(components[kk], xx[ii])
    end
    push!(KK_list, count_active_clusters(nn))


    # beta variables for mixing proprtions, Ishwaran and James 2001
    pi_tilde = ones(Float64, KK_truncation)
    for kk = 1:KK_truncation-1
        gamma1 = 1 + nn[kk]
        gamma2 = dpm.aa + sum(nn[kk+1:KK_truncation])
        pi_tilde[kk] = gamma1 / (gamma1 + gamma2)
    end
    pi_tilde[KK_truncation] = 1.0

    elapsed_time = toq()


    #########################################################
    ################ starting the MCMC chain ################
    #########################################################

    for iteration = 1:n_iterations

        # Verbose
        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, aa=%.2f, time=%.2f, likelihood=%.2f", 
            iteration, dpm.KK, indmax(hist(KK_list, .5:maximum(KK_list)+0.5)[2]), dpm.aa, elapsed_time, log_likelihood))
        log_likelihood = 0.0
         

        tic()
        for ii = randperm(NN)

            # 1
            # remove the point
            kk = zz[ii]
            delitem!(components[kk], xx[ii])
            nn[kk] -= 1


            # 2
            # sample zz
            for kk = 1:KK_truncation
                pp[kk] = log(pi_tilde[kk]) + sum(log(1-pi_tilde[1:kk-1])) + logpredictive(components[kk], xx[ii])
            end
            lognormalize!(pp)
            kk = sample(pp)


            # 3
            # add the point to the newly resampled cluster
            zz[ii] = kk
            nn[kk] += 1
            additem!(components[kk], xx[ii])
            log_likelihood += loglikelihood(components[kk], xx[ii])
        end


        # 4
        # sample pi_tilde
        for kk = 1:KK_truncation-1
            gamma1 = 1 + nn[kk]
            gamma2 = dpm.aa + sum(nn[kk+1:KK_truncation])
            pi_tilde[kk] = rand(Distributions.Beta(gamma1, gamma2))
        end
        pi_tilde[KK_truncation] = 1.0

        dpm.KK = count_active_clusters(nn)

        
        # 5
        # sample the hyperparameter
        if sample_hyperparam
            sample_hyperparam!(dpm, NN, n_internals)
        end
        elapsed_time = toq()


        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))

            push!(KK_list, dpm.KK)
            KK_dict[dpm.KK] = deepcopy(zz)

            if sample_n % store_every == 0
                normalized_filename = normpath(joinpath(results_path, filename))
                storesample(dpm, KK_list, KK_dict, sample_n, normalized_filename)
            end
        end

    end # iteration


    sample_n = convert(Int, (n_iterations-n_burnins)/(n_lags+1))
    normalized_filename = normpath(joinpath(results_path, filename))
    storesample(dpm, KK_list, KK_dict, sample_n, normalized_filename)

    KK_list, KK_dict
end



#######################
###    Posterior    ###
#######################

# posterior for DPM with collapsed Gibbs sampling
function posterior{T1, T2}(
    dpm::DPM{T1},
    xx::Vector{T2},
    KK_dict::Dict{Int, Vector{Int}},
    KK::Int)


    components = Array(typeof(dpm.component), KK)
    for kk = 1:KK
        components[kk] = deepcopy(dpm.component)
    end

    zz = KK_dict[KK]
    NN = length(xx)
    nn = zeros(Int, KK)

    # since in truncated Gibbs sampling we might have a case where the sampler jumps over
    # a cluster and doesn't allocate any point to it, we have to map them back
    zz_unq = sort(unique(zz))
    for ii = 1:NN
        kk = find(x -> x==zz[ii], zz_unq)[1]
        additem!(components[kk], xx[ii])
        nn[kk] += 1
    end

    return([posterior(components[kk]) for kk=1:KK], nn)
end