#=
LDA.jl

Adham Beyki, odinay@gmail.com
27/10/2015
=#

##########################################
###### Latent Dirichlet Allocation ######
##########################################
immutable LDA{T} <: hMixtureModel
    component::T
    KK::Int
    aa::Float64

    LDA{T}(c::T, KK::Int, aa::Float64) = new(c, KK, aa)
end
LDA{T}(c::T, KK::Int, aa::Real) = LDA{typeof(c)}(c, KK, convert(Float64, aa))


function Base.show(io::IO, lda::LDA)
    println(io, "Latent Dirichlet Allocation model with $(lda.KK) $(typeof(lda.component)) components")
end


function storesample(zz::Vector{Vector{Int}}, sample_n::Int, filename::ASCIIString)

    n_groups = length(zz)
    n_group_j = [length(zz[jj]) for jj=1:n_groups]
    println(typeof(n_group_j))
    zz_flat = Int[]
    for jj = 1:n_groups
        append!(zz_flat, zz[jj])
    end

    if endswith(filename, "_")
        dummy_filename = string(filename, sample_n, ".h5")
    else
        dummy_filename = string(filename, "_", sample_n, ".h5")
    end

    println("storing on disk...")
    HDF5.h5open(dummy_filename, "w") do file
        HDF5.write(file, "n_groups", n_groups)
        HDF5.write(file, "n_group_j", n_group_j)
        HDF5.write(file, "zz_flat", zz_flat)
        HDF5.write(file, "sample_n", sample_n)
    end
end


function collapsed_gibbs_sampler{T1, T2}(
    lda::LDA{T1},
    xx::Vector{Vector{T2}},
    zz::Vector{Vector{Int}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    store_every::Int=250, results_path="", filename::ASCIIString="LDA_results_")


    components = Array(typeof(lda.component), lda.KK)
    for kk = 1:lda.KK
        components[kk] = deepcopy(lda.component)
    end

    n_groups       = length(xx)
    n_group_j      = [length(zz[jj]) for jj = 1:n_groups]
    n_iterations   = n_burnins + (n_samples)*(n_lags+1)
    lda_aa         = fill(lda.aa, lda.KK)
    pp             = zeros(Float64, lda.KK)
    nn             = zeros(Int, n_groups, lda.KK)
    log_likelihood = 0.0


    # initializing the components
    tic()
    for jj = 1:n_groups
        for ii = 1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            nn[jj, kk] += 1
            log_likelihood += loglikelihood(components[kk], xx[jj][ii])
        end
    end
    elapsed_time = toq()


    # starting the MCMC chain
    for iteration = 1:n_iterations

        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, time=%.2f, likelihood=%.2f", iteration, lda.KK, elapsed_time, log_likelihood))
        log_likelihood = 0.0

        tic()
        @inbounds for jj = randperm(n_groups)
            @inbounds for ii = randperm(n_group_j[jj])

                # 1
                # remove the datapoint
                kk = zz[jj][ii]
                delitem!(components[kk], xx[jj][ii])
                nn[jj, kk] -= 1

                # 2
                # sample zz
                @inbounds for kk = 1:lda.KK
                    pp[kk] = log(nn[jj, kk] + lda.aa) + logpredictive(components[kk], xx[jj][ii])
                end

                lognormalize!(pp)
                kk = sample(pp)

                # 3
                # add the datapoint to the newly sampled cluster
                zz[jj][ii] = kk
                additem!(components[kk], xx[jj][ii])
                nn[jj, kk] += 1
                log_likelihood += loglikelihood(components[kk], xx[jj][ii])
            end #ii
        end # jj
        elapsed_time = toq()

        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))

            if sample_n % store_every == 0
                normalized_filename = normpath(joinpath(results_path, filename))            
                storesample(zz, sample_n, normalized_filename)
            end
        end
    end # iteration

    sample_n = convert(Int, (n_iterations-n_burnins)/(n_lags+1))
    normalized_filename = normpath(joinpath(results_path, filename))            
    storesample(zz, sample_n, normalized_filename)
end


function posterior{T1, T2}(lda::LDA{T1}, xx::Vector{Vector{T2}}, zz::Vector{Vector{Int}})

    components = Array(typeof(lda.component), lda.KK)
    for kk = 1:lda.KK
        components[kk] = deepcopy(lda.component)
    end
    
    n_groups = length(xx)
    n_group_j = [length(zz[jj]) for jj = 1:n_groups]
    nn = zeros(Int, n_groups, lda.KK)
    

    for jj = 1:n_groups
        for ii = 1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            nn[jj, kk] += 1
        end
    end

    return([posterior(components[kk]) for kk =1:lda.KK], nn)
end
