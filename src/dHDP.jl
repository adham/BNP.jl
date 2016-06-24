#=
dHDP.jl

30/10/2015
Adham Beyki, odinay@gmail.com
=#
############################################################
##  dynamic Hierarchical Dirichlet Process Mixture Models ##
############################################################
type dHDP{T} <: hMixtureModel
    component::T
    KK::Int
    gg::Float64
    g1::Float64
    g2::Float64
    aa::Float64
    a1::Float64
    a2::Float64
    aw::Float64
    bw::Float64

    dHDP{T}(c::T, KK::Int, gg::Float64, g1::Float64, g2::Float64, aa::Float64,
        a1::Float64, a2::Float64, aw::Float64, bw::Float64) = new(c, KK, gg, g1, g2, aa, a1, a2, aw, bw)
end
dHDP{T}(c::T, KK::Int, gg::Real, g1::Real, g2::Real, aa::Real, a1::Real, a2::Real, aw::Real, bw::Real) =
    dHDP{typeof(c)}(c, KK, convert(Float64, gg), convert(Float64, g1), convert(Float64, g2),
    convert(Float64, aa), convert(Float64, a1), convert(Float64, a2), convert(Float64, aw), convert(Float64, bw))


function Base.show(io::IO, dhdp::dHDP)
    println(io, "dynamic Hierarchical Dirichlet Process Mixture Model with $(dhdp.KK) $(typeof(dhdp.component)) components")
end


function storesample{T}(
    dhdp::dHDP{T},
    KK_list::Vector{Int},
    KK_dict::Dict{Int, Vector{Vector{Int}}},
    w_tilde::Vector{Float64},
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
        HDF5.write(file, "w_tilde", w_tilde)
        HDF5.write(file, "sample_n", sample_n)
        HDF5.write(file, "n_groups", n_groups)
        HDF5.write(file, "n_group_j", n_group_j)
    end
end



function sample_hyperparam!(dhdp::dHDP, n_group_j::Vector{Int}, m::Int)
    #= NOTE
    resampling the group level concentration parameter α0 using auxiliary variables
    w and s, Eq. 50 Teh etal 04 the Gamma distribution in Eq.50 is expressed using
    shape and rate. We have to rescale them to shape and scale representation to be
    able to use standard random gamma functions in Julia to draw from it. Also:
    Gamma(a, 1/b) = Gamma(a) / b
    =#

    n_groups = length(n_group_j)
    w = zeros(Float64, n_groups)
    for jj = 1:n_groups
        w[jj] = rand(Distributions.Beta(dhdp.aa+1, n_group_j[jj]))
    end
    p = n_group_j / dhdp.aa
    p ./= (p+1.0)

    s = zeros(Int, n_groups)
    for jj = 1:n_groups
        s[jj] = rand(Distributions.Binomial(1, p[jj]))
    end

    aa_shape = dhdp.a1 + m - sum(s)
    aa_rate  = dhdp.a2 - sum(log(w))
    dhdp.aa = rand(Distributions.Gamma(aa_shape)) / aa_rate

    # resampling the top level concentration parameter γ, Escobar and West 95
    eta = rand(Distributions.Beta(dhdp.gg+1, m))
    rr = (dhdp.g1 + dhdp.KK - 1) / (m*(dhdp.g2 - log(eta)))
    pi_eta = rr / (1.0 + rr)

    if rand() < pi_eta
        dhdp.gg = rand(Distributions.Gamma(dhdp.g1 + dhdp.KK)) / (dhdp.g2-log(eta))
    else
        dhdp.gg = rand(Distributions.Gamma(dhdp.g1+dhdp.KK-1)) / (dhdp.g2-log(eta))
    end
end # sample_hyperparam



function truncated_gibbs_sampler{T1, T2}(
    dhdp::dHDP{T1},
    KK_truncation::Int,
    xx::Vector{Vector{T2}},
    zz::Vector{Vector{Int}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true, n_internals::Int=10,
    store_every::Int=250, results_path="", filename::ASCIIString="dHDP_results_")

    BETA_THRESHOLD = 0.0001

    
    n_iterations    = n_burnins + (n_samples)*(n_lags+1)
    n_groups        = length(xx)
    n_group_j       = zeros(Int, n_groups)
    pp              = zeros(Float64, KK_truncation)
    log_likelihood  = 0.0

    for jj = 1:n_groups
        n_group_j[jj] = length(zz[jj])
    end


    components = Array(typeof(dhdp.component), KK_truncation)
    for kk = 1:KK_truncation
        components[kk] = deepcopy(dhdp.component)
    end

    KK_list = Int[]
    KK_dict = Dict{Int, Vector{Vector{Int}}}()

    beta_tilde = zeros(Float64, KK_truncation)
    pi_tilde   = zeros(Float64, n_groups, KK_truncation)
    logpi      = zeros(Float64, n_groups, KK_truncation)

    snumbers_file = string(Pkg.dir(), "\\BNP\\src\\StirlingNums_10K.mat")
    snumbers_data = MAT.matread(snumbers_file)
    snumbers = snumbers_data["snumbersNormalizedSparse"]



    ###################################################################
    #                      Initializing the model                     #
    ###################################################################
    # 1. add the data points to the components
    # 2. construct w_tilde
    # 3. construct β
    # 4. construct pi_tilde
    # 5. construct rr


    tic()
    print_with_color(:red, "\nInitializing the model\n")


    # 1
    # adding the data points
    for jj = 1:n_groups
        for ii = 1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            log_likelihood += loglikelihood(components[kk], xx[jj][ii])
        end
    end


    # 2
    # constructing w_tilde
    w_tilde = fill(dhdp.aw / (dhdp.aw + dhdp.bw), n_groups-1)
    insert!(w_tilde, 1, 1.0)


    # 4
    # construncting rr
    rr = Array(Vector{Int}, n_groups)
    for jj = 1:n_groups
        rr[jj] = rand(1:jj, n_group_j[jj])
    end

    nrz = zeros(Int, n_groups, KK_truncation)
    for jj=1:n_groups
        for ii=1:n_group_j[jj]
            nrz[rr[jj][ii], zz[jj][ii]] += 1
        end
    end
    push!(KK_list, count_active_clusters(nrz))


    # 3
    # constructing my_beta
    for kk = 1:KK_truncation-1
        gamma1 = 1 + sum(nrz[:, kk])
        gamma2 = dhdp.gg + sum(nrz[:, kk+1:KK_truncation])
        beta_tilde[kk] = gamma1 / (gamma1 + gamma2)
    end
    beta_tilde[KK_truncation] = 1.0
    my_beta = stick_breaking(beta_tilde)

    

    
    # 5
    # constructing beta variables for pi_j
    for ll = 1:n_groups
        for kk = 1:KK_truncation-1
            gamma1 = dhdp.aa * my_beta[kk] + nrz[ll, kk]
            gamma2 = dhdp.aa * (1 - sum(my_beta[1:kk])) + sum(nrz[ll, kk+1:KK_truncation])
            pi_tilde[ll, kk] = gamma1 / (gamma1 + gamma2)
        end
        pi_tilde[ll, KK_truncation] = 1.0
        logpi[ll, :] = log_stick_breaking(pi_tilde[ll, :][:])
    end

    

    elapsed_time = toq()






    ###################################################################
    #                    starting the MCMC chain                      #
    ###################################################################

    for iteration = 1:n_iterations

        # Verbose
        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, gg=%.2f, aa=%.2f, time=%.2f, likelihood=%.2f",
            iteration, dhdp.KK, indmax(hist(KK_list, .5:maximum(KK_list)+0.5)[2]), dhdp.gg, dhdp.aa,
            elapsed_time, log_likelihood))
        log_likelihood = 0.0


        tic()

        logww_prime = [0.0]
        for jj = 1:n_groups

            if jj == 1
                logww = [0.0]
            else
                logww = zeros(Float64, jj)
                logww[1:jj-1] = logww_prime + log(1 - w_tilde[jj])
                logww[jj] = log(w_tilde[jj])
                logww_prime = logww
            end

            for ii = 1:n_group_j[jj]


                # sample rr
                kk     = zz[jj][ii]
                rr_old = rr[jj][ii]
                
                pp_rr = zeros(Float64, jj)
                for ll = 1:jj
                    pp_rr[ll] = logww[ll] + logpi[ll, kk]
                end 
                lognormalize!(pp_rr)
                rr[jj][ii] = sample(pp_rr)


                # sample zz
                delitem!(components[kk], xx[jj][ii])

                pp_zz = zeros(Float64, KK_truncation)
                for kk = 1:KK_truncation
                    pp_zz[kk] = logpi[rr[jj][ii], kk] + logpredictive(components[kk], xx[jj][ii])
                end

                lognormalize!(pp_zz)
                kk = sample(pp_zz)

                zz[jj][ii] = kk
                additem!(components[kk], xx[jj][ii])
                log_likelihood += loglikelihood(components[kk], xx[jj][ii])

            end # ii
        end # jj


        # construct njr and nrz
        # njr[jj, rr] is the count of draws from random measure rr in group jj
        # nrz[rr, kk] is the count of draws from component kk of measure rr
        njr = zeros(Int, n_groups, n_groups)
        nrz = zeros(Int, n_groups, KK_truncation)
        for jj=1:n_groups
            for ii=1:n_group_j[jj]
                njr[jj, rr[jj][ii]] += 1
            end
        end
        njr = reverse_cumsum(njr)

        for jj=1:n_groups
            for ii=1:n_group_j[jj]
                nrz[rr[jj][ii], zz[jj][ii]] += 1
            end
        end
        dhdp.KK = count_active_clusters(nrz)

        # resample beta_tilde
        for kk = 1:KK_truncation-1
            gamma1 = 1 + sum(nrz[:, kk])
            gamma2 = dhdp.gg + sum(nrz[:, kk+1:KK_truncation])
            beta_tilde[kk] = rand(Distributions.Beta(gamma1, gamma2))
        end
        beta_tilde[KK_truncation] = 1.0
        my_beta = stick_breaking(beta_tilde)

        # resampling w_tilde, Eq.15 Ren etal 2008
        for ll=1:n_groups-1
            # left  = dhdp.aw + sum(njr[ll+1:n_groups, ll+1])
            # right = dhdp.bw + sum(njr[ll+1:n_groups, 1:ll])
            left  = dhdp.aw + njr[ll, ll+1]
            right = dhdp.bw + sum(njr[ll, 1:ll])
            w_tilde[ll+1] = rand(Distributions.Beta(left, right))
        end
        
        for ll = 1:n_groups
            for kk = 1:KK_truncation-1
                left  = dhdp.aa * my_beta[kk] + nrz[ll, kk]
                right = dhdp.aa * (1-sum(my_beta[1:kk])) + sum(nrz[ll, kk+1:KK_truncation])

                if left < BETA_THRESHOLD || right < BETA_THRESHOLD
                    pi_tilde[ll, kk] = 0.0
                else
                    pi_tilde[ll, kk] = rand(Distributions.Beta(left, right))
                end
            end
            pi_tilde[ll, KK_truncation] = 1.0
            logpi[ll, :] = log_stick_breaking(pi_tilde[ll, :][:])
        end

        if sample_hyperparam
            M = zeros(Int, n_groups, KK_truncation)
            for hh in 1:n_internals
                for jj = 1:n_groups
                    for kk = 1:KK_truncation
                        if sum(nrz[:, kk]) != 0
                            if nrz[jj, kk] == 0
                                M[jj, kk] = 0
                            else
                                pp_rr = zeros(Float64, nrz[jj, kk])
                                for mm = 1:nrz[jj, kk]
                                    pp_rr[mm] = log(snumbers[nrz[jj, kk], mm]) + mm*log(dhdp.aa * my_beta[kk])
                                end
                                lognormalize!(pp_rr)
                                M[jj, kk] = sample(pp_rr)
                            end
                        end
                    end # kk
                end # n_groups

                m = sum(M)
                sample_hyperparam!(dhdp, n_group_j, m)
            end # n_internals
        end # sample_hyperparam



        elapsed_time = toq()

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))

            push!(KK_list, dhdp.KK)
            KK_dict[dhdp.KK] = deepcopy(zz)


            if sample_n % store_every == 0
                normalized_filename = normpath(joinpath(results_path, filename))
                storesample(dhdp, KK_list, KK_dict, w_tilde, sample_n, normalized_filename)
            end
        end
    end # iteration

    
    sample_n = convert(Int, (n_iterations-n_burnins)/(n_lags+1))
    normalized_filename = normpath(joinpath(results_path, filename))
    storesample(dhdp, KK_list, KK_dict, w_tilde, sample_n, normalized_filename)


    # before returning KK_dict, flatten zz to make it consistent
    # with the stored results
    KK_dict_flat = Dict{Int, Vector{Int}}()
    KK_dict_keys = collect(keys(KK_dict))
    for kk in KK_dict_keys
        zz_flat = Int[]
        for jj=1:n_groups
            append!(zz_flat, KK_dict[kk][jj])
        end
        KK_dict_flat[kk] = zz_flat
    end


    KK_list, KK_dict_flat, w_tilde
end



function truncated_gibbs_sampler2{T1, T2}(
    dhdp::dHDP{T1},
    xx::Vector{Vector{T2}},
    zz::Vector{Vector{Int}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true, n_internals::Int=10,
    store_every::Int=250, results_path="", filename::ASCIIString="dHDP_results_")
    

    BETA_THRESHOLD = 0.0001

    n_iterations    = n_burnins + (n_samples)*(n_lags+1)
    n_groups        = length(xx)
    n_group_j       = zeros(Int, n_groups)
    log_likelihood  = 0.0
    KK_flag         = false


    for jj = 1:n_groups
        n_group_j[jj] = length(zz[jj])
    end


    components = Array(typeof(dhdp.component), dhdp.KK)
    for kk = 1:dhdp.KK
        components[kk] = deepcopy(dhdp.component)
    end

    KK_list = Int[]
    KK_dict = Dict{Int, Vector{Vector{Int}}}()


    snumbers_file = string(Pkg.dir(), "\\BNP\\src\\StirlingNums_10K.mat")
    snumbers_data = MAT.matread(snumbers_file)
    snumbers = snumbers_data["snumbersNormalizedSparse"]



    tic()
    print_with_color(:red, "\nInitializing the model\n")
  
    for jj = 1:n_groups
        for ii = 1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            log_likelihood += loglikelihood(components[kk], xx[jj][ii])
        end
    end
    push!(KK_list, dhdp.KK)

    w_tilde = fill(dhdp.aw / (dhdp.aw + dhdp.bw), n_groups-1)
    insert!(w_tilde, 1, 1.0)


    my_beta = ones(Float64, dhdp.KK+1) / (dhdp.KK+1)


    rr = Array(Vector{Int}, n_groups)
    for jj = 1:n_groups
        rr[jj] = rand(1:jj, n_group_j[jj])
    end

    
    # njr[jj, rr] is the count of draws from random measure rr in group jj
    # nrz[rr, kk] is the count of draws from component kk of measure rr
    njr = zeros(Int, n_groups, n_groups)
    nrz = zeros(Int, n_groups, dhdp.KK)
    for jj=1:n_groups
        for ii=1:n_group_j[jj]
            njr[jj, rr[jj][ii]] += 1
        end
    end
    njr = reverse_cumsum(njr)

    for jj=1:n_groups
        for ii=1:n_group_j[jj]
            nrz[rr[jj][ii], zz[jj][ii]] += 1
        end
    end

    pi_tilde   = zeros(Float64, n_groups, dhdp.KK+1)
    logpi      = zeros(Float64, n_groups, dhdp.KK+1)
    for ll = 1:n_groups
        for kk = 1:dhdp.KK
            gamma1 = dhdp.aa * my_beta[kk] + nrz[ll, kk]
            gamma2 = dhdp.aa * (1 - sum(my_beta[1:kk])) + sum(nrz[ll, kk+1:dhdp.KK])
            pi_tilde[ll, kk] = gamma1 / (gamma1 + gamma2)        
        end
        pi_tilde[ll, dhdp.KK+1] = 1.0
        logpi[ll, :] = log_stick_breaking(pi_tilde[ll, :][:])
    end

    elapsed_time = toq()


    for iteration = 1:n_iterations

        # Verbose
        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, gg=%.2f, aa=%.2f, time=%.2f, likelihood=%.2f",
            iteration, dhdp.KK, indmax(hist(KK_list, .5:maximum(KK_list)+0.5)[2]), dhdp.gg, dhdp.aa,
            elapsed_time, log_likelihood))
        log_likelihood = 0.0


        tic()

        logww_prime = [0.0]
        for jj = 1:n_groups

            if jj == 1
                logww = [0.0]
            else
                logww = zeros(Float64, jj)
                logww[1:jj-1] = logww_prime + log(1 - w_tilde[jj])
                logww[jj] = log(w_tilde[jj])
                logww_prime = logww
            end

            for ii = 1:n_group_j[jj]

                # sample rr
                kk     = zz[jj][ii]
                rr_old = rr[jj][ii]
                nrz[rr_old, kk] -= 1
                
                pp_rr = zeros(Float64, jj)
                for ll = 1:jj
                    pp_rr[ll] = logww[ll] + logpi[ll, kk]
                end 
                lognormalize!(pp_rr)
                rr_new = sample(pp_rr)
                
                rr[jj][ii] = rr_new
                nrz[rr_new, kk] += 1

                # sample zz
                delitem!(components[kk], xx[jj][ii])
                nrz[rr_new, kk] -= 1

                if sum(nrz[:, kk]) == 0
                    println("\tcomponent $kk has become inactive")
                    nrz = del_column(nrz, kk)
                    splice!(components, kk)

                    for ll = 1:n_groups
                        idx = find(x -> x>kk, zz[ll])
                        zz[ll][idx] -= 1
                    end

                    my_beta[dhdp.KK+1] += my_beta[kk]
                    splice!(my_beta, kk)
                    
                    for ll = 1:n_groups
                        logpi[ll, dhdp.KK+1] = log(exp(logpi[ll, dhdp.KK+1]) + exp(logpi[ll, kk]))
                    end
                    
                    pi_tilde = del_column(pi_tilde, kk)
                    logpi    = del_column(logpi, kk)

                    dhdp.KK -= 1
                end


                pp_zz = zeros(Float64, dhdp.KK+1)
                for kk = 1:dhdp.KK
                    pp_zz[kk] = logpi[rr[jj][ii], kk] + logpredictive(components[kk], xx[jj][ii])
                end
                pp_zz[dhdp.KK+1] = logpi[rr[jj][ii], kk] + logpredictive(dhdp.component, xx[jj][ii])

                lognormalize!(pp_zz)
                kk = sample(pp_zz)

                if kk == dhdp.KK+1

                    KK_flag = true

                    println("\tcomponents $(kk) activated.")
                    push!(components, deepcopy(dhdp.component))
                    nrz = add_column(nrz)

                    b = rand(Distributions.Beta(1, dhdp.gg))
                    b_new = my_beta[dhdp.KK+1]
                    my_beta[dhdp.KK+1] = b * b_new
                    push!(my_beta, (1-b)*b_new)

                    dhdp.KK += 1
                end

                zz[jj][ii] = kk
                nrz[rr_new, kk] += 1
                additem!(components[kk], xx[jj][ii])
                log_likelihood += loglikelihood(components[kk], xx[jj][ii])
                

                if KK_flag
                    pi_tilde = add_column(pi_tilde)
                    logpi = add_column(logpi)

                    for ll = 1:n_groups
                        gamma1 = dhdp.aa * my_beta[dhdp.KK] + nrz[ll, dhdp.KK]
                        gamma2 = dhdp.aa * (1 - sum(my_beta[1:dhdp.KK]))
                        pi_tilde[ll, dhdp.KK] = rand(Distributions.Beta(gamma1, gamma2))
                        pi_tilde[ll, dhdp.KK+1] = 1.0

                        # Since only the following two items change, instead of log stick breaking 
                        # we could just update them, but for sum reason the result doesn't sum to one
                        # logpi[ll, dhdp.KK] = log(pi_tilde[ll, dhdp.KK]) + logpi[ll, dhdp.KK]
                        # logpi[ll, dhdp.KK+1] = log(1 - pi_tilde[ll, dhdp.KK]) + logpi[ll, dhdp.KK]
                        logpi[ll, :] = log_stick_breaking(pi_tilde[ll, :][:])
                    end

                    KK_flag = false
                end




            end # ii
        end # jj



           
        # construct njr and nrz
        # njr[jj, rr] is the count of draws from random measure rr in group jj
        # nrz[rr, kk] is the count of draws from component kk of measure rr
        njr = zeros(Int, n_groups, n_groups)
        nrz = zeros(Int, n_groups, dhdp.KK)
        for jj=1:n_groups
            for ii=1:n_group_j[jj]
                njr[jj, rr[jj][ii]] += 1
            end
        end
        njr = reverse_cumsum(njr)

        for jj=1:n_groups
            for ii=1:n_group_j[jj]
                nrz[rr[jj][ii], zz[jj][ii]] += 1
            end
        end

        # resampling w_tilde, Eq.15 Ren etal 2008
        for ll=1:n_groups-1
            # gamma1  = dhdp.aw + sum(njr[ll+1:n_groups, ll+1])
            # gamma2 = dhdp.bw + sum(njr[ll+1:n_groups, 1:ll])
            gamma1  = dhdp.aw + njr[ll, ll+1]
            gamma2 = dhdp.bw + sum(njr[ll, 1:ll])
            w_tilde[ll+1] = rand(Distributions.Beta(gamma1, gamma2))
        end
        

        M = zeros(Int, n_groups, dhdp.KK)
        for hh in 1:n_internals
            for jj = 1:n_groups
                for kk = 1:dhdp.KK
                    if nrz[jj, kk] == 0
                        M[jj, kk] = 0
                    else
                        pp_rr = zeros(Float64, nrz[jj, kk])
                        for mm = 1:nrz[jj, kk]
                            pp_rr[mm] = log(snumbers[nrz[jj, kk], mm]) + mm*log(dhdp.aa * my_beta[kk])
                        end
                        lognormalize!(pp_rr)
                        M[jj, kk] = sample(pp_rr)
                    end
                end # kk
            end # n_groups
            MM = convert(Vector{Float64}, vec(sum(M, 1)))
            push!(MM, dhdp.gg)
            my_beta = rand(Distributions.Dirichlet(MM))

            if sample_hyperparam
                m = sum(M)
                sample_hyperparam!(dhdp, n_group_j, m)
            end
        end # n_internals


        for ll = 1:n_groups
            for kk = 1:dhdp.KK
                gamma1 = dhdp.aa * my_beta[kk] + nrz[ll, kk]
                gamma2 = dhdp.aa * (1-sum(my_beta[1:kk])) + sum(nrz[ll, kk+1:dhdp.KK])

                if gamma1 < BETA_THRESHOLD || gamma2 < BETA_THRESHOLD
                    pi_tilde[ll, kk] = 0.0
                else
                    pi_tilde[ll, kk] = rand(Distributions.Beta(gamma1, gamma2))
                end
            end
            pi_tilde[ll, dhdp.KK+1] = 1.0
            logpi[ll, :] = log_stick_breaking(pi_tilde[ll, :][:])
        end


        elapsed_time = toq()

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))

            push!(KK_list, dhdp.KK)
            KK_dict[dhdp.KK] = deepcopy(zz)


            if sample_n % store_every == 0
                normalized_filename = normpath(joinpath(results_path, filename))
                storesample(dhdp, KK_list, KK_dict, w_tilde, sample_n, normalized_filename)
            end
        end
    end # iteration

    

    sample_n = convert(Int, (n_iterations-n_burnins)/(n_lags+1))
    normalized_filename = normpath(joinpath(results_path, filename))
    storesample(dhdp, KK_list, KK_dict, w_tilde, sample_n, normalized_filename)



    # before returning KK_dict, flatten zz to make it consistent
    # with the stored results
    KK_dict_flat = Dict{Int, Vector{Int}}()
    KK_dict_keys = collect(keys(KK_dict))
    for kk in KK_dict_keys
        zz_flat = Int[]
        for jj=1:n_groups
            append!(zz_flat, KK_dict[kk][jj])
        end
        KK_dict_flat[kk] = zz_flat
    end


    KK_list, KK_dict_flat, w_tilde
end


function posterior{T1, T2}(
    dhdp::dHDP{T1},
    xx::Vector{Vector{T2}},
    zz_flat::Vector{Int},
    KK::Int)


    n_groups = length(xx)
    nn = zeros(Int, n_groups, KK)
    n_group_j = Array(Int, n_groups)

    for jj = 1:n_groups
        n_group_j[jj] = length(xx[jj])
    end

    n_group_cs = cumsum(n_group_j)
    zz = Array(Vector{Int}, n_groups)
    zz[1] = zz_flat[1:n_group_cs[1]]
    for jj = 2:n_groups
        zz[jj] = zz_flat[n_group_cs[jj-1]+1:n_group_cs[jj]]
    end

    components = Array(typeof(dhdp.component), KK)
    for kk = 1:KK
        components[kk] = deepcopy(dhdp.component)
    end

    zz_unq = sort(unique(zz_flat))
    for jj = 1:n_groups
        for ii = 1:n_group_j[jj]
            kk = find(x -> x==zz[jj][ii], zz_unq)[1]
            additem!(components[kk], xx[jj][ii])
            nn[jj, kk] += 1
        end
    end
    pij = nn + dhdp.aa
    pij = pij ./ sum(pij, 1)

    pos_components = Array(typeof(posterior(dhdp.component)), KK)
    for kk = 1:KK
        pos_components[kk] = posterior(components[kk])
    end

    pos_components, nn, pij

end