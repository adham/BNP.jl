#=
common.jl

Adham Beyki, odinay@gmail.com
=#


abstract MixtureModel
abstract hMixtureModel


immutable Sent
    w::Vector{Int}
    c::Vector{Int}
    Sent(w::Vector{Int}, c::Vector{Int}) = new(w, c)
end
Sent(n::Int) = Sent(zeros(Int, n), zeros(Int, n))
Base.length(x::Sent) = length(x.w)



"""
normalizes a vector with logarithmic scale values
"""
function lognormalize!(pp::Vector{Float64})
    pp_len = length(pp)
    pp_max = maximum(pp)
    @inbounds for kk = 1:pp_len
        pp[kk] = exp(pp[kk] - pp_max)
    end
    pp_sum = sum(pp)
    @inbounds for kk = 1:pp_len
        pp[kk] /= pp_sum
    end
end



"""
draws samples from a probability vector
"""
function sample(w::Vector{Float64})
    r = rand()
    n = length(w)
    i = 1
    cw = w[1]
    while cw < r && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end
function sample(w::Vector{Float64}, n::Int)
    ret = zeros(Int, n)
    @inbounds for i = 1:n
        ret[i] = sample(w)
    end
    return ret
end

rand_Dirichlet(alpha::Vector{Float64}) = Distributions.rand(Distributions.Dirichlet(alpha))
rand_Dirichlet(alpha::Vector{Float64}, n) = Distributions.rand(Distributions.Dirichlet(alpha), n)


"""
deletes column kk from matrix nn and returns nn
"""
function del_column{T}(nn::Matrix{T}, kk::Int)
  KK = size(nn, 2)
  mask = 1:KK .!= kk
  return nn[:, mask]
end

"""
adds a column to the rightmost of nn and returns nn
"""
function add_column{T}(nn::Matrix{T})
  r, c = size(nn)
  mm = zeros(T, r, c+1)
  mm[:, 1:c] = nn
  return mm
end


"""
generates bar topics
"""
function gen_bars(n_bars, n_vocab, noise_level)
    KK = round(Int, sqrt(n_vocab))
    bars = zeros(Float64, n_bars, n_vocab)

    for kk in 1:round(Int, n_bars/2)
        b = zeros(KK, KK) + noise_level
        b[kk, :] = ones(KK)
        b /= sum(b)
        bars[kk, :] = b[:]
    end

    for kk in round(Int, n_bars/2)+1 : n_bars
        b = zeros(KK, KK) + noise_level
        b[:, kk-round(Int, n_bars/2)] = ones(KK)
        b /= sum(b)
        bars[kk, :] = b[:]
    end
    bars
end



"""
vertical cumulative sum in reverse order from bottom to top
"""
function reverse_cumsum{T}(nn::Matrix{T})

  J = size(nn, 1)
  mm = zeros(T, J, J)
  mm[J, :] = nn[J, :]
  for jj = 2:J
    mm[J-jj+1, :] = nn[J-jj+1, :] + mm[J-jj+2, :]
  end
  mm
end



"""
writes a topic into a CSV file
"""
function topic2csv(filename, vocab, alpha)

  idx_sorted = sortperm(alpha, rev=true)

  words = Array(Tuple{ASCIIString, Float64}, length(vocab))
  for i = 1:length(idx_sorted)
    words[i] = (vocab[idx_sorted[i]], alpha[idx_sorted[i]])
  end

  csvfile = open(filename, "w")

  for i = 1:length(vocab)
    write(csvfile, join(words[i], ","), "\n")
  end

  close(csvfile)
end


"""
writes top n topics used in a document
"""
function write_top_doctopics(pij, filename, topn=5)
  csvfile = open(filename, "w")

  for kk = 1:size(pij, 2)
    idx = sortperm(pij[:, kk], rev=true)[1:topn]
    insert!(idx, 1, kk)
    write(csvfile, join(idx, ","), "\n")
  end

  close(csvfile)
end


# macro for argument checking
macro check_args(D, cond)
    quote
        if !($cond)
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)), " is not satisfied.")))
        end
    end
end

function stick_breaking(vv::Vector{Float64})
    KK = length(vv)
    pp = ones(Float64, KK)

    pp[2:KK] = 1-vv[1:KK-1]
    pp = vv .* cumprod(pp)
    pp
end

function log_stick_breaking(vv::Vector{Float64})
  KK = length(vv)
  pp = zeros(Float64, KK)

  pp[2:KK] = log(1 - vv[1:KK-1])
  pp = log(vv) + cumsum(pp)
  pp
end

"""
vertical cumulative sum in reverse order
from bottom to top
"""
function reverse_cumsum{T}(nn::Matrix{T})

  J = size(nn, 1)
  mm = zeros(T, J, J)
  mm[J, :] = nn[J, :]
  for jj = 2:J
    mm[J-jj+1, :] = nn[J-jj+1, :] + mm[J-jj+2, :]
  end
  mm
end
