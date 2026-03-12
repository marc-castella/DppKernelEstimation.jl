# src/DppUtils.jl

"""
    DppUtils

Useful functions for DPP maximum likelihood estimation

# Exported names

   `onespluseye, generatedppsamples, complementdpp, makepsdmatrix,
   dppdist, zeroprobempty, negloglik, negloglikconvex, negloglikgrad, penalcrit,
   DppSamples, DppProb`

"""
module DppUtils
export onespluseye, generatedppsamples, complementdpp,
    makepsdmatrix, dppdist, zeroprobempty,
    negloglik, negloglikconvex, negloglikgrad, penalcrit,
    DppSamples, DppProb, cardmin, cardmax

using LinearAlgebra
using DeterminantalPointProcesses
using Distributions
using Shuffle
using Combinatorics: powerset


"""
    DppSamples

Structure for samples of subsets in {1,...,N}.

Fields: samples, N, L

Overloaded funtions: length, iterate, getindex, copy: 
Object DppSamples can be manipulated as a vector of vector (operating on field samples).
"""
struct DppSamples
    samples :: Vector{}
    N :: Int64
    L 
    function DppSamples(samples::Vector{},
                        N::Int, L)
        new(samples, N, L)
    end
    function DppSamples(samples::Vector{},
                        N::Int)
        new(samples, N, nothing)
    end
    function DppSamples(N::Int)
        new([], N, nothing)
    end
end
Base.length(data::DppSamples) = length(data.samples)
Base.iterate(data::DppSamples) = iterate(data.samples)
Base.iterate(data::DppSamples, n) = iterate(data.samples, n)
Base.getindex(data::DppSamples, n) = getindex(data.samples, n)
Base.copy(data::DppSamples) = copy(data.samples)
function Base.show(io::IO, data::DppSamples)
    println("DppSamples: ")
    println("   N       = $(data.N) (groundset size)")
    println("   samples = $(length(data)) (vector size)")
    if data.L == nothing
        println("   L       : not defined")
    else
        println("   L       : defined")
    end
end


"""
    DppProb

Structure for (discrete) probability law on subsets in {1,...,N}.

Fields: prob, N, L 
"""
struct DppProb
    prob:: Dict{}
    N :: Int64
    L
end
"""
    DppProb(L::AbstractMatrix)

Discrete probability corresponding to L-ensemble (DPP) given by matrix L
"""
function DppProb(L::AbstractMatrix)
    N = size(L)[1]
    denom = det(L+I)
    setequalprob = Dict((subs, det(L[subs, subs])/denom) for subs in powerset(1:N))
    return DppProb(setequalprob, N, L)
end
"""
    DppProb(data::DppSamples; all_subsets=false)

Empirical probability corresponding to samples in data.

By default (all_subsets=false), does not include in subsets with zero 
probability. Set all_subsets=true to include probability of all subsets.
"""
function DppProb(data::DppSamples; all_subsets=false)
    n = length(data)
    setequalprob = Dict{Any, Float64}() # dict with probability of each subset
    if all_subsets == true
        for subs in powerset(1:data.N)
            setequalprob[subs] = sum(map(x->x==subs, data))/n
        end
    elseif all_subsets == false
        tmp = copy(data.samples) # indispensable
        while length(tmp) > 0
            subs = tmp[1]
            setequalprob[subs] = sum(map(x->x==subs, data))/n
            filter!(x->x!=subs, tmp)
        end
    end
    return DppProb(setequalprob, data.N, data.L)
end
"""
    DppProb(prob::Dict{}, N::Int64)

Discrete probability over groundset {1,...N} as given by prob.
"""
DppProb(prob::Dict{}, N::Int64) = DppProb(prob, N, nothing)
Base.keys(x::DppProb) = keys(x.prob)
Base.values(x::DppProb) = values(x.prob)
Base.iterate(x::DppProb) = iterate(x.prob)
# Base.getindex(x::DppProb, n) = getindex(x.prob, n)
Base.length(x::DppProb) = length(x.prob)
function Base.show(io::IO, data::DppProb)
    println("DppProb: ")
    println("   N       = $(data.N) (groundset size)")
    println("   prob    = $(length(data)) (vector size)")
    if data.L == nothing
        println("   L       : not defined")
    else
        println("   L       : defined")
    end
end

"""
    cardmin(x)

Minimum cardinality in DppSamples or DppProb
"""
cardmin(x::DppSamples) = minimum(length, x)
cardmin(x::DppProb) = minimum(length, [i for i in keys(x) if x.prob[i]>0])
"""
    cardmax(x)

Maximum cardinality in DppSamples or DppProb
"""
cardmax(x::DppSamples) = maximum(length, x)
cardmax(x::DppProb) = maximum(length, [i for i in keys(x) if x.prob[i]>0])


"""
    zeroprobempty(p, cardmin=1)

Set to zero probability of sets with cardinal <cardmin. Probability values are
renormalized so that their sum equals one.
"""
function zeroprobempty(p::Dict, cardmin=1)
    pmodif = copy(p)
    for (key, value) in pmodif
        length(key) < cardmin ? pmodif[key] = 0 : nothing
    end
    renorm = sum(values(pmodif))
    for (key, value) in pmodif
        pmodif[key] = value/renorm
    end
    return pmodif
end
zeroprobempty(dpp::DppProb, q=1) = DppProb(zeroprobempty(dpp.prob, q), dpp.N)


"""
    dppdist(dpp1, dpp2, type=:l2)

Evaluate distance between DPP probability laws.

dpp1 and dpp2 can be DppProb, DppSamples or AbstractMatrix
type can be :l1, :l2, :tv, :kl, :hellinger or, for AbstractMatrix only :l2mat
"""
function dppdist(dpp1::DppProb, dpp2::DppProb, type=:l2)
    @assert(dpp1.N==dpp2.N, "dppdist : dpp1.N not equal dpp2.N")
    for subs in powerset(1:dpp1.N)
        subs ∉ keys(dpp1) ? dpp1.prob[subs] = 0 : nothing
        subs ∉ keys(dpp2) ? dpp2.prob[subs] = 0 : nothing
    end
    vec1 = collect(values(sort(dpp1.prob)))
    vec2 = collect(values(sort(dpp2.prob)))
    # avoid small negative values (smaller than precision)
    vec1 = max.(vec1, 0)
    vec2 = max.(vec2, 0)

    if type == :l2
        return norm(vec1 - vec2)
    elseif type == :l1          # same as TV up to a factor 1/2
        return norm(vec1 - vec2, 1)
    elseif type == :tv
        return 1/2*norm(vec1 - vec2, 1)
    elseif type == :kl         
        # Kullback-Leibler
        return sum(vec1.*log.(vec1./vec2))
    elseif type == :hellinger
        return 1/sqrt(2)*norm(@. sqrt(vec1) - sqrt(vec2))
    end
end
dppdist(dpp1::DppProb, dpp2::AbstractMatrix, type=:l2) = dppdist(dpp1, DppProb(dpp2), type)
dppdist(dpp1::AbstractMatrix, dpp2::DppProb, type=:l2) = dppdist(DppProb(dpp1), dpp2, type)
function dppdist(L1::AbstractMatrix, L2::AbstractMatrix, type=:l2)
    if type == :l2mat
        N = size(L1)[1]
        res = Inf
        for d in range(0, 2^N-1)    # possible to iterate on powerset(1:N)
            s = digits(d, base=2)
            Ltmp = copy(Matrix(L2)) # copy necessary du to the loop
            for i in range(1, length(s))
                if s[i] == 1
                    Ltmp[i, :] *= -1
                    Ltmp[:, i] *= -1
                end
            end
            candidate_dist = norm(L1 - Ltmp)
            candidate_dist < res ? res = candidate_dist : nothing
        end
        return res
    else
        return dppdist(DppProb(L1), DppProb(L2), type)
    end
end


"""
    makepsdmatrix(N; seed = nothing, init = :wishart, wparam=nothing)

Generate symmetric psd matrix of size N x N.
Possible choices for init are
- `:wishart`. In this case, rank of the matrix equals `wparam`. E{matrix} = I
- `:basic`
- `:determ` (equivalent to: `onespluseye`)
"""
function makepsdmatrix(N; seed = nothing, init = :wishart, wparam=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    if N == 0
        return zeros(0,0)
    end
    if (init == :wishart || init == :Wishart || init == :WISHART)
        wparam == nothing ? wparam=N : nothing
        Linit = rand(Wishart(wparam, diagm(ones(N)))) / wparam
    elseif (init == :basic || init == :Basic || init == :BASIC)
        Vinit = rand(Uniform(0, √2), (N, N)) / N
        Linit = Vinit * Vinit'
    elseif (init == :determ || init == :Determ || init == :DETERM)
        Linit = ones(N, N) + I
    elseif (init == :ident || init == :Ident || init == :IDENT)
        Linit = Matrix{Float64}(I(N))
    else
        throw(ArgumentError("Bad keyword argument \"init\"."))
    end
    return Linit
end


"""
    onespluseye(n)

Return matrix ones(n,n) + I
"""
onespluseye(n) = ones(n,n) + I


"""
    generatedppsamples(groundset_size, nsamples; nempty=nothing, L=nothing)

Generate samples from a DPP.

# Arguments
- `groundset_size::Integer`
- `nsamples::Integer`
- (optional) `nempty:Integer`: number of empty sets (default: random)
- (optional) `L::Matrix`: kernel of L ensemble (default: randomly drawn)

# Outputs
- data::DppSamples
- testdata
"""
function generatedppsamples(groundset_size, nsamples; nempty=nothing, L=nothing)
    if L == nothing
        V = randn(groundset_size, groundset_size)
        L = V*transpose(V)
    end
    mp = DPP(L)

    testdata = rand(mp, nsamples)
    if nempty == nothing
        traindata = rand(mp, nsamples)
    else
        traindata = []
        for i in range(1, nempty)
            push!(traindata, [])
        end
        for i in range(1, 4*nsamples)
            length(traindata) == nsamples ? break :
            onesample = rand(mp)
            if length(onesample) > 0
                push!(traindata, onesample)
            end
        end
        shuffle!(traindata)
    end
    return DppSamples(traindata, groundset_size, L), testdata
end


"""
    complementdpp(data::DppSamples)

Return DppSamples with complement samples.
"""
function complementdpp(data::DppSamples)
    # prev. calculating maximum integer in sets by: maximum(reduce(vcat, data))
    complementsets = map(X -> setdiff(collect(range(1,data.N)), X),
                         data)
    return DppSamples(complementsets, data.N)
end
"""
    complementdpp(dpp:DppProb)

Return DppProb of complement DPP.
"""
function complementdpp(dpp::DppProb)
    complprob = Dict((setdiff(collect(range(1,dpp.N)), subs),
                      dpp.prob[subs]) for subs in keys(dpp.prob))
    return DppProb(complprob, dpp.N)
end


"""
nucnorm(m)

Computes the nuclear norm of a matrix `m`.
"""
function nucnorm(m::AbstractMatrix)
    norm(svdvals(m),1)
end


"""
    negloglikconvex(L, data)

Evaluate convex part of neg-log-likelihood criterion: 
    - 1/n*∑ᵢ logdet(L_Yᵢ)
"""
function negloglikconvex(L::Hermitian{Float64, Matrix{Float64}}, data)
    isposdef(L) || return Inf
    sum = 0
    for sample in data
        L_y = L[sample, sample]
        sum += logdet(L_y)
    end
    n = length(data)
    return -(1/n)*sum
end
"""
    negloglikconvex(L, p::Dict)

Evaluate convex part of neg-log-likelihood criterion:
    - ∑_{Y∈ keys(p)} p[Y] logdet(L_Y)
"""
function negloglikconvex(L::Hermitian{Float64, Matrix{Float64}}, p::Dict{})
    isposdef(L) || return Inf
    sum = 0
    for (subset, prob) in p
        L_y = L[subset, subset]
        sum += -prob*logdet(L_y)
    end
    return sum
end
negloglikconvex(L::Hermitian{Float64, Matrix{Float64}}, p::DppProb) = negloglikconvex(L, p.prob)
"""
   negloglik(L, data, λ=0)

Evaluate neg-log-likelihood criterion + logdet(.+I) penalization
logdet(L+I) - 1/n*∑ᵢ logdet(L_Yᵢ) + λ*logdet(L+I)
logdet(L+I) - ∑_{Y∈ keys(p)} p[Y] logdet(L_Y) + λ*logdet(L+I)

"""
function negloglik(L, data, λ=0)
    return (1+λ)*logdet(L+I) + negloglikconvex(L, data)
end


"""
    negloglikgrad(L, data; λ=0)

Compute gradient of negloglik function.
(L+I)^{-1} - 1/n∑ᵢ Uᵢ (Uᵢᵀ L Uᵢ)^{-1}Uᵢᵀ
"""
function negloglikgrad(L, data; λ=0)
    n = length(data)
    res = zeros(size(L)) +(1+λ)*inv(L+I)
    for samp in data
        res[samp, samp] -= (1/n)inv(L[samp, samp])
    end
    # floating point: res not symmetric, avoid error accumulation
    return hermitianpart(res)  
end
"""
    negloglikgrad(L, p::Dict{}; λ=0)
"""
function negloglikgrad(L, p::Dict{}; λ=0)
    res = zeros(size(L)) +(1+λ)*inv(L+I)
    for (subset, prob) in p
        res[subset, subset] -= prob*inv(L[subset, subset])
    end
    # floating point: res not symmetric, avoid error accumulation
    return hermitianpart(res)  
    
end
negloglikgrad(L, x::DppProb; λ=0) = negloglikgrad(L, x.prob; λ=λ)


"""
    penalcrit(L, terms...)

Evaluate penalization criterion at L. Depending on vararg `terms`, it is given 
by the sum of following elements:
- (:nuc, λ, L₀) : λ*||L-L₀||_{nuc} (nuclear norm)
- (:fro, λ, L₀) : λ*||L-L₀||_F² (squared Frobenius norm)
- (:l1,  λ, L₀) : λ*||L-L₀||₁ (l1 norm sum(abs.( )) )
- (:ldi, λ, L₀) : λ*logdet(L+I)
"""
function penalcrit(L, terms...)
    pen = 0
    for penargs in terms
        length(penargs) == 3 ? LminusL0 = L - penargs[3] : LminusL0 = L
        lambda = penargs[2]
        if penargs[1] == :nuc
            pen += lambda*nucnorm(LminusL0)
        elseif penargs[1] == :fro
            pen += lambda*norm(LminusL0)^2
        elseif penargs[1] == :l1
            pen += lambda*norm(LminusL0, 1)
        elseif penargs[1] == :ldi
            pen += lambda*logdet(LminusL0+I)
        end
    end
    return pen
end
end                             # module DppUtils
