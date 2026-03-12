# src/DppMleAlgs.jl

"""
    DppMleAlgs

DPP maximum likelihood estimation: fixed point and low-rank gradient algorithms

# Exported names

   `mle_fixpt, mle_lowrk`

"""
module DppMleAlgs
export mle_fixpt, mle_lowrk

using LinearAlgebra
using DataStructures: CircularBuffer
using ..DppUtils


### functions for fixed point algorithm [Mariet15] ###
"""
    Δ(L, data; λ=0)

Compute Δ matrix for fixed-point optimization algorithm in paper [Mariet15].
Δ(L, data; λ=0) = -negloglikgrad(L, data; λ=λ)

Optimization step for minimization of negloglikelihood is 
L ← L+LΔL (fixed-point algorithm `mle_fixpt`, [Mariet15]).

With λ=0, result identical to .py function.

When λ>0, computes similarly for minimization of penalized criterion
(1+λ)logdet(L+I) - 1/n∑ᵢ logdet(Uᵢᵀ L Uᵢ)

# References
[Mariet15] Mariet, Z., & Sra, S., Fixed-point algorithms for learning
    determinantal point processes, CoRR, (), (2015).
"""
Δ(L, data; λ=0) = -negloglikgrad(L, data; λ=λ)


"""
    mle_fixpt(data, λ=0; params, init=:determ, testdata=nothing)

# Arguments
- data: DppSamples or Dict with discrete probabilities
- λ: 
- params: named tuple (params.maxiter, params.step, params.prec)
- init:
- testdata:

# Outputs
- L
- crit_trace
- (test_trace, norml_trace, negloglik_trace)
"""
function mle_fixpt(data, λ=0; params, init=:determ, testdata=nothing)
    if init == :determ
        N = nothing             # N should be created outside try/catch/end
        try
            N = data.N          # ok if data isa DppSamples
        catch
            N = maximum(length, keys(data)) # if data isa Dict{}
        end
        L = hermitianpart(onespluseye(N))
    else
        L = hermitianpart(init)
    end
    negloglik_trace = [negloglik(L, data)]
    crit_trace = [negloglik(L, data, λ)]
    if testdata != nothing
        test_trace = Dict("crit" => [negloglik(L, testdata, λ)],
                          "negloglik" => [negloglik(L, testdata)],
                          "l2" => [dppdist(L, testdata, :l2)],
                          "tv" => [dppdist(L, testdata, :tv)],
                          "he" => [dppdist(L, testdata, :hellinger)])
    elseif testdata == nothing
        test_trace = Dict()
    end
    norml_trace = [norm(L)]
    # L_trace = [L]
    for iter in range(2, params.maxiter)
        L = hermitianpart(L + params.step*L*Δ(L, data; λ=λ)*L)
        push!(negloglik_trace, negloglik(L, data))
        push!(crit_trace, negloglik(L, data, λ))
        if testdata != nothing
            push!(test_trace["crit"], negloglik(L, testdata, λ))
            push!(test_trace["negloglik"], negloglik(L, testdata))
            push!(test_trace["l2"], dppdist(L, testdata, :l2))
            push!(test_trace["tv"], dppdist(L, testdata, :tv))
            push!(test_trace["he"], dppdist(L, testdata, :hellinger))
        end
        push!(norml_trace, norm(L))
    end
    return (L, crit_trace, (test_trace, norml_trace, negloglik_trace))
end


### functions for low-rank optimization ###
"""
    logdetvv_grad(V)

Compute gradient of V -> logdet( VV^T)

Similar to .py function
"""
function logdetvv_grad(V)
    L = V * transpose(V)
    A = inv(L)
    grad = 2 * A * V
    return grad
    
end


"""
    logdetvvi_grad(V)

Compute gradient of V -> logdet( VV^T+Id)
Similar to .py function
"""
function logdetvvi_grad(V)
    m, k = size(V)
    # Below is (I+VV^T)^{-1} (matrix inversion lemma):
    B = I(m) - V * inv( I(k) + transpose(V)*V ) * transpose(V)
    grad = 2 * B * V
    return grad
end


"""
    lowrank_map(V, data, lr; λ=0)

Apply one (full) gradient step to V.

No longer similar to .py function.
But lowrank_map(V, data, lr*length(data)) should yield same result as
lowrank_map(V, data, lr) in .py function.
"""
function lowrank_map(V, data, lr; λ=0)
    n = length(data)
    V_new = copy(V)
    for sample in data
        V_n = V[sample, :]
        grad_n = logdetvv_grad(V_n)
        V_new[sample, :] += lr/n*grad_n
    end
    V_new -= (1+λ)*lr*logdetvvi_grad(V)
    return V_new
end
function lowrank_map(V, p::Dict{}, lr; λ=0)
    V_new = copy(V)
    for (subset, prob) in p
        V_n = V[subset, :]
        grad_n = logdetvv_grad(V_n)
        V_new[subset, :] += lr*prob*grad_n
    end
    V_new -= (1+λ)*lr*logdetvvi_grad(V)
    return V_new
end
lowrank_map(V, x::DppProb, lr; λ=0) = lowrank_map(V, x.prob, lr;λ=λ)



"""
    makeposdef(L, tol)

Rendre def. pos. une matrice qui ne l'est pas. Renvoie L, avec toutes vp>= tol.
Warning si une vp<tol.
"""
function makeposdef(L, tol)
    vals, vecs = eigen(L)
    any(vals.<-tol) ? println("WARNING: vp<tol... remise>tol") : nothing
    L  = Hermitian(vecs*diagm(max.(tol, vals))*transpose(vecs))
    return L
end


"""
    mle_lowrk(data, λ=0; params, init, testdata=nothing)

# Arguments
- data
- init
- params
- (optional) λ
- (optional) testdata

# Outputs
- L
- crit_trace
- other = (test_trace, norml_trace, negloglik_trace, V)
"""
function mle_lowrk(data, λ=0; params, init, testdata=nothing)
    V = init
    L = hermitianpart(V*transpose(V))
    norml_trace = [norm(L)]
    negloglik_trace = [negloglik(L, data)]
    crit_trace = [negloglik(L, data, λ)]
    if testdata != nothing
        # if maximum(map(length, testdata)) > size(init)[2]
        #     println("Some sets in testdata have cardinality > #cols in init")
        #     println("                                          (i.e. rank L)")
        #     println("Test likelihoods not evaluated.")
        #     println("---------------------------------------------------------")
        #     testdata = nothing
        # end
        test_trace = Dict("crit" => [negloglik(L, testdata, λ)],
                          "negloglik" => [negloglik(L, testdata)],
                          "l2" => [dppdist(L, testdata, :l2)],
                          "tv" => [dppdist(L, testdata, :tv)],
                          "he" => [dppdist(L, testdata, :hellinger)])
    elseif testdata == nothing
        test_trace = Dict()
    end
    for iter in range(2, params.maxiter)
        V = lowrank_map(V, data, params.step, λ=λ)
        #
        L = hermitianpart(makeposdef(V*transpose(V), 1e-10))
        #
        push!(negloglik_trace, negloglik(L, data))
        push!(crit_trace, negloglik(L, data, λ))
        push!(norml_trace, norm(L))
        if testdata != nothing
            push!(test_trace["crit"], negloglik(L, testdata, λ))
            push!(test_trace["negloglik"], negloglik(L, testdata))
            push!(test_trace["l2"], dppdist(L, testdata, :l2))
            push!(test_trace["tv"], dppdist(L, testdata, :tv))
            push!(test_trace["he"], dppdist(L, testdata, :hellinger))
        end
    end
    other = (test_trace, norml_trace, negloglik_trace, V)
    return (L, crit_trace, other)
end

end                             # module DppMleAlgs
