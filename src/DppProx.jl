# src/DppProx.jl

module DppProx
export fwdbwd_all

using LinearAlgebra
using DataStructures: CircularBuffer
using ..DppUtils
using Parameters: @unpack

### functions for proximal algorithms ###
"""
    prox_logdet(L, γ)

Proximity operator of γ*(-logdet()) at L
"""
function prox_logdet(L, γ)
    vals, P = eigen(L)
    hermitianpart(P*diagm(vals + sqrt.(vals.^2 .+ 4*γ))*transpose(P)/2)
end


"""
    proj_Sε(L, mineigval)

Projection of L onto Sε={M: M>mineigval*I}
"""
function proj_Sε(L, mineigval)
    vals, P = eigen(L)
    hermitianpart(P*diagm(max.(vals, mineigval))*transpose(P))
end


"""
    soft_threshold(beta, X)
"""
function soft_threshold(beta, X)
    sign.(X) .* max.(abs.(X) .- beta, 0)
end
"""
     prox_l1(X, beta) = soft_threshold(beta, X)
"""
prox_l1(X, beta) = soft_threshold(beta, X)


"""
    crit_prox_dbcfb(X, L, γ, data)

Identical to:  neglogliconvex(hermitianpart(X), data) 
               + penalcrit(hermitianpart(X), (:fro, 1/(2*γ), L))

"""
function crit_prox_dbcfb(X, L, γ, data)
    Xloc = hermitianpart(X)
    out = negloglikconvex(Xloc, data)
    out += penalcrit(Xloc, (:fro, 1/(2*γ), L))
    return out
end


"""
    prox_nuc_Sε(L, γ; mineigval=0)

Evaluate prox of indic(L>=mineigval*I) + γ||L||_nuc
"""
function prox_nuc_Sε(L, γ; mineigval=0)
    vals, P = eigen(L)
    return hermitianpart(P*diagm(max.(vals.-γ, mineigval))*transpose(P))
end    


"""
    dbfb1_all(γ_F, L, data, ε=1e-4, λₙ=0, λ₁=0; params, init=:determ)

Compute:  Argmin_X -1/n ∑ᵢ logdet X[data[i], data[i]] 
                      + ı_Sε(X) + λₙ||X||_{nuc}
                      + λ₁||X||₁
                      + 1/(2*γ_F)||X-L||^2
Algorithm (34) in reference [1]

# Outputs
- X
- crit_trace
- (Y,)

# Reference 

- [1] Abboud, F., Chouzenoux, E., Pesquet, J., Chenot, J., & Laborelli, L., Dual
block-coordinate forward-backward algorithm with application to deconvolution
and deinterlacing of video sequences, "J. Math. Imaging Vision", 59(3), 415–431
(2017).
"""
function dbfb1_all(γ_F, L, data, ε=1e-4, λₙ=0, λ₁=0; params, init=:determ)
    N = size(L)[1]
    n = length(data)
    if data isa DppSamples || data isa Vector
        coeflogdet = fill(1/n, n)
        locdat = data
    elseif data isa Dict{} || data isa DppProb
        coeflogdet = collect(values(data))
        locdat = collect(keys(data))
    end
    m = n + 1
    if init == :determ
        Y = [onespluseye(length(sample)) for sample in locdat]
        push!(Y, onespluseye(N))
    elseif init isa Vector{Matrix{Float64}}
        Y = init
    end
    c = params.step
    critcur = Inf
    rel_decr_deque = CircularBuffer{Float64}(m); fill!(rel_decr_deque, Inf)
    Z = - Y[m]
    for (i, Yi) in zip(locdat, Y[1:n])
        Z[i, i] -= Yi
    end
    X = nothing                 # creates local variable outside for loop
    crit_trace = []
    for k in range(0, params.maxiter-1) # alternatively iterate on zip(locdat, Y)?
        j = (k % m) + 1
        X = prox_nuc_Sε(L+Z, γ_F*λₙ, mineigval=ε)
        if j<= n
            Y_prev = Y[j]
            Ytilde = Y_prev + c*X[locdat[j], locdat[j]]
            Y[j] = Ytilde - c*prox_logdet(Ytilde/c, coeflogdet[j]*γ_F/c)
            Z[locdat[j], locdat[j]] -= Y[j]-Y_prev
        elseif j == n+1         # add a test λ₁ == 0 ?
            Y_prev = Y[j]
            Ytilde = Y_prev + c*X
            Y[j] = Ytilde - c*prox_l1(Ytilde/c, γ_F*λ₁/c)
            Z -= Y[j]-Y_prev
        end
        # Rough test for exit : average relative decrease over (n+1) iter.
        critprec = critcur
        critcur = negloglikconvex(X, data) + penalcrit(X, (:nuc, λₙ), (:l1, λ₁), (:fro, 1/(2*γ_F), L))
        push!(crit_trace, critcur)
        push!(rel_decr_deque, abs((critprec - critcur)/critcur))
        sum(rel_decr_deque)/length(rel_decr_deque) < params.prec && break
    end
    return X, crit_trace, (Y, )
end


"""
    fwdbwd_all(data, regul = (ε=1e-4, λ=0, λ₁=0, λₙ=0);
                    outparams, inparams, init=:determ, testdata=nothing)

# Outputs
- L
- crit_trace
- (test_trace, norml_trace, negloglik_trace)
"""
function fwdbwd_all(data, regul = (ε=1e-4, λ=0, λ₁=0, λₙ=0);
                    outparams, inparams, init=:determ, testdata=nothing)
    @unpack ε, λ, λ₁, λₙ = regul
    c = inparams.step           # used to define parameters in 1st iteration
    γ = outparams.step
    if γ > 2/(1+λ)
        γ = 1.99/(1+λ)
        println("WARNING: forward-backward step too large, set to $γ")
    end
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
    crit_trace = [negloglik_trace[end] + penalcrit(L, (:ldi, λ), (:nuc, λₙ), (:l1, λ₁))]
    norml_trace = [norm(L)]
    if testdata != nothing
        test_trace = Dict("crit" => [negloglik(L, testdata, λ) + penalcrit(L, (:ldi, λ), (:nuc, λₙ), (:l1, λ₁))],
                          "negloglik" => [negloglik(L, testdata)],
                          "l2" => [dppdist(L, testdata, :l2)],
                          "tv" => [dppdist(L, testdata, :tv)],
                          "he" => [dppdist(L, testdata, :hellinger)])
    elseif testdata == nothing
        test_trace = Dict()
    end
    
    # first iteration of alg. outside loop to set inparams.maxiter large enough
    # (smaller value is ok later due to warm start)
    L_interm = L - γ*(1+λ)*inv(L+I)
    prox = dbfb1_all(γ, L_interm, data, ε, λₙ, λ₁, params=(maxiter=2000, step=c, prec=1e-4))
    L, (Y₀,) = hermitianpart(prox[1]), prox[3]
    push!(norml_trace, norm(L))
    push!(negloglik_trace, negloglik(L, data))
    push!(crit_trace, negloglik_trace[end] + λ*logdet(L+I) + penalcrit(L, (:ldi, λ), (:nuc, λₙ), (:l1, λ₁)))
    if testdata != nothing
        push!(test_trace["crit"], negloglik(L, testdata, λ)  + penalcrit(L, (:ldi, λ), (:nuc, λₙ), (:l1, λ₁)) )
        push!(test_trace["negloglik"], negloglik(L, testdata))
        push!(test_trace["l2"], dppdist(L, testdata, :l2))
        push!(test_trace["tv"], dppdist(L, testdata, :tv))
        push!(test_trace["he"], dppdist(L, testdata, :hellinger))
    end
    for k in range(2, outparams.maxiter-1)
        L_interm = L - γ*(1+λ)*inv(L+I)
        prox = dbfb1_all(γ, L_interm, data, ε, λₙ, λ₁, params=inparams, init=Y₀)
        L, (Y₀,) = hermitianpart(prox[1]), prox[3]
        push!(norml_trace, norm(L))
        push!(negloglik_trace, negloglik(L, data))
        push!(crit_trace, negloglik_trace[end]  + penalcrit(L, (:ldi, λ), (:nuc, λₙ), (:l1, λ₁)))
        if testdata != nothing
            push!(test_trace["crit"], negloglik(L, testdata, λ) + penalcrit(L, (:ldi, λ), (:nuc, λₙ), (:l1, λ₁)))
            push!(test_trace["negloglik"], negloglik(L, testdata))
            push!(test_trace["l2"], dppdist(L, testdata, :l2))
            push!(test_trace["tv"], dppdist(L, testdata, :tv))
            push!(test_trace["he"], dppdist(L, testdata, :hellinger))
        end
    end
    return (L, crit_trace, (test_trace, norml_trace, negloglik_trace))
end

end                             # module DppProx
