# experiments/figure1.jl

using DppKernelEstimation
import PythonPlot as plt
using LinearAlgebra
using Printf
using Format
using Combinatorics: powerset
import Random


parent_dir = dirname(@__DIR__)
const FIGURES_DIR = joinpath(parent_dir, "figures")
if !isdir(FIGURES_DIR)
    mkdir(FIGURES_DIR)
end


"""
    generate_figure1(; savename="figure1")

Figure pour le cas N=2, illustre non coercivite.
"""
function generate_figure1(; savename="figure1")
    disttype="tv"               # use TV distance
    Random.seed!(27062025)
    # Génération de données/DPP
    # =========================
    N = 2
    
    tmp = rand(2^N-1)
    tmp = tmp/sum(tmp)
    
    tmp2 = Dict()
    for (i, j) in enumerate(powerset(1:N))
        if !isempty(j)
            tmp2[j] = pop!(tmp)
        end
    end
    learndata = DppProb(tmp2, N)
    
    # Algo fixed pt
    # =============
    params = (maxiter=500, step=1, prec=0) # fp alg parameters
    
    μ1 = 0
    L1, crit1, (test1, norml1, negloglik1) = mle_fixpt(learndata,
                                                       μ1,
                                                       params=params,
                                                       testdata=learndata)
    
    μ2 = 0.01
    L2, crit2, (test2, norml2, negloglik2) = mle_fixpt(learndata,
                                                       μ2,
                                                       params=params,
                                                       testdata=learndata)
    
    μ3 = 0.05
    L3, crit3, (test3, norml3, negloglik3) = mle_fixpt(learndata,
                                                       μ3,
                                                       params=params,
                                                       testdata=learndata)

    # Affichage fixed pt
    # ==================
    fig_fixpt, ax_fixpt = plt.subplots(3, 1, clear=true)
    ax_fixpt[0].plot(1:params.maxiter, crit1, "b-", label="μ="*string(μ1))
    ax_fixpt[1].plot(1:params.maxiter, norml1, "r-")
    ax_fixpt[2].plot(1:params.maxiter, test1[disttype], "g-")
    
    ax_fixpt[0].plot(1:params.maxiter, crit2, "b--", label="μ="*string(μ2))
    ax_fixpt[1].plot(1:params.maxiter, norml2, "r--")
    ax_fixpt[2].plot(1:params.maxiter, test2[disttype], "g--")
    
    ax_fixpt[0].plot(1:params.maxiter, crit3, "b-.", label="μ="*string(μ3))
    ax_fixpt[1].plot(1:params.maxiter, norml3, "r-.")
    ax_fixpt[2].plot(1:params.maxiter, test3[disttype], "g-.")

    disttype == "l2" ? ax_fixpt[2].set_yscale("log") : nothing
    disttype == "tv" ? ax_fixpt[2].set_yscale("log") : nothing
    for i in 0:2
        ax_fixpt[i].set_xlim([0, params.maxiter])
    end
    ax_fixpt[0].legend(fontsize=13)
    ax_fixpt[0].set_ylabel("Criterion", fontsize=14)
    ax_fixpt[1].set_ylabel("Norm(L)", fontsize=14)
    ax_fixpt[2].set_ylabel("Dist. true DPP", fontsize=14)
    ax_fixpt[2].set_xlabel("Algorithm iterations", fontsize=14)
    ax_fixpt[0].set_title("Fixed point algorithm \n with different "*
        "regularization parameter", fontsize=14)


    # Algo fixed low-rank
    # ===================
    params = (maxiter=1000, step=0.1, prec=0) # lr alg parameter
    Vinit = cholesky(onespluseye(N))
    Vinit = Vinit.L[:,1:N]

    μ1 = 0
    L1, crit1, (test1, norml1, negloglik1, V1) = mle_lowrk(learndata,
                                                           μ1,
                                                           init=Vinit,
                                                           params=params,
                                                           testdata=learndata)
    μ2 = 0.01
    L2, crit2, (test2, norml2, negloglik2, V2) = mle_lowrk(learndata,
                                                           μ2,
                                                           init=Vinit,
                                                           params=params,
                                                           testdata=learndata)
    μ3 = 0.05
    L3, crit3, (test3, norml3, negloglik3, V3) = mle_lowrk(learndata,
                                                           μ3,
                                                           init=Vinit,
                                                           params=params,
                                                           testdata=learndata)
    
    # Affichage low-rank
    # ==================
    fig_lowrk, ax_lowrk = plt.subplots(3, 1, clear=true)
    ax_lowrk[0].plot(1:params.maxiter, crit1, "b-", label="μ="*string(μ1))
    ax_lowrk[1].plot(1:params.maxiter, norml1, "r-")
    ax_lowrk[2].plot(1:params.maxiter, test1[disttype], "g-")
    
    ax_lowrk[0].plot(1:params.maxiter, crit2, "b--", label="μ="*string(μ2))
    ax_lowrk[1].plot(1:params.maxiter, norml2, "r--")
    ax_lowrk[2].plot(1:params.maxiter, test2[disttype], "g--")
    
    ax_lowrk[0].plot(1:params.maxiter, crit3, "b-.", label="μ="*string(μ3))
    ax_lowrk[1].plot(1:params.maxiter, norml3, "r-.")
    ax_lowrk[2].plot(1:params.maxiter, test3[disttype], "g-.")

    disttype == "l2" ? ax_lowrk[2].set_yscale("log") : nothing
    disttype == "tv" ? ax_lowrk[2].set_yscale("log") : nothing
    for i in 0:2
        ax_lowrk[i].set_xlim([0, params.maxiter])
    end
    ax_lowrk[0].legend(fontsize=13)
    ax_lowrk[0].set_ylabel("Criterion", fontsize=14)
    ax_lowrk[1].set_ylabel("Norm(L)", fontsize=14)
    ax_lowrk[2].set_ylabel("Dist. true DPP", fontsize=14)
    ax_lowrk[2].set_xlabel("Algorithm iterations", fontsize=14)
    ax_lowrk[0].set_title("Low-rank algorithm \n with different "*
        "regularization parameter", fontsize=14)

    name1 = joinpath(FIGURES_DIR, savename*"_left_fixpt.pdf")
    name2 = joinpath(FIGURES_DIR, savename*"_right_lowrk.pdf")
    fig_fixpt.savefig(name1)
    println("Saved "*name1)
    fig_lowrk.savefig(name2)
    println("Saved "*name2)
end

println("--- Function generate_figure1() now defined. ---")
println("-"^70)
