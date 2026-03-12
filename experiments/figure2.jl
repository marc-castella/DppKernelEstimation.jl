#experiments/figure2.jl

using DppKernelEstimation
using PythonPlot
using LinearAlgebra
using Printf
using Random
using LaTeXStrings


parent_dir = dirname(@__DIR__)
const FIGURES_DIR = joinpath(parent_dir, "figures")
if !isdir(FIGURES_DIR)
    mkdir(FIGURES_DIR)
end


function generate_figure2(; savename="figure2")
    Random.seed!(0)
    
    N = 3
    Ltrue = makepsdmatrix(N)
    
    origdata = DppProb(Ltrue)
    pempty = origdata.prob[[]]
    mutheo = pempty/(1-pempty)
    
    learndata1 = zeroprobempty(origdata)
    n = 200
    learndata2, _ = generatedppsamples(N, n, L=Ltrue, nempty=0)
    n = 100
    learndata3, _ = generatedppsamples(N, n, L=Ltrue, nempty=0)
    
    fpparams = (maxiter=2000, step=1, prec=0) # fp alg parameters
    
    alllambdas = LinRange(0, 0.35, 21)
    distances_Ltrue_1 = []
    distances_Ltrue_2 = []
    distances_Ltrue_3 = []
    for λ in alllambdas
        Ltmp1, crittmp1, (_, norml1, negloglik1) = mle_fixpt(learndata1, λ, params=fpparams)
        append!(distances_Ltrue_1, dppdist(Ltrue, Ltmp1, :tv))
        Ltmp1, crittmp1, (_, norml1, negloglik1) = mle_fixpt(learndata2, λ, params=fpparams)
        append!(distances_Ltrue_2, dppdist(Ltrue, Ltmp1, :tv))
        Ltmp1, crittmp1, (_, norml1, negloglik1) = mle_fixpt(learndata3, λ, params=fpparams)
        append!(distances_Ltrue_3, dppdist(Ltrue, Ltmp1, :tv))
    end
    
    
    fig, ax = pyplot.subplots(1, 2, clear=true,
                               figsize=[12.8, 4.8],
                               layout="constrained")
    
    ax[0].plot(alllambdas, distances_Ltrue_1, ls="-", label=L"$\hat{p}_{\emptyset}$ set to zero")
    ax[0].plot(alllambdas, distances_Ltrue_2, ls="--", label="n=$(length(learndata2.samples)) samples")
    ax[0].plot(alllambdas, distances_Ltrue_3, ls="-.", label="n=$(length(learndata3.samples)) samples")
    ax[0].axvline(x=mutheo, color="black", ls=":")
    ax[0].set_xlabel("μ", fontsize=14)
    ax[0].set_ylabel("Dist. true DPP", fontsize=14)
    ax[0].set_title("Ground set of cardinality N=$N", fontsize=14)
    ax[0].legend(loc="upper right", fontsize=14)
    ax[0].set_xlim(0, 0.35)
    ax[0].annotate(
        L"$\mu=p_{\emptyset}/(1-p_{\emptyset})$",
        xy=(mutheo, 0.2),                 # point ciblé (sur le trait)
        xycoords="data",
        xytext=(-50, 10),               # décalage automatique (en points)
        textcoords="offset points",
        arrowprops=Dict(
            "arrowstyle" => "->",
            "linewidth"  => 1.2
        ),
        ha="center",
        va="bottom",
        fontsize = 13
    )
    
    ################################################################################
    N = 8
    Ltrue = makepsdmatrix(N)
    
    origdata = DppProb(Ltrue)
    pempty = origdata.prob[[]]
    mutheo = pempty/(1-pempty)
    
    learndata1 = zeroprobempty(origdata)
    n = 100
    learndata3, _ = generatedppsamples(N, n, L=Ltrue, nempty=0)
    n = 200
    learndata2, _ = generatedppsamples(N, n, L=Ltrue, nempty=0)
    
    fpparams = (maxiter=2000, step=1, prec=0) # fp alg parameters
    
    alllambdas = LinRange(0, 0.35, 21)
    distances_Ltrue_1 = []
    distances_Ltrue_2 = []
    distances_Ltrue_3 = []
    for λ in alllambdas
        Ltmp1, crittmp1, (_, norml1, negloglik1) = mle_fixpt(learndata1, λ, params=fpparams)
        append!(distances_Ltrue_1, dppdist(Ltrue, Ltmp1, :tv))
        Ltmp1, crittmp1, (_, norml1, negloglik1) = mle_fixpt(learndata2, λ, params=fpparams)
        append!(distances_Ltrue_2, dppdist(Ltrue, Ltmp1, :tv))
        Ltmp1, crittmp1, (_, norml1, negloglik1) = mle_fixpt(learndata3, λ, params=fpparams)
        append!(distances_Ltrue_3, dppdist(Ltrue, Ltmp1, :tv))
    end
    ax[1].plot(alllambdas, distances_Ltrue_1, ls="-", label=L"$\hat{p}_{\emptyset}$ set to zero")
    ax[1].plot(alllambdas, distances_Ltrue_2, ls="--", label="n=$(length(learndata2.samples)) samples")
    ax[1].plot(alllambdas, distances_Ltrue_3, ls="-.", label="n=$(length(learndata3.samples)) samples")
    ax[1].axvline(x=mutheo, color="black", ls=":")
    ax[1].set_xlabel("μ", fontsize=14)
    ax[1].set_ylabel("Dist. true DPP", fontsize=14)
    ax[1].set_title("Ground set of cardinality N=$N", fontsize=14)
    ax[1].legend(loc="lower right", fontsize=14)
    ax[1].set_xlim(0, 0.35)
    
    ax[1].annotate(
        L"$\mu=p_{\emptyset}/(1-p_{\emptyset})$",
        xy=(mutheo, 0.2),                 # point ciblé (sur le trait)
        xycoords="data",
        xytext=(50, 10),               # décalage automatique (en points)
        textcoords="offset points",
        arrowprops=Dict(
            "arrowstyle" => "->",
            "linewidth"  => 1.2
        ),
        ha="center",
        va="bottom",
        fontsize=13
    )

    name = joinpath(FIGURES_DIR, savename*".pdf")
    fig.savefig(name)
    println("Saved "*name)
end
println("--- Function generate_figure2() now defined. ---")
println("-"^70)
