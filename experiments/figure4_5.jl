# experiments/figure4_5.jl

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
    function fig_compar_init(;N=8, n=20, mu=nothing,
                             savepdf=false,
                             savename="fig_compar_init.pdf", seedinit=nothing)

Figure illustration point fixe + forward-backward + low-rank. Minima locaux.
Valeurs avec simulations typiques et élégantes:
N=8, n=1, seedinit=27
N=8, n=20, seedinit=35
"""
function fig_compar_init(;N=8, n=20, mu=nothing,
                         savename="fig_compar_init", seedinit=nothing)
    Random.seed!(seedinit)
    # println("=== tmp2")
    # printfmtln("Simulations avec groundset-size N = {1} et n = {2} samples",
    #            N, n)

    # # Génération de données/DPP
    # # =========================
    Lrand = Hermitian(makepsdmatrix(N, wparam=N))
    if isfinite(n)
        data, _ = generatedppsamples(N, n, L=Lrand, nempty=0)
        sample = data.samples[1]
    else
        N > 5 ? (return "=== STOPPED: N might be to large for n infinite! ===") : nothing
        data = DppProb(Lrand)
    end
    # printfmtln("En le vrai L: negloglik(Lrand) = {}", negloglik(Lrand, data))
    
    # Paramètres algo
    # ===============
    fpparams = (maxiter=500, step=1, prec=0) # fp alg parameters
    lrparams = (maxiter=500, step=0.1, prec=0) # lr alg parameters
    outparams = (maxiter=500, step=1.5, prec=1e-4)
    inparams = (maxiter=1000, step=1.5, prec=1e-4)

    # Choix régularisation
    # ====================
    isnothing(mu) ? mu = 1/(det(Lrand+I)-1) : nothing
    # printfmtln("mu = {}", mu)
    regulparams = (ε=1e-4, λ=mu, λ₁=0, λₙ=0)
    
    fig, ax = plt.subplots(1, 3, clear=true,
                           figsize=[19.2, 4.8],
                           layout="constrained",
                           sharey=true)
    axfp = ax[0]
    axlr = ax[1]
    axfb = ax[2]                # retour COLT: supprimer ax[2] et adapter figsize
    
    # Minimisation critère
    # ====================
    L1_all, crit1_all, L2_all, crit2_all, L3_all, crit3_all = [], [], [], [], [], []
    NREAL = 10
    for nreal in 1:NREAL
        if nreal < NREAL
            Linit = makepsdmatrix(N, wparam=N)
        else                        # init diagonale
            Linit = I(N)
        end
        Vinit = cholesky(Linit)
        Vinit = Vinit.L[:,1:N]

        
        L1, crit1, (L_trace1, norml1, nloglik1) = mle_fixpt(data,
                                                            regulparams.λ,
                                                            params=fpparams, init=Linit)
    
        L2, crit2, (test2, norml2, negloglik2) = fwdbwd_all(data, regulparams,
                                                            outparams=outparams,
                                                            inparams=inparams,
                                                            init=Linit)
        
        L3, crit3, (test3, norml3, negloglik3, V3) = mle_lowrk(data,
                                                               regulparams.λ,
                                                               init=Vinit,
                                                               params=lrparams)
        tmp = eigen(L3)
        L3  = Hermitian(tmp.vectors*diagm(max.(1e-10, tmp.values))*transpose(tmp.vectors))
        
        append!(L1_all, [L1])
        append!(crit1_all, crit1[end])
        append!(L2_all, [L2])
        append!(crit2_all, crit2[end])
        append!(L3_all, [L3])
        append!(crit3_all, negloglik(L3, data, regulparams.λ))
        
        if nreal == 1           # label uniquement sur première courbe
            axfp.plot(1:fpparams.maxiter, crit1, color="black",
                      label="Random init.")
            axfb.plot(1:outparams.maxiter, crit2, color="black",
                     label="Random init.") # *string(regulparams.ε))
            axlr.plot(1:lrparams.maxiter, crit3, color="black",
                     label="Random init.") # *string(regulparams.ε))
        elseif nreal == NREAL       # init diagonale
            axfp.plot(1:fpparams.maxiter, crit1, ls="--", label="Diagonal init.")
            axfb.plot(1:outparams.maxiter, crit2, ls="--", label="Diagonal init.")
            axlr.plot(1:lrparams.maxiter, crit3, ls="--", label="Diagonal init.")
        else
            axfp.plot(1:fpparams.maxiter, crit1, color="black")
            axfb.plot(1:outparams.maxiter, crit2, color="black")
            axlr.plot(1:lrparams.maxiter, crit3, color="black")
        end
    end
    
    # ### distances entre estimations de L
    v1 = [dppdist(L1_all[i], L1_all[j], :tv)
          for (i, j) in  Iterators.product(1:NREAL, 1:NREAL)]
    v2 = [dppdist(L2_all[i], L2_all[j], :tv)
          for (i, j) in  Iterators.product(1:NREAL, 1:NREAL)]
    v3 = [dppdist(L3_all[i], L3_all[j], :tv)
          for (i, j) in  Iterators.product(1:NREAL, 1:NREAL)]
    
    v1moy = sum(v1)/(NREAL*(NREAL-1))
    v2moy = sum(v2)/(NREAL*(NREAL-1))
    v3moy = sum(v3)/(NREAL*(NREAL-1))
    
    axfp.legend(fontsize=14)
    axfp.set_xlabel("Algorithm iterations", fontsize=14)
    axfp.set_ylabel("Criterion", fontsize=14)
    axfp.set_title("Fixed-point algorithm", fontsize=14)
    
    axfb.legend(fontsize=14)
    axfb.set_xlabel("Algorithm iterations", fontsize=14)
    axfb.set_ylabel("Criterion", fontsize=14)
    axfb.set_title("FB algorithm", fontsize=14)
    
    axlr.legend(fontsize=14)
    axlr.set_xlabel("Algorithm iterations", fontsize=14)
    axlr.set_ylabel("Criterion", fontsize=14)
    axlr.set_title("Low-rank algorithm [Gartrell et al. 2017]", fontsize=14)

    # printfmtln("Fixed-point avg. dist. between estimates: {}", v1moy)
    # printfmtln("Low-rk VVT  avg. dist. between estimates: {}", v3moy)
    # printfmtln("Forwa-backw avg. dist. between estimates: {}", v2moy)
    # println(" crit1_all (fp) / crit3_all (lr) / crit2_all (fb)")
    # display(hcat(crit1_all, crit3_all, crit2_all))
    
    # println("===")
    name = joinpath(FIGURES_DIR, savename*".pdf")
    fig.savefig(name)
    println("Saved "*name)
end

generate_figure4() = fig_compar_init(N=8, n=1, seedinit=27, savename="figure4")
println("--- Function generate_figure4() now defined. ---")
generate_figure5() = fig_compar_init(N=8, n=20, seedinit=35, savename="figure5")
println("--- Function generate_figure5() now defined. ---")
println("-"^70)
