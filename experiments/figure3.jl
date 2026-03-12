# experiments/figure3.jl

using DppKernelEstimation
import PythonPlot as plt
using LinearAlgebra
using Format
import Random
using LaTeXStrings


parent_dir = dirname(@__DIR__)
const FIGURES_DIR = joinpath(parent_dir, "figures")
if !isdir(FIGURES_DIR)
    mkdir(FIGURES_DIR)
end


function fixed_point_amazon_dataset(categorynum, max_iterations;
                                    savename="fig", verbose=false)
    categories = ["apparel", "bath", "bedding", "carseats", "decor", "diaper",
                  "feeding", "furniture", "gear", "gifts", "health", "media",
                  "moms", "pottytrain", "safety", "strollers", "toys"]

    category = categories[categorynum]

    samples, N, nsamples = DppKernelEstimation.DataLoad.load_amazon(categorynum)
    
    da = DppSamples(samples, N)
    learndata = DppProb(da)                # DppProb(samples, N) non défini
    if verbose
        println("--- Running fixed point on Amazon Baby Registry dataset.")
        println("--- Category $categorynum ($(categories[categorynum]))---")
        println("--- Category no $categorynum ($(categories[categorynum]))---")
        isempty(setdiff(union(samples...), 1:N)) ? msg = "ok!" : msg = "FAILED!"
        println("Union_i X_i =       \\mX : "*msg)
        isempty(intersect(samples...)) ? msg = "ok!" : msg = "FAILED!"
        println("Inter_i X_i = \\emptyset : "*msg)
        [] in samples ? msg = "ok!" : msg = "FAILED!"
        println("Empty set in samples    : "*msg)
        println("----------------------")
    end

    fpparams = (maxiter=max_iterations, step=1, prec=0) # fp alg parameters
    regulparams = (ε=1e-6, λ=0, λ₁=0, λₙ=0)
    
    L1, crit1, (_, norml1, negloglik1) = mle_fixpt(learndata, regulparams.λ,
                                                   params=fpparams,
                                                   init=:determ,
                                                   testdata=nothing)
    
    fig, ax = plt.subplots(2, 1, clear=true, layout="constrained")
    ax[0].plot(crit1)
    ax[0].set_title(L"Likelihood criterion: $f_{\mathrm{ML}}(L_k)$")
    ax[1].plot(norml1)
    ax[1].set_title(L"Norm of the iterates: $\|L_k\|_{\mathrm{F}}$")
    ax[1].set_xlabel(L"Iteration number: $k$")
    fig.suptitle("Amazon baby registry set: Category number $categorynum")
    # fig.tight_layout()

    name = joinpath(FIGURES_DIR, savename*".pdf")
    fig.savefig(name)

end


function generate_figure3()
    fixed_point_amazon_dataset(1, 5000, savename="figure3_left")
    fixed_point_amazon_dataset(7, 20000, savename="figure3_right")
end
println("--- Function generate_figure3() now defined. ---")
println("-"^70)
