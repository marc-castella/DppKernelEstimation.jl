# scripts/run_fwdbwd.jl

using DppKernelEstimation
using PythonPlot
using LinearAlgebra
using Printf

# Data generation
# ===============
N, n = 8, 50                    # groundset and sample size
Ltrue = makepsdmatrix(N, wparam = 3) # true kernel

data, _ = generatedppsamples(N, n, L=Ltrue)
learndata = DppProb(data)       # empirical probabilities

testdata = DppProb(data.L)      # with Ltrue

outparams = (maxiter=5000, step=1.5, prec=1e-4) # outer-loop parameters
inparams = (maxiter=1000, step=1.5, prec=1e-4)  # inner-loop parameters
regulparams1 = (ε=1e-6, λ=0, λ₁=0, λₙ=0) # regularization parameters

# Run algorithm (possible on learndata or data)
# =============================================
L1, crit1, (test1, norml1, negloglik1) = fwdbwd_all(learndata, regulparams1,
                                                    outparams=outparams,
                                                    inparams=inparams, init=:determ,
                                                    testdata=testdata)

L1, crit1, (test1, norml1, negloglik1) = fwdbwd_all(data, regulparams1,
                                                    outparams=outparams,
                                                    inparams=inparams, init=:determ,
                                                    testdata=testdata)

# Show results
# ============
if haskey(learndata.prob,[])
    if learndata.prob[[]] > 0
        msg = "Probability of ∅ in learndata > 0: likelihood coercive."
    else
        msg = "Probability of ∅ in learndata = 0: likelihood non coercive."
    end
else
    msg = "Probability of ∅ in learndata = 0: likelihood non coercive."
end
println(msg)
println("--- L matrix returned by the algorithm:")
display(L1)

fig, ax = pyplot.subplots(3, 1, num=1, clear=true)
ax[0].plot(1:outparams.maxiter, crit1, "b", label="Minimized criterion")
ax[0].plot(1:outparams.maxiter, negloglik1, "r", label="Neg-loglikelihood")
ax[0].plot(1:outparams.maxiter, test1["negloglik"], "g", label="Test data neg-noglikelihood")
# ax[0].set_title("Criterion/negloglik/test negloglik")
ax[0].legend()

ax[1].plot(1:outparams.maxiter, test1["tv"], "r", label="TV distance to true DPP")
# ax[1].set_title("Distance to true DPP")
ax[1].legend()

ax[2].plot(1:outparams.maxiter, norml1, "b", label="Norm of iterates L_k")
#ax[2].set_title("Norm of iterates L_k")
ax[2].legend()

fig.suptitle(msg)
