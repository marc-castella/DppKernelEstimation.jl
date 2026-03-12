# experiments/table1.jl

using DppKernelEstimation
using PythonPlot
using LinearAlgebra
using Printf
using Random


"""
    generate_table1()

Generate two dictionnarys with results in respective rows of Table 1.
"""
function generate_table1()
    println("--- Generating table with different values of ϵ ---")
    println("    WARNING: no random seed initialized.")
    println("             values are likely to differ slightly at each run.")

    N = 10
    # Draw kernel matrix
    Ltrue = makepsdmatrix(N)
    data = DppProb(Ltrue)
    pempty = data.prob[[]]
    mutheo = pempty/(1-pempty)

    # Real samples and empirical probability
    n = 50
    datasamp, _ = generatedppsamples(N, n, L=Ltrue, nempty=0)
    learndata = DppProb(datasamp)
    
    # Algorithm parameters
    outparams = (maxiter=1000, step=1.5, prec=1e-4)
    inparams = (maxiter=1000, step=1.5, prec=1e-4)

    # Run algorithm for different values of epsilon
    allepsilon = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    L_regul0 = Dict()
    L_regul1 = Dict()
    for (i, local_epsilon) in enumerate(allepsilon)
        # second row in article (mu = mutheo)
        regulparams1 = (ε=local_epsilon, λ=mutheo, λ₁=0, λₙ=0)
        L1, crit1, (test1, norml1, negloglik1) = fwdbwd_all(learndata, regulparams1,
                                                            outparams=outparams,
                                                            inparams=inparams, init=:determ)
        L_regul1[local_epsilon] = L1
        
        # first row in article (mu = 0)
        regulparams2 = (ε=local_epsilon, λ=0, λ₁=0, λₙ=0)
        L2, crit2, (test2, norml2, negloglik2) = fwdbwd_all(learndata, regulparams2,
                                                            outparams=outparams,
                                                            inparams=inparams, init=:determ)
        L_regul0[local_epsilon] = L2
    end

    # Store results in d0, d1 (d0 is first row in article)
    refkey = 1e-6
    L_regul1[refkey]
    d0, d1 = Dict(), Dict()
    for k in keys(L_regul1)
        d1[k] = norm(L_regul1[refkey] - L_regul1[k])/norm(L_regul1[refkey])
        d0[k] = norm(L_regul0[refkey] - L_regul0[k])/norm(L_regul0[refkey])    
    end
    println("--- Values computed; you may use print_sorted_dict to display results.")
    println("-"^70)
    return d0, d1
end


"""
    fmt(x; digits=2)

Format real number in LaTeX scientific notation:
x.xx \\times 10^{n}
"""
function fmt(x; digits=2)
    x == 0 && return "0"
    s = @sprintf("%.*e", digits, x)   # ex: "1.23e-04"
    a, b = split(s, 'e')
    return "$(a) \\times 10^{$(parse(Int,b))}"
    ### ALT ### return "$(a) e$(parse(Int,b))"
end


"""
    dicts_to_latex_table(d1::Dict, d2::Dict;
                              digits=2,
                              colnames=("clé", "valeur 1", "valeur 2"),
                              transpose=false)

Custom function to transform results from dict to LaTeX
"""
function dicts_to_latex_table(d1::Dict, d2::Dict;
                              digits=2,
                              colnames=("clé", "valeur 1", "valeur 2"),
                              transpose=false)

    keys_sorted = sort(collect(keys(d1)))
    io = IOBuffer()

    if !transpose
        # ---- Format standard : clés en lignes ----
        println(io, "\\begin{tabular}{lcc}")
        println(io, "\\hline")
        println(io, join(colnames, " & "), " \\\\")
        println(io, "\\hline")

        for k in keys_sorted
            v1 = fmt(d1[k]; digits=digits)
            v2 = fmt(d2[k]; digits=digits)
            # println(io, "$k & $v1 & $v2 \\\\")
            println(io, "$k & \$$(fmt(d1[k]; digits=digits))\$ & \$$(fmt(d2[k]; digits=digits))\$ \\\\")

        end

        println(io, "\\hline")
        println(io, "\\end{tabular}")

    else
        # ---- Format transposé : clés en colonnes ----
        n = length(keys_sorted)
        colspec = "l" * repeat("c", n)

        println(io, "\\begin{tabular}{$colspec}")
        println(io, "\\hline")

        # ligne d'en-tête : clés
        println(io, colnames[1], " & ", join(keys_sorted, " & "), " \\\\")
        println(io, "\\hline")

        # ligne dictionnaire 1
        # row1 = [fmt(d1[k]; digits=digits) for k in keys_sorted]
        row1 = ["\$$(fmt(d1[k]; digits=digits))\$" for k in keys_sorted]

        println(io, colnames[2], " & ", join(row1, " & "), " \\\\")

        # ligne dictionnaire 2
        # row2 = [fmt(d2[k]; digits=digits) for k in keys_sorted]
        row2 = ["\$$(fmt(d2[k]; digits=digits))\$" for k in keys_sorted]
        println(io, colnames[3], " & ", join(row2, " & "), " \\\\")

        println(io, "\\hline")
        println(io, "\\end{tabular}")
    end

    return String(take!(io))
end


"""
    print_sorted_dict(d::AbstractDict)

Custom function to print results in Julia REPL.
"""
function print_sorted_dict(d::AbstractDict)
    for k in sort(collect(keys(d)))
        @printf("%.2e => %.2e\n", k, d[k])
    end
end

println("--- Function generate_table1() now defined. ---")
println("-"^70)
