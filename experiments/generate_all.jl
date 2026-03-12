# experiments/generate_all.jl

using PythonPlot: pygui

pygui(false)

include("table1.jl")
include("figure1.jl")
include("figure2.jl")
include("figure3.jl")
include("figure4_5.jl")

### Table 1
row1, row2 = generate_table1()
println("="^70)
println("--- Table 1: first row")
print_sorted_dict(d0)
println("--- Table1: second row")
print_sorted_dict(d1)
### Figures
generate_figure1()
println("===> Figure 1 (left and right) have been generated in directory figures/`.")
generate_figure2()
println("===> Figure 2 has been generated in directory figures/`.")
generate_figure3()
println("===> Figures 3 (left and right) have been generated in directory figures/`.")
generate_figure4()
println("===> Figure 4 has been generated in directory figures/`.")
generate_figure5()
println("===> Figure 5 has been generated in directory figures/`.")

