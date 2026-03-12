# src/AffichDataIO.jl

module DataLoad
export load_amazon
using CSV, DataFrames
using Distributions: sample

"""
    load_amazon(no=nothing)

Load data from Amazon Baby Registry.
(17 categories)

Returns:
- samples
- N
- nsamples
"""
function load_amazon(no=nothing, verbose=false)
    amazon_dir = joinpath(@__DIR__, "../data", "Amazon-baby-registry")
    categories = ["apparel", "bath", "bedding", "carseats", "decor", "diaper",
                  "feeding", "furniture", "gear", "gifts", "health", "media",
                  "moms", "pottytrain", "safety", "strollers", "toys"]
    
    if typeof(no) == String
        category = no
    elseif typeof(no) == Int
        category = categories[no]
    elseif no == nothing
        category = "apparel"
    end
    reg_name = "1_100_100_100_$(category)_regs.csv"
    txt_name = "1_100_100_100_$(category)_item_names.txt"

    samples = CSV.read(joinpath(amazon_dir, reg_name), DataFrame, header = 0,
                       silencewarnings=true) |>
        eachrow .|>
        Vector .|>
        skipmissing .|>
        collect

    N = length(readlines(joinpath(amazon_dir, txt_name))) # num. items groundset
    nsamples = length(samples)                            # num. samples
    if verbose
        println("-"^50)
        println("Loading from Amazon Baby Registry: "*category)
        println("Ground set size = $N / Number of samples = $(nsamples)")
        println("-"^50)
    end
    return samples, N, nsamples
end

end                             # module DataLoad
