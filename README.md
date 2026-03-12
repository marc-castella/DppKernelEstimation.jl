# DppKernelEstimation

Optimization programs for kernel estimation of a determinantal point process.

This repository accompanies the paper:

**Kernel Matrix Estimation of a Determinantal Point Process from a Finite Set of Samples: Properties and
Algorithms**,  
Marc Castella and Jean-Christophe Pesquet,  
Transaction on Machine Learning Research, 2026,  
OpenReview: <https://openreview.net/forum?id=Cyx9LwB5IN>

### Code organization

-   `src/`\
    Core implementation of all methods, functions, and algorithms
    introduced in the paper.

-   `experiments/`\
    Scripts used to reproduce the figures and table from the paper.

-   `generate_all.jl`\
    Entry point to run all experiments at once.

-   `scripts/`\
    Scripts to run each algorithm independently.


# Environment and setup

Tested with Julia 1.12.5 : all programs are implemented in Julia.\
To install an start Julia, run:
```bash
> git clone https://github.com/marc-castella/DppKernelEstimation.jl
> cd DppKernelEstimation.jl
> julia --project=.
```

``` julia
using Pkg
Pkg.instantiate()

# Don't forget to load the module (automatic in the provided scripts)
using DppKernelEstimation
```

# Basic usage
## Running the algorithms

Minimal standalone scripts are provided in `scripts/`to run each algorithm
independently:

-   **Fixed-point algorithm** from [Mariet et al. 2015]

``` bash
julia --project=. scripts/run_fixpt.jl
```

-   **Forward-backward (proximal) algorithm** from [Castella et al. 2026]

``` bash
julia --project=. scripts/run_fwdbwd.jl
```

-   **Low-rank algorithm** from [Gartrell et al. 2017]

``` bash
julia --project=. scripts/run_lowrk.jl
```

These scripts illustrate how to use the core methods on simple
instances.



## Reproducing the results

- The experiments used to generate the figures and table from the paper are available in the `experiments/` directory.
- Each script is self-contained (once the environment is instantiated) and can be run independently.
- Results (figures/tables) are generated automatically when running the scripts.
- Figures are saved in `figures/`.
- Total running time for `generate_all.jl`is ~30 min.

### Figures and table

-   **Figure 1**

``` bash
julia --project=. experiments/figure1.jl
```

-   **Figure 2**

``` bash
julia --project=. experiments/figure2.jl
```

-   **Figure 3**

``` bash
julia --project=. experiments/figure3.jl
```

-   **Figures 4 and 5**

``` bash
julia --project=. experiments/figure4_5.jl
```

-   **Table 1**

``` bash
julia --project=. experiments/table1.jl
```

### Run all experiments

To reproduce all figures and the table in a single run:

``` bash
julia --project=. generate_all.jl
```

# References

- Marc Castella and Jean-Christophe Pesquet. "Kernel Matrix Estimation of a Determinantal Point Process from a Finite Set of Samples: Properties and Algorithms" _Transactions on Machine Learning Research 2026_
- Takahiro Kawashima and Hideitsu Hino. "Minorization-Maximization for Learning Determinantal Point Processes," _Transactions on Machine Learning Research, November 2023_
- Mike Gartrell, Ulrich Paquet and Noam Koenigstein. "Low-rank factorization of determinantal point processes" _AAAI 2017_
- Zelda Mariet and Suvrit Sra. "Fixed-point Algorithms for Learning Determinantal Point Pr
ocesses," _ICML 2015_
- Jennifer Gillenwater, Alex Kulesza, Emily Fox and Ben Taskar. "Expectation-Maximization for Learning Determinantal Point Processes," _NeurIPS 2014_
