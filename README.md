# NavierStokes_Parallel

[![Build Status](https://github.com/markowkes/NavierStokes_Parallel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/markowkes/NavierStokes_Parallel.jl/actions/workflows/CI.yml?query=branch%3Amain)

Run on 1 processor
```
>> julia
julia>> include("examples/example1.jl")
```

Run on 2 processors
`>> mpiexecjl --project=. -np 1 julia examples/example1.jl`