<<<<<<< HEAD
# NS_Multiphase
=======
# NavierStokes_Parallel

[![Build Status](https://github.com/markowkes/NavierStokes_Parallel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/markowkes/NavierStokes_Parallel.jl/actions/workflows/CI.yml?query=branch%3Amain)

Run on 1 processor
```
>> julia
julia>> include("examples/example1.jl")
```

Run on 2 processors
`>> mpiexecjl --project=. -np 1 julia examples/example1.jl`

## Optimization flags
This software does not use `@inbound`, `@fastmath`, etc. to speed up the software as it is meant as a prototyping & testing framework.  After ensuring the code runs properly for you problem, these optimizations can be enabled from the command line using the flags `--optimize=3 --math-mode=fast --check-bounds=no`.  

For example, 
```
mpiexecjl --project=. -np 1 julia --optimize=3 --math-mode=fast --check-bounds=no examples/example4_deformation.jl
```

also using thread based parallelism 
```
mpiexecjl --project=. -np 4 julia --optimize=3 --math-mode=fast --check-bounds=no --threads 10 examples/example5_deformation3D.jl
```
>>>>>>> master
