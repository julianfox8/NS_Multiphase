
# NS_Multiphase


[![CI](https://github.com/julianfox8/NS_Multiphase/actions/workflows/CI.yml/badge.svg)](https://github.com/julianfox8/NS_Multiphase/actions/workflows/CI.yml)


Mist.jl is high-performance Navier-Stokes solver written in Julia as a test bed for extending one of the state-of-the-art unsplit, geometric VOF routines known as the semi-Lagrangian. All current unsplit geometric VOF routines use some form of a flux-correction to handle a discrepancy with how the incompressibility constraint is satisfied (under a Finite Volume or Finite Difference discretization) and how the interface is advected (under the semi-Lagrangian discretization). The goal of this codebase is to evaluate and quantify the consequence of using flux-corrections as opposed to our novel solution of applying the semi-Lagrangian discretization to satisfy the incompressibility constraint, removing the need for flux-corrections.



---

## Installation

Activate your Julia environment and add the package:

```julia
using Pkg
Pkg.activate(".")       # optional, if you want a project environment
Pkg.add(url="https://github.com/julianfox8/NS_Multiphase.jl")
```

---

## Running your first simulation

Run on 1 processor
```
>> julia
julia>> include("examples/example1.jl")
```

Run on 2 processors
`>> mpiexecjl --project=. -np 1 julia examples/example1.jl`

### Optimization flags
This software does not use `@inbound`, `@fastmath`, etc. to speed up the software as it is meant as a prototyping & testing framework.  After ensuring the code runs properly for you problem, these optimizations can be enabled from the command line using the flags `--optimize=3 --math-mode=fast --check-bounds=no`.  

For example, 
```
mpiexecjl --project=. -np 1 julia --optimize=3 --math-mode=fast --check-bounds=no examples/example4_deformation.jl
```

also using thread based parallelism 
```
mpiexecjl --project=. -np 4 julia --optimize=3 --math-mode=fast --check-bounds=no --threads 10 examples/example5_deformation3D.jl
```

---

## Examples



---

## To-do's / Goals

-implement an error metric to quantify the difference between an anlytic pre-image and a flux-corrected (or pressure-corrected) pre-image
-continuously add unit tests and verification of the solver


>>>>>>> master
