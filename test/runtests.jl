using NavierStokes_Parallel
using Test
using MPI

@testset "NavierStokes_Parallel.jl" begin
    
    # Example 1 - serial
    include("../examples/example1.jl")
    @test true
    
    # Example 2
    nprocs = 2; # number of processes
    run(`mpiexecjl --project=. -n $nprocs $(Base.julia_cmd()) examples/example2.jl`)
    @test true
    
end
