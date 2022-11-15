using NavierStokes_Parallel
using Test
using MPI

@testset "NavierStokes_Parallel.jl" begin
    #run_solver(1,1)

    nprocs = 1; # number of processes
    #mpiexec() do mpirun # MPI wrapper
        run(`mpiexecjl --project=. -n $nprocs $(Base.julia_cmd()) examples/example1.jl`)
        @test true
    #end

    nprocs = 2; # number of processes
    #mpiexec() do mpirun # MPI wrapper
        run(`mpiexecjl --project=. -n $nprocs $(Base.julia_cmd()) examples/example2.jl`)
        
        @test true
    #end
end
