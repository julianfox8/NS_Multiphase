using NavierStokes_Parallel
using Test
using MPI

NS = NavierStokes_Parallel

function testValue(test,value,expected,tol)
    if maximum(abs.(value.-expected)) <= tol 
        return true
    else
        println("Unexpected value in test: $test")
        println("Computed Value $value")
        println("Expected Value $expected")
        println("Specified tolerance $tol")
        return false
    end
end

@testset "NavierStokes_Parallel.jl" begin
    
    # # Example 1 - serial
    # include("../examples/example1.jl")
    # @test true
    
    # # Example 2
    # nprocs = 2; # number of processes
    # run(`mpiexecjl --project=. -n $nprocs $(Base.julia_cmd()) examples/example2.jl`)
    # @test true

    # Test PLIC distance calc and cutting 
    # with random normals and volume fractions 
    begin
        test="test_PLIC_dist_cutting"
        include(test*".jl")
        for t=1:1000
            nx,ny,nz=rand(3)*2.0.-1.0
            VF=rand()
            norm=sqrt(nx^2+ny^2+nz^2)
            nx=nx/norm
            ny=ny/norm
            nz=nz/norm
            value,expected,tol = test_PLIC_dist_cutting(nx,ny,nz,VF) 
            @test testValue(test,value,expected,tol)
        end
    end
end

nothing
