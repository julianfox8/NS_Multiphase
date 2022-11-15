"""
Soler based on
https://www.sciencedirect.com/science/article/pii/S0021999110001208?ref=cra_js_challenge&fr=RR-1
"""
module NavierStokes_Parallel

export run_solver, parameters, mask_object, @unpack

using MPI
using UnPack
using OffsetArrays
using Plots
using Printf

include("Parameters.jl")
include("Mesh.jl")
include("Parallel.jl")
include("Mask.jl")
include("Tools.jl")
include("BoundaryConditions.jl")
include("Velocity.jl")
include("Pressure.jl")
include("Poisson.jl")
include("WriteData.jl")

function run_solver(param, IC!, BC!; mask_obj=nothing)
    @unpack stepMax,tFinal = param

    # Create parallel environment
    par_env = parallel_init(param)

    # Create mesh
    mesh = create_mesh(param,par_env)

    # Create mask of object
    mask=mask_create(mask_obj,mesh);

    # Create work arrays
    P,u,v,w,us,vs,ws,Fx,Fy,Fz = initArrays(mesh)

    # Create initial condition
    t = 0.0
    IC!(P,u,v,w,mesh)

    # Apply boundary conditions
    BC!(u,v,w,mesh,par_env)

    # Update Processor boundaries
    update_borders!(u,mesh,par_env)
    update_borders!(v,mesh,par_env)
    update_borders!(w,mesh,par_env)

    # Initialize VTK outputs
    pvd = VTK_init()

    # Loop over time
    nstep = 0
    while nstep<stepMax && t<tFinal

        println("================ $nstep ===============")

        # Update step counter
        nstep += 1

        # Compute timestep and update time
        dt = compute_dt(u,v,w,param,mesh,par_env)
        t += dt;

        @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
        printArray("u",u[:,:,kmin_:kmax_],par_env)
        printArray("v",v[:,:,kmin_:kmax_],par_env)

        # Predictor step
        predictor!(us,vs,ws,P,u,v,w,Fx,Fy,Fz,dt,param,mesh,par_env,mask)

        # Apply boundary conditions
        BC!(us,vs,ws,mesh,par_env)


        printArray("us",us[:,:,kmin_:kmax_],par_env)
        printArray("vs",vs[:,:,kmin_:kmax_],par_env)

        # Call pressure Solver 
        pressure_solver!(P,us,vs,ws,dt,param,mesh,par_env)

        printArray("P",P[:,:,kmin_:kmax_],par_env)

        # Corrector step
        corrector!(u,v,w,us,vs,ws,P,dt,param,mesh,mask)

        # Check divergence
        divg = divergence(u,v,w,mesh,par_env)
        printArray("divg",divg[:,:,kmin_:kmax_],par_env)


        # Apply boundary conditions
        BC!(u,v,ws,mesh,par_env)

        printArray("u",u[:,:,kmin_:kmax_],par_env)
        printArray("v",v[:,:,kmin_:kmax_],par_env)

        # Output
        VTK(nstep,t,P,u,v,w,mesh,par_env,pvd)

    end

    # Finalize
    VTK_finalize(pvd)
    #parallel_finalize()

end # run_solver

end
