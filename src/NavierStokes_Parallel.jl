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
    @unpack isroot = par_env

    # Create mesh
    mesh = create_mesh(param,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # Create mask of object
    mask=mask_create(mask_obj,mesh);

    # Create work arrays
    P,u,v,w,us,vs,ws,uf,vf,wf = initArrays(mesh)

    # Create initial condition
    t = 0.0
    IC!(P,u,v,w,mesh)

    # Interpolate velocity to cell faces
    interpolateFace!(u,v,w,uf,vf,wf,mesh)

    # Apply boundary conditions
    BC!(uf,vf,wf,mesh,par_env)

    # Interpolate velocity to cell centers
    interpolateCenter!(u,v,w,uf,vf,wf,mesh)

    # Update Processor boundaries
    update_borders!(u,mesh,par_env)
    update_borders!(v,mesh,par_env)
    update_borders!(w,mesh,par_env)

    # Initialize VTK outputs
    pvd = VTK_init()

    # Loop over time
    nstep = 0
    while nstep<stepMax && t<tFinal

        # Update step counter
        nstep += 1

        # Compute timestep and update time
        dt = compute_dt(uf,vf,wf,param,mesh,par_env)
        t += dt;

        # Predictor step
        predictor!(us,vs,ws,u,v,w,dt,param,mesh,par_env,mask)

        # Create face velocities
        interpolateFace!(us,vs,ws,uf,vf,wf,mesh)

        # Apply boundary conditions to face velocities
        BC!(uf,vf,wf,mesh,par_env)

        # Call pressure Solver 
        pressure_solver!(P,uf,vf,wf,dt,param,mesh,par_env)

        # Corrector face velocities
        corrector!(uf,vf,wf,P,dt,param,mesh,mask)

        # Interpolate velocity to cell centers
        interpolateCenter!(u,v,w,uf,vf,wf,mesh)
        
        # Check divergence
        divg = divergence(uf,vf,wf,mesh,par_env)
        max_divg = parallel_max(maximum(divg),par_env)

        # Output
        VTK(nstep,t,P,u,v,w,divg,mesh,par_env,pvd)

        # Std-out 
        if isroot 
            rem(nstep,10)==1 && @printf(" Iteration  max(divg) \n")
            @printf(" %9i   %8.3g \n",nstep,max_divg)
        end

    end

    # Finalize
    VTK_finalize(pvd)
    #parallel_finalize()

end # run_solver

end
