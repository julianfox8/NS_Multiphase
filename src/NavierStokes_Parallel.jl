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
include("Velocity.jl")
include("Pressure.jl")
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

        # Update step counter
        nstep += 1

        # Compute timestep and update time
        dt = compute_dt(u,v,w,param,mesh,par_env)
        t += dt;

        # Predictor step
        predictor!(us,vs,ws,u,v,w,dt,param,mesh,par_env,mask)

        # Apply boundary conditions
        BC!(us,vs,ws,mesh,par_env)

        # Create face velocities
        interpolateFace!(us,vs,ws,uf,vf,wf,mesh)

        # Call pressure Solver 
        pressure_solver!(P,uf,vf,wf,dt,param,mesh,par_env)

        # Corrector face velocities
        corrector!(uf,vf,wf,P,dt,param,mesh,mask)

        # Interpolate velocity to cell centers
        interpolateCenter!(u,v,w,uf,vf,wf,mesh)

        # Update Processor boundaries
        update_borders!(u,mesh,par_env)
        update_borders!(v,mesh,par_env)
        update_borders!(w,mesh,par_env)
        
        # Check divergence
        divg = divergence(uf,vf,wf,mesh,par_env)
        
        # Output
        std_out(nstep,t,P,u,v,w,divg,par_env)
        VTK(nstep,t,P,u,v,w,divg,param,mesh,par_env,pvd)

    end

    # Finalize
    #VTK_finalize(pvd) (called in VTK)
    #parallel_finalize()

end # run_solver

end
