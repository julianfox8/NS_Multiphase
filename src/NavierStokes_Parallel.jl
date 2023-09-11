module NavierStokes_Parallel

export run_solver, parameters, VFcircle, VFsphere, VFbubble2d, VFbubble3d, @unpack

using MPI
using UnPack
using OffsetArrays
using Printf
using StaticArrays
using FoldsThreads
using FLoops
using .Threads
using NLsolve
using Statistics
using LinearAlgebra

include("Parameters.jl")
include("Mesh.jl")
include("Parallel.jl")
include("Tools.jl")
include("Velocity.jl")
include("Transport.jl")
include("Pressure.jl")
# include("Pressure_semilag.jl")
include("VF.jl")
include("VFgeom.jl")
include("ELVIRA.jl")
include("WriteData.jl")

function run_solver(param, IC!, BC!)
    @unpack stepMax,tFinal,solveNS = param

    # Create parallel environment
    par_env = parallel_init(param)
    @unpack isroot = par_env

    if isroot; println("Starting solver ..."); end 
    print("on $(nthreads()) threads\n")

    # Create mesh
    mesh = create_mesh(param,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # Create work arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz = initArrays(mesh)

    # Create initial condition
    t = 0.0 :: Float64
    IC!(P,u,v,w,VF,mesh)
    #printArray("VF",VF,par_env)

    # Apply boundary conditions
    BC!(u,v,w,mesh,par_env)

    # Update processor boundaries (overwrites BCs if periodic)
    # update_borders!(u,mesh,par_env)
    # update_borders!(v,mesh,par_env)
    # update_borders!(w,mesh,par_env)
    # update_borders!(VF,mesh,par_env)

    # Compute density and viscosity at intial conditions
    compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

    # Create face velocities
    interpolateFace!(u,v,w,uf,vf,wf,mesh)

    # Compute band around interface
    computeBand!(band,VF,param,mesh,par_env)


    # Compute interface normal 
    computeNormal!(nx,ny,nz,VF,param,mesh,par_env)

    # Compute PLIC reconstruction 
    computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)

    dt = compute_dt(u,v,w,param,mesh,par_env)

    # Check semi-lagrangian divergence
    divg = divergence(uf,vf,wf,dt,band,mesh,par_env)

    
    # Initialize VTK outputs
    pvd,pvd_PLIC = VTK_init(param,par_env)

    # Output IC
    t_last =[-100.0,]
    h_last =[100]
    std_out(h_last,t_last,0,t,P,u,v,w,divg,0,param,par_env)
    VTK(0,t,P,u,v,w,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz)

    # Loop over time
    nstep = 0
    iter = 0

    while nstep<stepMax && t<tFinal

        # define density and viscosity for each iteration
        if iter > 0 
            compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)
        end

        # Update step counter
        nstep += 1

        # Compute timestep and update time
        dt = compute_dt(u,v,w,param,mesh,par_env)
        t += dt

        # Set velocity for iteration if not using Navier-Stokes solver
        if !solveNS
            defineVelocity!(t,u,v,w,uf,vf,wf,param,mesh)
        end
        # println("u-star before transport ", us[5,5,1])
        # Predictor step (including VF transport)
        transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,tmp1,tmp2,tmp3,tmp4,Curve,dt,param,mesh,par_env,BC!,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz)
        # println("u-star after transport ", us[5,5,1])
        if solveNS
            # if iter > 0
                # Create face velocities
                # println(us)
            interpolateFace!(us,vs,ws,uf,vf,wf,mesh)
                # println(uf)
            # end

            # # Call pressure Solver (handles processor boundaries for P)
            iter = pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mesh,par_env,denx,deny,denz)

            # Corrector face velocities
            corrector!(uf,vf,wf,P,dt,denx,deny,denz,mesh)

            # Interpolate velocity to cell centers (keeping BCs from predictor)
            interpolateCenter!(u,v,w,us,vs,ws,uf,vf,wf,mesh)


            # Update Processor boundaries
            update_borders!(u,mesh,par_env)
            update_borders!(v,mesh,par_env)
            update_borders!(w,mesh,par_env)
        end

        
        # # Check divergence
        divg = divergence(u,v,w,dt,band,mesh,par_env)
        # println("mean of dP = ",mean(P[:,begin,:]-P[:,end,:]))
        println("Max divergence occurs at ",argmax(divg), " with value ", maximum(abs.(divg))) 
        println(jmin_," ",jmax_)
        # Check semi-lagrangian divergence
        # divg = semi_lag_divergence(uf,vf,wf,dt,mesh,par_env)

        # Output
        std_out(h_last,t_last,nstep,t,P,u,v,w,divg,iter,param,par_env)
        VTK(nstep,t,P,u,v,w,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz)

    end

    # Finalize
    #VTK_finalize(pvd) (called in VTK)
    #parallel_finalize()

end # run_solver

# Precompile
include("precompile.jl")

end
