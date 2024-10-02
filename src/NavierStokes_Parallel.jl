module NavierStokes_Parallel

export run_solver, parameters, VFcircle, VFsphere, VFbubble2d, VFbubble3d, @unpack

using MPI
using HYPRE
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
using SparseArrays
using HYPRE.LibHYPRE


include("Parameters.jl")
include("Mesh.jl")
include("Parallel.jl")
include("Tools.jl")
include("Velocity.jl")
include("Transport.jl")
include("Pressure.jl")
include("VF.jl")
include("VFgeom.jl")
include("ELVIRA.jl")
include("WriteData.jl")
include("ReadData.jl")
# include("hyp.jl")

function run_solver(param, IC!, BC!, outflow,restart_files = nothing)
    @unpack stepMax,tFinal,solveNS,pressure_scheme,restart,tol = param

    # Create parallel environment
    par_env = parallel_init(param)
    @unpack isroot,irank = par_env

    if isroot; println("Starting solver ..."); end 
    print("on $(nthreads()) threads\n")

    # Create mesh
    mesh = create_mesh(param,par_env)
    @unpack x,xm,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # Create work arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz = initArrays(mesh)

    HYPRE.Init()
    
    p_min,p_max = prepare_indices(tmp3,par_env,mesh)
    
    MPI.Barrier(par_env.comm)

    # Check simulation param for restart
    if restart == true
        pvtk_file,pvd_file,pvtk_dict = gather_restart_files(restart_files,mesh,par_env)
        # domain_check(mesh,pvtk_dict)
        t,nstep = fillArrays(pvtk_file,pvd_file,pvtk_dict,P,uf,vf,wf,VF,param,mesh,par_env)
        Neumann!(VF,mesh,par_env)
        if isroot ; println("Solver restart at time: ", round(t,digits= 4)); end        # Update processor boundaries (overwrites BCs if periodic)
        update_VF_borders!(VF,mesh,par_env)
        # Create cell centered velocities
        interpolateCenter!(u,v,w,us,vs,ws,uf,vf,wf,mesh)
    else
        # Create initial condition
        t = 0.0 :: Float64
        nstep = 0
        IC!(P,u,v,w,VF,mesh)
        # Apply boundary conditions
        BC!(u,v,w,t,mesh,par_env)
        # Update processor boundaries (overwrites BCs if periodic)
        update_borders!(u,mesh,par_env)
        update_borders!(v,mesh,par_env)
        update_borders!(w,mesh,par_env)
        update_VF_borders!(VF,mesh,par_env)
        # Create face velocities
        interpolateFace!(u,v,w,uf,vf,wf,mesh)
    end

    MPI.Barrier(par_env.comm)

    # Initialize Jacobian matrix
    jacob_ref = Ref{HYPRE_IJMatrix}(C_NULL)
    HYPRE_IJMatrixCreate(par_env.comm,p_min,p_max,p_min,p_max,jacob_ref)
    jacob = jacob_ref[]
    HYPRE_IJMatrixSetObjectType(jacob,HYPRE_PARCSR)    
    HYPRE_IJMatrixInitialize(jacob)
    
    # Compute density and viscosity at intial conditions
    compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

    # Compute band around interface
    computeBand!(band,VF,param,mesh,par_env)

    # Compute interface normal 
    computeNormal!(nx,ny,nz,VF,param,mesh,par_env)

    # Compute PLIC reconstruction 
    computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)

    dt = compute_dt(u,v,w,param,mesh,par_env)

    # Check semi-lagrangian divergence
    divg = divergence(tmp1,uf,vf,wf,dt,band,mesh,param,par_env)
    # println(divg[imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1])

    # Initialize VTK outputs
    pvd,pvd_xface,pvd_yface,pvd_zface,pvd_PLIC = VTK_init(param,par_env)

    # Output IC
    t_last =[-100.0,]
    h_last =[100]
    std_out(h_last,t_last,nstep,t,P,u,v,w,divg,0,mesh,param,par_env)
    VTK(nstep,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_xface,pvd_yface,pvd_zface,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz)

    # Loop over time
    # nstep = 0
    iter = 0
    J=nothing
    while nstep<stepMax && t<tFinal

        # Update step counter
        nstep += 1

        # Compute timestep and update time
        dt = compute_dt(u,v,w,param,mesh,par_env)
        t += dt

        # Set velocity for iteration if not using Navier-Stokes solver
        if !solveNS
            defineVelocity!(t,u,v,w,uf,vf,wf,param,mesh)
        end
        
        # Predictor step (including VF transport)
        transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,tmp1,tmp2,tmp3,tmp4,Curve,dt,param,mesh,par_env,BC!,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,t)

        # Update density and viscosity with transported VF
        compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)
        
        #! test for setting density to 1
        # denx[:,:,:] .= 1.0
        # deny[:,:,:] .= 1.0
        # denz[:,:,:] .= 1.0


        if solveNS
  
            # Create face velocities
            interpolateFace!(us,vs,ws,uf,vf,wf,mesh)

            # # Call pressure Solver (handles processor boundaries for P)
            iter = pressure_solver!(P,uf,vf,wf,nstep,dt,band,VF,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,gradx,grady,gradz,outflow,BC!,jacob)
            # iter = pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,gradx,grady,gradz,outflow,J,nstep)

            # Corrector face velocities
            corrector!(uf,vf,wf,P,dt,denx,deny,denz,mesh)

            # Interpolate velocity to cell centers (keeping BCs from predictor)
            interpolateCenter!(u,v,w,us,vs,ws,uf,vf,wf,mesh)

            #! if setting density to 1 recompute properties here
            # compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

            # Update Processor boundaries
            update_borders!(u,mesh,par_env)
            update_borders!(v,mesh,par_env)
            update_borders!(w,mesh,par_env)
        end

        # # Check divergence
        divg = divergence(tmp1,uf,vf,wf,dt,band,mesh,param,par_env)

        # Output
        std_out(h_last,t_last,nstep,t,P,u,v,w,divg,iter,mesh,param,par_env)
        VTK(nstep,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_xface,pvd_yface,pvd_zface,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz)
        # error("stop")
    end

    # Finalize
    # VTK_finalize(pvd) (called in VTK)
    # parallel_finalize()

end # run_solver

# # Precompile
# include("precompile.jl")

end
