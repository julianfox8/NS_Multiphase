module NavierStokes_Parallel

export run_solver, parameters, VFcircle, VFsphere, VFbubble2d, VFellipbub3d, VFellipbub2d, VFdroplet2d, VFbubble3d, VFdroplet3d, @unpack

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
using HYPRE
using EzXML
using JSON

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

function run_solver(param, IC!, BC!)
    @unpack Nx,stepMax,tFinal,solveNS,pressure_scheme,restart,tol = param

    # Create parallel environment
    par_env = parallel_init(param)
    @unpack isroot,irank,nproc,comm = par_env

    if isroot; println("Starting solver ..."); end 
    print("on $(nthreads()) threads\n")

    # Create mesh
    mesh = create_mesh(param,par_env)
    @unpack dx,dy,dz,x,xm,imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # Create work arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,divg,mask,tets,verts,inds,vInds = initArrays(mesh)

    solveNS && HYPRE_Init()

    p_min,p_max = prepare_indices(tmp3,par_env,mesh)

    # Check simulation param for restart
    if restart == true
        t,nstep = fillArrays(P,uf,vf,wf,VF,param,mesh,par_env)
        if isroot ; println("Solver restart at time: ", round(t,digits= 4)); end        # Update processor boundaries (overwrites BCs if periodic)
        # Create cell centered velocities
        interpolateCenter!(u,v,w,us,vs,ws,uf,vf,wf,mesh)
        # Update Processor boundaries
        update_borders!(u,mesh,par_env)
        update_borders!(v,mesh,par_env)
        update_borders!(w,mesh,par_env)
    else
        # Create initial condition
        t = 0.0 :: Float64
        nstep = 0
        xo,yo,zo = IC!(P,u,v,w,VF,mesh)
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
    xo = 0.125
    yo = 0.125
    zo = 0.125


    # error("stop")
    # Initialize hypre matrices
    # jacob = HYPREMatrix(comm,Int32(p_min),Int32(p_max),Int32(p_min),Int32(p_max))
    # b_vec = HYPREVector(comm, Int32(p_min), Int32(p_max))
    # x_vec = HYPREVector(comm, Int32(p_min), Int32(p_max))

    jacob_ref = Ref{HYPRE_IJMatrix}(C_NULL)
    HYPRE_IJMatrixCreate(par_env.comm,p_min,p_max,p_min,p_max,jacob_ref)
    jacob = jacob_ref[]
    HYPRE_IJMatrixSetObjectType(jacob,HYPRE_PARCSR)    
    HYPRE_IJMatrixInitialize(jacob)

    b_ref = Ref{HYPRE_IJVector}(C_NULL)
    HYPRE_IJVectorCreate(par_env.comm,p_min,p_max,b_ref)
    b_vec = b_ref[]
    HYPRE_IJVectorSetObjectType(b_vec,HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(b_vec)

    x_ref = Ref{HYPRE_IJVector}(C_NULL)
    HYPRE_IJVectorCreate(par_env.comm,p_min,p_max,x_ref)
    x_vec = x_ref[]
    HYPRE_IJVectorSetObjectType(x_vec,HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(x_vec)
    
    # Compute density and viscosity at intial conditions
    compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

    # Compute band around interface
    computeBand!(band,VF,param,mesh,par_env)
    
    # Compute interface normal 
    # !restart && computeNormal!(nx,ny,nz,VF,param,mesh,par_env)
    computeNormal!(nx,ny,nz,VF,param,mesh,par_env)
    # Compute PLIC reconstruction 
    # !restart && computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)
    computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)
    !restart && csv_init!(param,par_env)
    grav_cl = grav_centerline(xo,yo,mesh,param,par_env)
    # bubble_height = term_vel(grav_cl,xo,yo,VF,D,param,mesh,par_env)
    # println(grav_cl)
    bubble_height = bub_height(grav_cl,xo,yo,zo,nx,ny,nz,D,mesh,param,par_env)

    # println(terminal_vel)
    # println(bubble_height)
    # error("stop")
    dt = compute_dt(u,v,w,param,mesh,par_env)

    divergence!(divg,uf,vf,wf,dt,band,verts,tets,param,mesh,par_env)

    # Initialize VTK outputs
    if restart == true && isroot == true
        pvd_file_cleanup!(t,param)
    end

    pvd,pvd_restart,pvd_PLIC = VTK_init(param,par_env)
    
    # Grab initial volume fraction sum
    VF_init = parallel_sum(VF[imin_:imax_,jmin_:jmax_,kmin_:kmax_]*dx*dy*dz,par_env)

    # Output IC
    t_last =[-100.0,]
    h_last =[100]

    std_out(h_last,t_last,nstep,t,P,VF,u,v,w,divg,VF_init,bubble_height,0,param,mesh,par_env)
    !restart && VTK(nstep,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_restart,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz,verts,tets)
    # error("stop")
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
        transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,mask,dt,param,mesh,par_env,BC!,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,t,verts,tets,inds,vInds)
        
        # Update bands with transported VF
        computeBand!(band,VF,param,mesh,par_env)

        # Update density and viscosity with transported VF
        compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)
    
        if solveNS
  
            # Create face velocities
            interpolateFace!(us,vs,ws,uf,vf,wf,mesh)
            
            # Call pressure Solver (handles processor boundaries for P)
            iter = pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,gradx,grady,gradz,verts,tets,BC!,jacob,b_vec,x_vec)

            # Corrector face velocities
            corrector!(uf,vf,wf,P,dt,denx,deny,denz,mesh)

            # Interpolate velocity to cell centers (keeping BCs from predictor)
            interpolateCenter!(u,v,w,us,vs,ws,uf,vf,wf,mesh)

            # Update Processor boundaries
            update_borders!(u,mesh,par_env)
            update_borders!(v,mesh,par_env)
            update_borders!(w,mesh,par_env)
    
        end

        # Compute case specific outputs
        # bubble_height = term_vel(grav_cl,xo,yo,VF,D,param,mesh,par_env)
        bubble_height = bub_height(grav_cl,xo,yo,zo,nx,ny,nz,D,mesh,param,par_env)
        # Check divergence
        divergence!(divg,uf,vf,wf,dt,band,verts,tets,param,mesh,par_env)

        # Output
        std_out(h_last,t_last,nstep,t,P,VF,u,v,w,divg,VF_init,bubble_height,iter,param,mesh,par_env)
        VTK(nstep,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_restart,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz,verts,tets)
        MPI.Barrier(comm)

    end

    # Finalize
    # VTK_finalize(pvd) (called in VTK)
    # parallel_finalize()

end # run_solver

# # Precompile
# include("precompile.jl")

end
