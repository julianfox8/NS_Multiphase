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
# include("hyp.jl")

function run_solver(param, IC!, BC!, outflow,restart_files = nothing)
    @unpack stepMax,tFinal,solveNS,pressure_scheme,restart,tol = param

    # Create parallel environment
    par_env = parallel_init(param)
    @unpack isroot,irank,nproc,comm = par_env

    if isroot; println("Starting solver ..."); end 
    print("on $(nthreads()) threads\n")

    # Create mesh
    mesh = create_mesh(param,par_env)
    @unpack x,xm,imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # Create work arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz = initArrays(mesh)

    HYPRE.Init()
    
    
    p_min,p_max = prepare_indices(tmp3,par_env,mesh)
    # #! compute the map from i,j,k indices to vec index
    # local_tuples = [((i, j, k),(tmp3[i, j, k])) for i in imin_:imax_, j in jmin_:jmax_, k in kmin_:kmax_]
    
    # #! allocate sendbuff on root proc
    # if isroot
    #         vec2mats = Array{Tuple{Tuple{Int,Int,Int},Float64 }}(undef, imax+1,jmax+1,kmax+1)
    # else
    #     vec2mats = nothing
    # end

    # #! gather local_tuples
    # MPI.Gather!(local_tuples, vec2mats , 0 ,par_env.comm)

    # #! store tuples in dict for mapping in test_V1.jl
    # if isroot
    #     ind_dict = Dict{Tuple{Int, Int, Int}, Float64}()
    #     for t in vec2mats
    #         ind, value = t
    #         if value >= 1.0
    #             ind_dict[ind] = value
    #         end
    #     end
    #     open("vec2mat_dict.json","w") do f
    #         JSON.print(f,ind_dict)
    #     end
    #     error("stop")
    # end
    MPI.Barrier(comm)

    # for i in 0:nproc-1
    #     if irank == i
    #         println("p_min=",p_min)
    #         println("p_max=",p_max)
    #         println("size =",p_max-p_min)
    #     end
    #     MPI.Barrier(par_env.comm)
    # end

    # Check simulation param for restart
    if restart == true
        # pvtk_file,pvd_file,pvtk_dict = gather_restart_files(restart_files,mesh,par_env)
        # domain_check(mesh,pvtk_dict)
        t,nstep = fillArrays(restart_files,P,uf,vf,wf,VF,param,mesh,par_env)
        Neumann!(VF,mesh,par_env)
        if isroot ; println("Solver restart at time: ", round(t,digits= 4)); end        # Update processor boundaries (overwrites BCs if periodic)
        update_VF_borders!(VF,mesh,par_env)
        update_xface_borders!(uf,mesh,par_env)
        update_yface_borders!(vf,mesh,par_env)
        update_zface_borders!(wf,mesh,par_env)
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
    # if parallel_max(abs.(divg),par_env) > tol
    #     error("divergence-free constraint not satisfied")
    # end
    # Initialize VTK outputs
    if restart == true && isroot == true
        pvd_file_cleanup!(restart_files,t)
    end

    pvd,pvd_xface,pvd_yface,pvd_zface,pvd_PLIC = VTK_init(param,par_env)

    # Output IC
    t_last =[-100.0,]
    h_last =[100]
    std_out(h_last,t_last,nstep,t,P,VF,u,v,w,divg,0,mesh,param,par_env)
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
        # compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)
        # for i in 0:nproc-1
        #     if irank == i
        #         println("iteration ",iter," on proc ",i)
        #         println("u_max=",maximum(us))
        #         println("u_min=",minimum(us))
        #         println("v_max=",maximum(vs))
        #         println("v_min=",minimum(vs))
        #         println("w_max=",maximum(ws))
        #         println("w_min=",minimum(ws))
        #     end
        #     MPI.Barrier(par_env.comm)
        # end
        
        if solveNS
  
            # Create face velocities
            interpolateFace!(us,vs,ws,uf,vf,wf,mesh)
            # for i in 0:nproc-1
            #     if irank == i
            #         println("iteration ",iter," on proc ",i)
            #         println("sfx_max=",maximum(sfx))
            #         println("sfx_min=",minimum(sfx))
            #         println("sfy_max=",maximum(sfy))
            #         println("sfy_min=",minimum(sfy))
            #         println("sfz_max=",maximum(sfz))
            #         println("sfz_min=",minimum(sfz))
            #     end
            #     MPI.Barrier(par_env.comm)
            # end
            # # Call pressure Solver (handles processor boundaries for P)
            iter = pressure_solver!(P,uf,vf,wf,nstep,dt,band,VF,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,gradx,grady,gradz,outflow,BC!,jacob)
            # iter = pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,gradx,grady,gradz,outflow,J,nstep)

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
        divg = divergence(tmp1,uf,vf,wf,dt,band,mesh,param,par_env)
        compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)
        # Output
        std_out(h_last,t_last,nstep,t,P,VF,u,v,w,divg,iter,mesh,param,par_env)
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
