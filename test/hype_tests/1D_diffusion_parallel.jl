using NavierStokes_Parallel
using OffsetArrays
using Printf
using Plots
using HYPRE.LibHYPRE
using HYPRE
using SparseArrays
using MPI


# HYPRE = HYPRE.LibHYPRE
NS = NavierStokes_Parallel

#Domain set up similar to "Numerical simulation of a single rising bubble by VOF with surface compression"
# Define parameters 
param = parameters(
    # Constants
    mu_liq=0.25,       # Dynamic viscosity
    mu_gas = 0.0001,
    rho_liq= 1000,           # Density
    rho_gas =0.1, 
    sigma = 0.0072, #surface tension coefficient
    gravity = 1e-2,
    Lx=1.0,            # Domain size 
    Ly=1/10,
    Lz=1/10,
    tFinal=100.0,      # Simulation time
 
    
    # Discretization inputs
    Nx=10,           # Number of grid cells
    Ny=1,
    Nz=1,
    stepMax=50,   # Maximum number of timesteps
    max_dt = 1e-3,
    CFL=0.4,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    std_out_period = 0.0,
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-6,

    # Processors 
    nprocx = 2,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = true,
    zper = true,

    # pressureSolver = "NLsolve",
    # pressureSolver = "Secant",
    pressureSolver = "sparseSecant",
    # pressureSolver = "GaussSeidel",
    # pressureSolver = "ConjugateGradient",
    # pressure_scheme = "finite-difference",
    iter_type = "standard",
    VTK_dir= "1D_diffusion"

)

function IC!(u,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,Lx,x = mesh
    # # #Sin IC
    # amp = 10.0
    # freq = 2.0
    # for i = imin_:imax_ 
    #     u[i,1,1] = amp * sin(2*pi*freq*x[i]/Lx)
    # end

    # #Gaussian kernel
    μ = Lx / 2.0  # Mean (center) of the Gaussian
    σ = 0.1     # Standard deviation (spread) of the Gaussian

    for k = 1, j = 1, i = imin_:imax_
        u[i, j, k] = 10*exp(-((x[i] - μ)^2/ (2 * σ^2)) )/ sqrt(2π * σ^2)
    end
end

# function diffusion(param,IC!)
#     par_env = NS.parallel_init(param)
#     @unpack isroot = par_env

#     if isroot; println("Starting solver ..."); end 
#     print("on $(Threads.nthreads()) threads\n")


#     # Create mesh
#     mesh = NS.create_mesh(param,par_env)

#     # Create work arrays
#     P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz = NS.initArrays(mesh)

#     @unpack Nx,dx,imin_,imax_,imino_,imaxo_ = mesh
#     @unpack mu_liq,CFL,max_dt,tFinal,stepMax = param

#     # Viscous Δt 
#     viscous_dt = dx/mu_liq
#     dt = min(max_dt,CFL*minimum(viscous_dt))

#     #Fourier mesh coefficients
#     F = dt*mu_liq/dx^2

#     #Apply IC
#     IC!(us,mesh)
#     # println(us)
#     # error("stop")
#     #setup data structures
#     A = OffsetArray{Float64}(undef,imino_:imaxo_,imino_:imaxo_); fill!(A,0.0)
#     b = OffsetArray{Float64}(undef,imin_:imax_); fill!(b,0.0)

#     for i = imin_:imax_
#         A[i,i-1] = -F
#         A[i,i+1] = -F
#         A[i,i] = 1 + 2*F
#     end
#     A[:,0] = A[0,:] = A[:,Nx+1] = A[Nx+1,:] .= 0
#     A[1,1] = A[Nx,Nx] = 1

#     t = 0.0 :: Float64
#     anim = @animate for n in range(0,stepMax)
#         # nstep += 1
#         for i = imin_:imax_
#             b[i]= us[i,1,1]
#         end
#         b[imin_] = b[imax_] = 0
#         A0 = A[imin_:imax_,imin_:imax_]

#         # u_new = A0\b
#         u[imin_:imax_,1,1] = A0\b


#         # println(u[imin_:imax_,1,1])
#         # error("stop")
#         us[imin_:imax_,1,1] = u[imin_:imax_,1,1]

#         plot(u[imin_:imax_,1,1], label = "Time step: $n",xlabel = "Location",ylabel="Value",)
#         ylims!(-100,100)
#     end

#     gif(anim,"diffusion_1D.gif",fps = 15 )
# end

# function diffusion_sparse(param,IC!)
#     par_env = NS.parallel_init(param)
#     @unpack isroot,irankx = par_env

#     if isroot; println("Starting solver ..."); end 
#     print("on $(Threads.nthreads()) threads\n")


#     # Create mesh
#     mesh = NS.create_mesh(param,par_env)

#     # Create work arrays
#     P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,main,lower,upper = NS.initArrays(mesh)

#     @unpack Nx,dx,imin_,imax_,imino_,imaxo_ = mesh
#     @unpack mu_liq,CFL,max_dt,tFinal,stepMax = param

#     # Viscous Δt 
#     viscous_dt = dx/mu_liq
#     dt = min(max_dt,CFL*minimum(viscous_dt))

#     #Fourier mesh coefficients
#     F = dt*mu_liq/dx^2

#     #Apply IC
#     IC!(us,mesh)

#     #setup data structures
#     offset = [-1,0,1]


#     main[:] .= 1 + 2*F
#     lower[:] .= -F
#     upper[:] .= -F

#     A = spdiagm(
#         offset[1] => lower[:],
#         offset[2] => main[:],
#         offset[3] => upper[:]
#     )

#     A[1,1] = A[Nx,Nx] = 1
#     display(A)
#     b = OffsetArray{Float64}(undef,imin_:imax_); fill!(b,0.0)
#     #Initialize HYPRE
#     HYPRE.Init()
#     t = 0.0 :: Float64
#     anim = @animate for n in range(0,stepMax)

#         # nstep += 1
#         for i = imin_:imax_
#             b[i]= us[i,1,1]
#         end
#         b[imin_] = b[imax_] = 0

#         # b0 = b[:]
#         b0 = zeros(eltype(b), length(b))  # Specify the type of b0

#         for i=imin_:imax_
#             b0[i] = b[i]
#         end
#         # #simple backslash
#         # u_new = A\b0

#         #HYPRE solve
#         solver = HYPRE.GMRES()
#         u_new = HYPRE.solve(solver,A,b0)


#         u[imin_:imax_,1,1] = u_new
#         us[imin_:imax_,1,1] = u[imin_:imax_,1,1]

#         plot(u[imin_:imax_,1,1], label = "Time step: $n",xlabel = "Location",ylabel="Value",)
#         ylims!(-100,100)
#     end

#     gif(anim,"diffusion_1D.gif",fps = 15 )
# end

function diffusion_parallel(param,IC!)
    par_env = NS.parallel_init(param)
    @unpack iroot,isroot,irankx = par_env

    if isroot; println("Starting solver ..."); end 
    print("on $(Threads.nthreads()) threads\n")


    # Create mesh
    mesh = NS.create_mesh(param,par_env)

    # Create work arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,main,lower,upper = NS.initArrays(mesh)

    @unpack Nx,dx,imin_,imax_,imino_,imaxo_ = mesh
    @unpack mu_liq,CFL,max_dt,tFinal,stepMax = param

    # Viscous Δt 
    viscous_dt = dx/mu_liq
    dt = min(max_dt,CFL*minimum(viscous_dt))

    #Fourier mesh coefficients
    F = dt*mu_liq/dx^2

    #Apply IC
    IC!(us,mesh)

    #setup data structures
    offset1 = [-1,0,1]
    offset2 = [0,1,2]


    main[:] .= 1 + 2*F
    lower[:] .= -F
    upper[:] .= -F
    if irankx == 0
        A = spdiagm(imax_-imin_+1,imax_-imin_+2,
            offset1[1] => lower[imin_:imax_-1],
            offset1[2] => main[:],
            offset1[3] => upper[:]
        )
    elseif irankx == 1
        A = spdiagm(imax_-imin_+1,imax_-imin_+2,    
            offset2[1] => lower[:],
            offset2[2] => main[:],
            offset2[3] => upper[imin_:imax_-1]
        )
    end
    if imin_ == 1
        A[1,1] = 1.0
        A[1,2] = 0
    end 
    if imax_ == Nx
        A[imax_-imin_+1,imax_-imin_+2] = 1.0
        A[imax_-imin_+1,imax_-imin_+1] = 0
    end

    # if irankx == 0
    #     display(A)
    # end
    # MPI.Barrier(par_env.comm)
    # if irankx == 1
    #     display(A)
    #     error("stop")
    # end
    # g_main = MPI.Gather([A[i-imin_+1,i-imin_+1] for i in imin_:imax_],par_env.comm; root=0)
    # g_lower = MPI.Gather([A[i-imin_+2,i-imin_+1] for i in imin_+1:imax_],par_env.comm; root=0)
    # g_upper = MPI.Gather([A[i-imin_+1,i-imin_+2] for i in imin_+1:imax_],par_env.comm; root=0)

    # if irankx == 0
    #     println(size(g_lower)) 
    #     g_A = spdiagm(
    #         offset[1] => g_lower[:],
    #         offset[2] => g_main[:],
    #         offset[3] => g_upper[:]
    #     )
    #     # Print the global tridiagonal matrix
    #     display(g_A)
    #     error("stop")
    # end
    # if irankx == 0
    #     display(A)
    #     error("stop")
    # end
    # MPI.Barrier(par_env.comm)
    b = OffsetArray{Float64}(undef,imin_:imax_); fill!(b,0.0)
    # if irankx == 0
    #     println(us[:,1,1])
    # end
        #Initialize HYPRE
    HYPRE.Init()
    t = 0.0 :: Float64
    for n in range(0,stepMax)
        # nstep += 1
        for i = imin_:imax_
            b[i]= us[i,1,1]
        end
        if irankx == 0
            b[imin_] = 0
        end
        if irankx == 1 
            b[imax_] = 0
        end

        # if irankx == 0    
        #     sendbuf = OffsetArrays.no_offset_view(b[imax_])
        #     MPI.send([sendbuf],par_env.comm;dest=1)
        # elseif irankx == 1
        #     recbuf = MPI.recv(par_env.comm; source=0, tag=0) 
        #     b[imino_] = recbuf[1]
        # end

        # if irankx == 1    
        #     sendbuf = OffsetArrays.no_offset_view(b[imin_])
        #     MPI.send([sendbuf],par_env.comm;dest=0)
        # elseif irankx == 0
        #     recbuf = MPI.recv(par_env.comm; source=1, tag=0) 
        #     b[imaxo_] = recbuf[1]
        # end


        # data = MPI.Sendrecv!([sendbuf],[recbuff],par_env.comm;dest=1,sendtag=12,source=0,recvtag=12)
        b0 = OffsetArrays.no_offset_view(b)

        # b0 = zeros(eltype(b), length(b))  # Specify the type of b0

        # for i=imin_:imax_
        #     b0[i-imin_+1] = b[i]
        # end

        # # # g_b  = MPI.Gather(b[:],par_env.comm; root=0)
        # g_b0  = MPI.Gather(b0,par_env.comm; root=0)
        # if irankx ==0
        #     # display(g_b)
        #     println(g_b0)
        #     error("stop")
        # end
        # #simple backslash
        # u_new = A\b0

        MPI.Barrier(par_env.comm)

        #HYPRE solve
        solver = HYPRE.BiCGSTAB()
        us[imin_-imin_+1:imax_-imin_+1,1,1] = HYPRE.solve(solver,A,b0)
        

        if irankx == 0
            println(us[imin_-imin_+1:imax_-imin_+1,1,1])
            # println(axes(A))#,A)
            # println(axes(b0))
        end
        # MPI.Barrier(par_env.comm)
        # if irankx == 1
        #     println(u_new)
        #     error("stop")
        # end

        
        u[imin_:imax_,1,1] = u_new[imin_-imin_+1:imax_-imin_+1,1,1]
        us[imin_:imax_,1,1] = u[imin_:imax_,1,1]

        #update processor bounds
        MPI.Barrier(par_env.comm)

        g_u = MPI.Gather(u[imin_:imax_,1,1],par_env.comm; root =0)
        # if irankx == 0

        #     p = plot(xlims=(1, Nx), ylims=(0, 50), xlabel="Location", ylabel="Value")
        #     plot!(p,g_u, label = "Time step: $n")
        #     savefig(p,"test/1d_diff/time_$n.pdf")
        # end
    end
    NS.parallel_finalize()
    # gif(anim,"diffuion_1D.gif",fps = 15 ) 
end

function diffusion_hypre_parallel(param,IC!)
    par_env = NS.parallel_init(param)
    @unpack iroot,isroot,irankx = par_env

    if isroot; println("Starting solver ..."); end 
    print("on $(Threads.nthreads()) threads\n")
    HYPRE_Init()

    # Create mesh
    mesh = NS.create_mesh(param,par_env)

    # Create work arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,main,lower,upper = NS.initArrays(mesh)

    @unpack Nx,dx,imin_,imax_,imino_,imaxo_ = mesh
    @unpack mu_liq,CFL,max_dt,tFinal,stepMax = param

    # Viscous Δt 
    viscous_dt = dx/mu_liq
    dt = min(max_dt,CFL*minimum(viscous_dt))

    #Fourier mesh coefficients
    F = dt*mu_liq/dx^2

    #Apply IC
    IC!(us,mesh)

    #setup diags
    main[:] .= 1 + 2*F
    lower[:] .= -F
    upper[:] .= -F

    A_ref = Ref{HYPRE_IJMatrix}(C_NULL)
    HYPRE_IJMatrixCreate(par_env.comm,imin_,imax_,imin_,imax_,A_ref)
    A = A_ref[]

    HYPRE_IJMatrixSetObjectType(A,HYPRE_PARCSR)

    HYPRE_IJMatrixInitialize(A)
    
    for i in max(imin_,2):min(imax_,Nx-1)
        for offset in [-1, 0, 1]
            j = i + offset
            value = (i==j) ? 1+2*F : -F
            HYPRE_IJMatrixSetValues(A, 1,pointer([1]),pointer([i]),pointer([j]), pointer([value]))
        end
    end
    if imin_ == 1
        HYPRE_IJMatrixSetValues(A,1,pointer([1]),pointer([imin_]),pointer([imin_]),pointer([1]))
        # HYPRE_IJMatrixAddToValues(A,1,pointer([1]),pointer([imax_]),pointer([imax_+1]),pointer([-F]))
    end

    if imax_ == Nx
        HYPRE_IJMatrixSetValues(A,1,pointer([1]),pointer([imax_]),pointer([imax_]),pointer([1]))
        # HYPRE_IJMatrixAddToValues(A,1,pointer([1]),pointer([imin_]),pointer([imin_-1]),pointer([-F]))
    end

    HYPRE_IJMatrixAssemble(A)

    # HYPRE_IJMatrixPrint(A,"A")
    # error("stop")
    parcsr_A_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(A, parcsr_A_ref)
    parcsr_A = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_A_ref[])


    for n in range(0,stepMax)

        b_ref = Ref{HYPRE_IJVector}(C_NULL)
        HYPRE_IJVectorCreate(par_env.comm,imin_,imax_,b_ref)
        b = b_ref[]
        HYPRE_IJVectorSetObjectType(b,HYPRE_PARCSR)
        HYPRE_IJVectorInitialize(b)

        x_ref = Ref{HYPRE_IJVector}(C_NULL)
        HYPRE_IJVectorCreate(par_env.comm, imin_, imax_, x_ref)
        x = x_ref[]
        HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR)
        HYPRE_IJVectorInitialize(x)

        for i in imin_:imax_
            HYPRE_IJVectorSetValues(b, 1, pointer([i]), pointer([us[i,1,1]]))
            HYPRE_IJVectorSetValues(x,1,pointer([i]),pointer([0.0]))
        end
        
        # Adjust values for boundary conditions
        if irankx == 0
            HYPRE_IJVectorSetValues(b, 1, pointer([imin_]), pointer([0.0]))
        end
        
        if irankx == 1
            HYPRE_IJVectorSetValues(b, 1, pointer([imax_]), pointer([0.0]))
        end

        HYPRE_IJVectorAssemble(b)

        par_b_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(b, par_b_ref)
        par_b = convert(Ptr{HYPRE_ParVector}, par_b_ref[])

        HYPRE_IJVectorAssemble(x)

        par_x_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(x, par_x_ref)
        par_x = convert(Ptr{HYPRE_ParVector}, par_x_ref[])

        # # Create a solver object
        solver_ref = Ref{HYPRE_Solver}(C_NULL)

        # # Create solver
        # HYPRE_ParCSRPCGCreate(par_env.comm, solver_ref)
        # solver = solver_ref[]

        # # Set some parameters (See Reference Manual for more parameters)
        # HYPRE_PCGSetMaxIter(solver, 1000) # max iterations
        # HYPRE_PCGSetTol(solver, 1e-7) # conv. tolerance
        # HYPRE_PCGSetTwoNorm(solver, 1) # use the two norm as the stopping criteria
        # HYPRE_PCGSetPrintLevel(solver, 2) # prints out the iteration info
        # HYPRE_PCGSetLogging(solver, 1) # needed to get run info later

        # # Now setup and solve!
        # HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x)
        # HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x)

        # HYPRE_IJVectorPrint(b,"b")
        # HYPRE_IJVectorPrint(x,"x")

        # Create solver
        HYPRE_BoomerAMGCreate(solver_ref)
        solver = solver_ref[]

        # Set some parameters (See Reference Manual for more parameters)
        HYPRE_BoomerAMGSetPrintLevel(solver, 3) # print solve info + parameters
        HYPRE_BoomerAMGSetOldDefault(solver)    # Falgout coarsening with modified classical interpolaiton
        HYPRE_BoomerAMGSetRelaxType(solver, 3)  # G-S/Jacobi hybrid relaxation
        HYPRE_BoomerAMGSetRelaxOrder(solver, 1) # uses C/F relaxation
        HYPRE_BoomerAMGSetNumSweeps(solver, 1)  # Sweeeps on each level
        HYPRE_BoomerAMGSetMaxLevels(solver, 20) # maximum number of levels
        HYPRE_BoomerAMGSetTol(solver, 1e-7)     # conv. tolerance

        # Now setup and solve!
        HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x)
        HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x)

        # HYPRE_IJVectorGetValues(x,imax_-imin_+1,pointer(indices),pointer([us[imin_:imax_,1,1]]))
        
        int_x = zeros(1)
        for i in imin_:imax_
            HYPRE_IJVectorGetValues(x,1,[i],int_x)
            us[i,1,1] = int_x[1]
        end
        # if irankx ==0
        #     println(us[:,1,1])
        #     # error("stop")
        # end
        # MPI.Barrier(par_env.comm)
        # if irankx ==1
        #     println(us[:,1,1])
        #     error("stop")
        # end
        # Destroy solver
        HYPRE_ParVectorDestroy(par_x)
        HYPRE_ParVectorDestroy(par_b)
        HYPRE_ParCSRPCGDestroy(solver)

        g_u = MPI.Gather(us[imin_:imax_,1,1],par_env.comm; root =0)
        if irankx == 0
            p = plot(xlims=(1, Nx), ylims=(0, 50), xlabel="Location", ylabel="Value")
            plot!(p,g_u, label = "Time step: $n")
            savefig(p,"test/1d_diff/time_$n.pdf")
        end
    end
    parallel_finalize()
end

# @time diffusion_sparse(param,IC!)
# @time diffusion_parallel(param,IC!)
@time diffusion_hypre_parallel(param,IC!)
# @time diffusion(param,IC!)

# gif(anim,"diffusion_1D.gif",fps = 15 )



