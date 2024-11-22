using NavierStokes_Parallel
using OffsetArrays
using Printf
using Plots
using HYPRE.LibHYPRE
using HYPRE
using SparseArrays
using MPI

# include("~/NS_Multiphase/src/Parallel.jl")


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
    Lx=10.0,            # Domain size 
    Ly=10.0,
    Lz=1/10,
    tFinal=100.0,      # Simulation time
 
    
    # Discretization inputs
    Nx=10,         # Number of grid cells
    Ny=10,
    Nz=1,
    stepMax=100,   # Maximum number of timesteps
    max_dt = 1e-3,
    CFL=0.4,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    std_out_period = 0.0,
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-6,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = true,

    # pressureSolver = "NLsolve",
    # pressureSolver = "Secant",
    pressureSolver = "sparseSecant",
    # pressureSolver = "GaussSeidel",
    # pressureSolver = "ConjugateGradient",
    # pressure_scheme = "finite-difference",
    iter_type = "standard",
    VTK_dir= "2D_diffusion"

)

function IC!(u,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,Lx,x,y,Ly,xm,ym,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack iroot,isroot,irankx,nprocx,iranky,irank,nproc,comm = par_env
    
    # # # #Sin IC
    # amp = 10.0
    # freq = 2.0
    # for j = jmin_:jmax_,i = imin_:imax_ 
    #     u[i,j,1] = amp * sin(2*pi*freq*x[i]/Lx)
    # end

    # #Gaussian kernel
    μ = Lx / 2.0  # Mean (center) of the Gaussian
    σ = 0.5     # Standard deviation (spread) of the Gaussian
    for n in 0:nproc
        if irank == n
            println("proc $n")
            for k = 1, j = jmin_:jmax_, i = imin_:imax_
                println("x loc $(x[i]) and y loc $(y[j])")
                u[i, j, k] = 10*exp(-((x[i] - μ)^2+(y[j]-μ)^2)  / (2 * σ^2)) / sqrt(2π * σ^2)
            end
        end
        MPI.Barrier(comm)
    end

    # for k = 1, j = jmin_:jmax_, i = imin_:imax_
    #     u[i, j, k] = 10*exp(-((x[i] - μ)^2+(y[j]-μ)^2)  / (2 * σ^2)) / sqrt(2π * σ^2)
    # end
end


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
    IC!(u,mesh)

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
            b[i]= u[i,1,1]
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
            println(u[imin_-imin_+1:imax_-imin_+1,1,1])
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
    @unpack iroot,isroot,irankx,nprocx,comm = par_env

    if isroot; println("Starting solver ..."); end 
    print("on $(Threads.nthreads()) threads\n")
    HYPRE_Init()

    # Create mesh
    mesh = NS.create_mesh(param,par_env)

    # Create work arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,main,lower,upper = NS.initArrays(mesh)

    @unpack Nx,Ny,dx,imin_,imino_,imax_,imaxo_,jmin_,jmino_,jmax_,jmaxo_,imin,imax,jmin,jmax = mesh
    @unpack mu_liq,CFL,max_dt,tFinal,stepMax = param

    # Viscous Δt 
    viscous_dt = dx/mu_liq
    dt = min(max_dt,CFL*minimum(viscous_dt))

    #Fourier mesh coefficients
    # F = dt*mu_liq/2*dx^2
    F = .5
    t = 0.0 :: Float64
    #Apply IC
    IC!(u,mesh,par_env)
    

    function n(i,j,Nx) 
        val = i + (j-1)*Nx
        # @show i,j,k,Ny,Nz,val
        return val
    end

    offset = [-Nx, -1, 0, 1, Nx] # Define the offset

    #! Questions:
    #? How careful do I need to be when using calls like MPI.Allgather!()?
    #? What was the point of the mask in the NGA code?

    function prepare_indices(temp_index,par_env,mesh)
        @unpack Nx,Ny,jmino_,imino_,jmaxo_,imaxo_,jmin_,imin_,jmax_,imax_= mesh
        @unpack comm,nproc,irank,iroot,isroot,iranky = par_env
        ntemps = 0
        local_ntemps = 0
        for j = jmin_:jmax_,i = imin_:imax_
                local_ntemps += 1
        end
    
        MPI.Allreduce!([local_ntemps], [ntemps], MPI.SUM, par_env.comm)
    
        ntemps_proc = zeros(Int, nproc)
    
        MPI.Allgather!([local_ntemps], ntemps_proc, comm)
    
        ntemps_proc = cumsum(ntemps_proc)
        local_count = ntemps_proc[irank+1] - local_ntemps
        for j = jmin_:jmax_, i = imin_:imax_
                local_count += 1
                temp_index[i, j,1] = local_count
        end

        MPI.Barrier(par_env.comm)

        NS.update_borders!(temp_index,mesh,par_env)
        
        temp_max = -1
        temp_min = maximum(temp_index)
        for j in jmin_:jmax_, i in imin_:imax_
            if temp_index[i,j,1] != -1
                temp_min = min(temp_min,temp_index[i,j,1])
                temp_max = max(temp_max,temp_index[i,j,1])
            end
        end
        return temp_min,temp_max
    
    end

    coeff_index = OffsetArray{Int32}(undef,imino_:imaxo_,jmino_:jmaxo_,1); fill!(coeff_index,0.0)

    temp_min,temp_max = prepare_indices(coeff_index,par_env,mesh)

    MPI.Barrier(par_env.comm)

    A_ref = Ref{HYPRE_IJMatrix}(C_NULL)
    # HYPRE_IJMatrixCreate(par_env.comm,temp_min,temp_max,n(imin,jmin,Nx),n(imax,jmax,Nx),A_ref)
    # HYPRE_IJMatrixCreate(par_env.comm,n(imin,jmin,Nx),n(imax,jmax,Nx),temp_min,temp_max,A_ref)
    HYPRE_IJMatrixCreate(par_env.comm,temp_min,temp_max,temp_min,temp_max,A_ref)
    A = A_ref[]
    
    HYPRE_IJMatrixSetObjectType(A,HYPRE_PARCSR)
    
    HYPRE_IJMatrixInitialize(A)

    function fill_matrix_coefficients!(matrix,coeff_index,offset, cols_, values_,mesh)
        @unpack  imin_, imax_, jmin_, jmax_,jmino_,imino_,jmaxo_,imaxo_ = mesh
        nrows = 1
        nx_ = imax_-imin_+1
        ny_ = jmax_-jmin_+1
        for j = jmin_:jmax_,i = imin_:imax_
            if i == imin || i == imax || j == jmin || j == jmax
                fill!(cols_,0)
                fill!(values_,0.0)
                nst = 1
                cols_[nst] = coeff_index[i,j]
                values_[nst] = 0.0
                ncols = nst
                rows_ = coeff_index[i,j]
                HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
            elseif coeff_index[i,j] != -1
                fill!(cols_,0)
                fill!(values_,0.0)
                nst = 0
                
                # Diagonal
                nst += 1
                cols_[nst] = coeff_index[i,j]
                values_[nst] = 1+4*F
    
                for st in offset
                    if st == 0
                        continue
                    end
                    # Left-Right
                    if nx_ != 1
                        ni = i + st # Get neighbor index in the x direction
                        nj = j # No change in the y direction
                        if ni in imin_:imax_ && nj in jmin_:jmax_ && coeff_index[ni,nj] != -1
                            nst += 1
                            cols_[nst] = coeff_index[ni,nj]
                            values_[nst] = -F
                        end
                    end
                    
                    # Top-Bottom
                    if ny_ != 1
                        ni = i # No change in the x direction
                        nj = j + st # Get neighbor index in the y direction
                        if ni in imin_:imax_ && nj in jmin_:jmax_ && coeff_index[ni,nj] != -1
                            nst += 1
                            cols_[nst] = coeff_index[ni,nj]
                            values_[nst] = -F
                        end
                    end
                end
    
                # Sort the points
                for st in 1:nst
                    ind = st + argmin(cols_[st:nst], dims=1)[1] - 1
                    tmpr = values_[st]
                    values_[st] = values_[ind]
                    values_[ind] = tmpr
                    tmpi = cols_[st]
                    cols_[st] = cols_[ind]
                    cols_[ind] = tmpi
                end
                
                ncols = nst
                rows_ = coeff_index[i,j]
    
                # Call function to set matrix values
                HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
            end
        end
    end
    cols_ = OffsetArray{Int32}(undef,1:size(offset,1)); fill!(cols_,0)
    values_ = OffsetArray{Float64}(undef,1:size(offset,1)); fill!(values_,0.0)
    fill_matrix_coefficients!(A,coeff_index,offset,cols_,values_,mesh)

    HYPRE_IJMatrixAssemble(A)
    HYPRE_IJMatrixPrint(A,"old_A")
    error("stop")
    parcsr_A_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(A, parcsr_A_ref)
    parcsr_A = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_A_ref[])

    divg = NS.divergence(uf,vf,wf,dt,band,mesh,param,par_env)

        
    # Initialize VTK outputs
    pvd,pvd_PLIC = NS.VTK_init(param,par_env)

    NS.VTK(0,t,P,u,v,w,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz)
        
    for step in 1:stepMax

        t+=dt
        b_ref = Ref{HYPRE_IJVector}(C_NULL)
        # HYPRE_IJVectorCreate(par_env.comm,temp_min,temp_max,b_ref)
        HYPRE_IJVectorCreate(par_env.comm,temp_min,temp_max,b_ref)
        b = b_ref[]
        HYPRE_IJVectorSetObjectType(b,HYPRE_PARCSR)
        HYPRE_IJVectorInitialize(b)

        x_ref = Ref{HYPRE_IJVector}(C_NULL)
        # HYPRE_IJVectorCreate(par_env.comm,temp_min,temp_max, x_ref)
        HYPRE_IJVectorCreate(par_env.comm,temp_min,temp_max,x_ref)
        x = x_ref[]
        HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR)
        HYPRE_IJVectorInitialize(x)

        for j in jmin_:jmax_, i in imin_:imax_
            row_ = coeff_index[i,j]
            HYPRE_IJVectorAddToValues(x,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            if i == imin
                HYPRE_IJVectorAddToValues(b, 1, pointer(Int32.([row_])), pointer(Float64.([0.0])))
            elseif i == imax
                HYPRE_IJVectorAddToValues(b, 1, pointer(Int32.([row_])), pointer(Float64.([0.0])))
            elseif j == jmin
                HYPRE_IJVectorAddToValues(b, 1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            elseif j == jmax
                HYPRE_IJVectorAddToValues(b, 1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            else
                HYPRE_IJVectorAddToValues(b, 1, pointer(Int32.([row_])), pointer(Float64.([u[i,j,1]])))
            end
        end

        HYPRE_IJVectorAssemble(b)

        par_b_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(b, par_b_ref)
        par_b = convert(Ptr{HYPRE_ParVector}, par_b_ref[])

        HYPRE_IJVectorAssemble(x)
        # HYPRE_IJVectorPrint(par_b_ref,"b2")
        MPI.Barrier(par_env.comm)
        # error("stop")

        par_x_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(x, par_x_ref)
        par_x = convert(Ptr{HYPRE_ParVector}, par_x_ref[])

        # # Create a solver object
        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)
        # # # Create solver
        # HYPRE_ParCSRPCGCreate(par_env.comm, solver_ref)
        # solver = solver_ref[]

        # # Set some parameters (See Reference Manual for more parameters)
        # HYPRE_PCGSetMaxIter(solver, 1000) # max iterations
        # HYPRE_PCGSetTol(solver, 1e-7) # conv. tolerance
        # HYPRE_PCGSetTwoNorm(solver, 1) # use the two norm as the stopping criteria
        # # HYPRE_PCGSetPrintLevel(solver, 2) # prints out the iteration info
        # # HYPRE_PCGSetLogging(solver, 1) # needed to get run info later


        # HYPRE_IJVectorPrint(b,"b")
        # HYPRE_IJMatrixPrint(A,"A")
        # # Now setup and solve!
        # HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x)
        # HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x)
        MPI.Barrier(par_env.comm)

        # println("first iter")

        # Create solver
        # HYPRE_BoomerAMGCreate(solver_ref)
        # solver = solver_ref[]

        # # Set some parameters (See Reference Manual for more parameters)
        # HYPRE_BoomerAMGSetPrintLevel(solver, 3) # print solve info + parameters
        # HYPRE_BoomerAMGSetOldDefault(solver)    # Falgout coarsening with modified classical interpolaiton
        # HYPRE_BoomerAMGSetRelaxType(solver, 3)  # G-S/Jacobi hybrid relaxation
        # HYPRE_BoomerAMGSetRelaxOrder(solver, 1) # uses C/F relaxation
        # HYPRE_BoomerAMGSetNumSweeps(solver, 3)  # Sweeeps on each level
        # HYPRE_BoomerAMGSetMaxLevels(solver, 20) # maximum number of levels
        # HYPRE_BoomerAMGSetTol(solver, 1e-7)     # conv. tolerance

        # # Now setup and solve!
        # HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x)
        # HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x)

        HYPRE_ParCSRPCGCreate(par_env.comm, solver_ref)
        solver = solver_ref[]

        # Set some parameters (See Reference Manual for more parameters)
        HYPRE_PCGSetMaxIter(solver, 1000) # max iterations
        HYPRE_PCGSetTol(solver, 1e-7) # conv. tolerance
        HYPRE_PCGSetTwoNorm(solver, 1) # use the two norm as the stopping criteria
        # HYPRE_PCGSetPrintLevel(solver, 2) # print solve info
        # HYPRE_PCGSetLogging(solver, 1) # needed to get run info later

        # Now set up the AMG preconditioner and specify any parameters
        HYPRE_BoomerAMGCreate(precond_ref)
        precond = precond_ref[]
        # HYPRE_BoomerAMGSetPrintLevel(precond, 1) # print amg solution info
        HYPRE_BoomerAMGSetCoarsenType(precond, 6)
        HYPRE_BoomerAMGSetOldDefault(precond)
        HYPRE_BoomerAMGSetRelaxType(precond, 6) # Sym G.S./Jacobi hybrid
        HYPRE_BoomerAMGSetNumSweeps(precond, 1)
        HYPRE_BoomerAMGSetTol(precond, 0.0) # conv. tolerance zero
        HYPRE_BoomerAMGSetMaxIter(precond, 1) # do only one iteration!

        # Set the PCG preconditioner
        HYPRE_PCGSetPrecond(solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond)

        # Now setup and solve!
        HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x)
        HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x)

        MPI.Barrier(par_env.comm)
        

        int_x = zeros(1)
        for j in jmin_:jmax_,i in imin_:imax_
            HYPRE_IJVectorGetValues(x,1,pointer(Int32.([coeff_index[i,j,1]])),int_x)
            u[i,j,1] = int_x[1]
        end

        # Destroy solver
        HYPRE_ParVectorDestroy(par_x)
        HYPRE_ParVectorDestroy(par_b)
        HYPRE_ParCSRPCGDestroy(solver)

        # error("stop")
        if isroot == true
            println(step)
        end
        MPI.Barrier(par_env.comm)
        NS.VTK(step,t,P,u,v,w,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz)

    end
    # parallel_finalize()
end

function diffusion_hypre_wrap(param,IC!)
    par_env = NS.parallel_init(param)
    @unpack iroot,isroot,irankx,nprocx,comm,nproc,irank = par_env

    if isroot; println("Starting solver ..."); end 
    print("on $(Threads.nthreads()) threads\n")
    HYPRE_Init()

    # Create mesh
    mesh = NS.create_mesh(param,par_env)

    # Create work arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,divg,tets,verts,inds,vInds = NS.initArrays(mesh)

    @unpack Nx,Ny,dx,imin_,imino_,imax_,imaxo_,jmin_,jmino_,jmax_,jmaxo_,imin,imax,jmin,jmax = mesh
    @unpack mu_liq,CFL,max_dt,tFinal,stepMax = param

    # Viscous Δt 
    viscous_dt = dx/mu_liq
    dt = min(max_dt,CFL*minimum(viscous_dt))

    #Fourier mesh coefficients
    # F = dt*mu_liq/2*dx^2
    F = 0.25
    t = 0.0 :: Float64
    #Apply IC
    IC!(u,mesh,par_env)
    
    offset = [-Nx, -1, 0, 1, Nx] # Define the offset

    function prepare_indices(temp_index,par_env,mesh)
        @unpack Nx,Ny,jmino_,imino_,jmaxo_,imaxo_,jmin_,imin_,jmax_,imax_= mesh
        @unpack comm,nproc,irank,iroot,isroot,iranky = par_env
        ntemps = 0
        local_ntemps = 0
        for j = jmin_:jmax_,i = imin_:imax_
                local_ntemps += 1
        end
    
        MPI.Allreduce!([local_ntemps], [ntemps], MPI.SUM, par_env.comm)
    
        ntemps_proc = zeros(Int, nproc)
    
        MPI.Allgather!([local_ntemps], ntemps_proc, comm)
    
        ntemps_proc = cumsum(ntemps_proc)
        local_count = ntemps_proc[irank+1] - local_ntemps
        for j = jmin_:jmax_, i = imin_:imax_
                local_count += 1
                temp_index[i, j,1] = local_count
        end

        MPI.Barrier(par_env.comm)

        NS.update_borders!(temp_index,mesh,par_env)
        
        temp_max = -1
        temp_min = maximum(temp_index)
        for j in jmin_:jmax_, i in imin_:imax_
            if temp_index[i,j,1] != -1
                temp_min = min(temp_min,temp_index[i,j,1])
                temp_max = max(temp_max,temp_index[i,j,1])
            end
        end
        return temp_min,temp_max
    
    end

    coeff_index = OffsetArray{Int32}(undef,imino_:imaxo_,jmino_:jmaxo_,1); fill!(coeff_index,0.0)

    temp_min,temp_max = prepare_indices(coeff_index,par_env,mesh)
    println(typeof(temp_min))
    A = HYPREMatrix(par_env.comm, temp_min,temp_max,temp_min,temp_max)
    A_assembler= HYPRE.start_assemble!(A)

    function fill_matrix_coefficients!(matrix_assembler,coeff_index,offset, cols_, values_,mesh)
        @unpack  Nx,Ny,imin_, imax_, jmin_, jmax_,jmino_,imino_,jmaxo_,imaxo_ = mesh
        nrows = 1

        for j = jmin_:jmax_,i = imin_:imax_
            if i == 1 || i == Nx || j == 1 || j == Ny
                fill!(cols_,0)
                fill!(values_,0.0)
                nst = 1
                cols_[nst] = coeff_index[i,j]
                values_[nst] = 0.0
                ncols = nst
                rows_ = coeff_index[i,j]
                HYPRE.assemble!(matrix_assembler,[rows_],[cols_[1]],hcat(values_[1]))
            elseif coeff_index[i,j] != -1
                fill!(cols_,0)
                fill!(values_,0.0)
                nst = 0

                # Diagonal
                nst += 1
                cols_[nst] = coeff_index[i,j]
                values_[nst] = 1+4*F
    
                for st in offset
                    if st == 0
                        continue
                    end
                    # Left-Right
                    ni = i + st # Get neighbor index in the x direction
                    nj = j # No change in the y direction
                    
                    if ni in imin_:imax_ && nj in jmin_:jmax_ && coeff_index[ni,nj] != -1
                        nst += 1
                        cols_[nst] = coeff_index[ni,nj]
                        values_[nst] = -F
                    end
                
                    
                    # Top-Bottom
                    ni = i # No change in the x direction
                    nj = j + st # Get neighbor index in the y direction
                    if ni in imin_:imax_ && nj in jmin_:jmax_ && coeff_index[ni,nj] != -1
                        nst += 1
                        cols_[nst] = coeff_index[ni,nj]
                        values_[nst] = -F
                    end
                end
    
                # Sort the points
                for st in 1:nst
                    ind = st + argmin(cols_[st:nst], dims=1)[1] - 1
                    tmpr = values_[st]
                    values_[st] = values_[ind]
                    values_[ind] = tmpr
                    tmpi = cols_[st]
                    cols_[st] = cols_[ind]
                    cols_[ind] = tmpi
                end
                
                ncols = nst
                rows_ = coeff_index[i,j]

                # Call function to set matrix values
                HYPRE.assemble!(matrix_assembler,[rows_],cols_,reshape(values_,(1,size(cols_,1))))
            end
        end
    end

    # cols_ = OffsetArray{Int32}(undef,1:size(offset,1)); fill!(cols_,0)
    cols_ = Vector{Int32}(undef,size(offset,1) );fill!(cols_,0)
    # values_ = OffsetArray{Float64}(undef,1:size(offset,1)); fill!(values_,0.0)
    values_ = Matrix{Float64}(undef, 1, size(offset,1) );; fill!(values_,0.0)

    fill_matrix_coefficients!(A_assembler,coeff_index,offset,cols_,values_,mesh)

    A = HYPRE.finish_assemble!(A_assembler)
    HYPRE_IJMatrixPrint(A,"new_A")
    # divg = NS.divergence!(uf,vf,wf,dt,band,mesh,param,par_env)

    # Initialize VTK outputs
    pvd,pvd_PLIC = NS.VTK_init(param,par_env)

    # NS.VTK(0,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_restart,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz,verts,tets)
    # Define path for the GIF
    gif_path = "test/3d_surface_evolution.gif"
    fps = 10  # Frames per second

    # Initialize an empty list to store frames
    frames = []
    xlims = (1,Nx)
    ylims = (1,Ny)
    zlims = (0,5)
    for step in 1:stepMax

        t+=dt

        b = HYPREVector(par_env.comm,temp_min,temp_max)
        b_assembler = HYPRE.start_assemble!(b)

        

        for j in jmin_:jmax_, i in imin_:imax_
            row_ = coeff_index[i,j]
            if i == imin
                HYPRE.assemble!(b_assembler,[row_],[0.0])
            elseif i == imax
                HYPRE.assemble!(b_assembler,[row_],[0.0])
            elseif j == jmin
                HYPRE.assemble!(b_assembler,[row_],[0.0])
            elseif j == jmax
                HYPRE.assemble!(b_assembler,[row_],[0.0])
            else
                HYPRE.assemble!(b_assembler,[row_],[u[i,j,1]])
            end
        end

        b = HYPRE.finish_assemble!(b_assembler)
        MPI.Barrier(par_env.comm)
        
        # solver = HYPRE.BiCGSTAB(par_env.comm;Tol = 1e-5)
        precond = HYPRE.BoomerAMG(; RelaxType = 6, CoarsenType = 6)
        solver = HYPRE.GMRES(comm; Tol= 1e-6, Precond = precond)
        x = HYPRE.solve(solver,A,b)
        int_x = zeros(1)
        for j in jmin_:jmax_,i in imin_:imax_
            HYPRE_IJVectorGetValues(x,1,pointer(Int32.([coeff_index[i,j,1]])),int_x)
            u[i,j,1] = int_x[1]
        end


        MPI.Barrier(par_env.comm)
        g_u = MPI.Gather(u[imin_:imax_,jmin_:jmax_,1],par_env.comm; root =0)
        if isroot
            # Reshape `g_u` appropriately after gathering for plotting
            # Assuming `g_u` is gathered as a contiguous vector, reshape it
            # gathered_data = reshape(g_u, Nx, Ny)  # Adjust `Nx` and `Ny` as needed based on domain decomposition
            
            # # Define the x and y axis ranges for the plot
            # x_vals = range(1, stop=Nx, length=Nx)
            # y_vals = range(1, stop=Ny, length=Ny)

            # # Create a 3D surface plot
            # p = surface(x_vals, y_vals, gathered_data,
            #             xlabel="X Location", ylabel="Y Location", zlabel="Value",
            #             title="3D Surface at Time Step $step")
            
            # # Save the figure
            # savefig(p, "test/2d_diff/time_$step.pdf")

            # Reshape `g_u` into the correct dimensions
            gathered_data = reshape(g_u, Nx, Ny)  # Adjust Nx and Ny based on the domain decomposition
    
            # Define the x and y axis ranges for the plot
            x_vals = range(1, stop=Nx, length=Nx)
            y_vals = range(1, stop=Ny, length=Ny)
    
            # Plot the 3D surface for this timestep
            p = surface(x_vals, y_vals, gathered_data,
                    xlabel="X Location", ylabel="Y Location", zlabel="Value",
                    title="3D Surface at Time Step $step",
                    color=:viridis,
                    xlims=xlims, ylims=ylims, zlims=zlims)
        
            push!(frames,p)
        end
        # NS.VTK(step,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_restart,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz,verts,tets)

    end

    if isroot 
        animation = @animate for frame in frames
            plot!(frame)
        end
    
        # Save the animation as a GIF
        gif(animation, gif_path, fps = fps)
    end
    # parallel_finalize()
end

# @time diffusion_sparse(param,IC!)
# @time diffusion_parallel(param,IC!)
# diffusion_hypre_parallel(param,IC!)
diffusion_hypre_wrap(param,IC!)
# @time diffusion(param,IC!)




