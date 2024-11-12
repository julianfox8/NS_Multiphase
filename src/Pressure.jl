# using JSON

# Solve Poisson equation: δP form
function pressure_solver!(P,uf,vf,wf,t,dt,band,VF,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,gradx,grady,gradz,verts,tets,outflow,BC!,jacob)
    @unpack pressure_scheme = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # RHS = nothing
    if pressure_scheme == "finite-difference"
        # RHS = @view tmp4[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        RHS = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            # RHS
            RHS[i,j,k]= 1/dt* ( 
                ( uf[i+1,j,k] - uf[i,j,k] )/(dx) +
                ( vf[i,j+1,k] - vf[i,j,k] )/(dy) +
                ( wf[i,j,k+1] - wf[i,j,k] )/(dz) )
        end
    else
        RHS = nothing
    end
    iter = poisson_solve!(P,RHS,uf,vf,wf,t,gradx,grady,gradz,band,VF,dt,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,verts,tets,outflow,BC!,jacob)

    return iter
end



function poisson_solve!(P,RHS,uf,vf,wf,t,gradx,grady,gradz,band,VF,dt,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,verts,tets,outflow,BC!,jacob)
    @unpack pressureSolver = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    if pressureSolver == "GaussSeidel"
        iter = GaussSeidel!(P,RHS,uf,vf,wf,t,denx,deny,denz,dt,outflow,BC!,param,mesh,par_env)
    elseif pressureSolver == "ConjugateGradient"
        iter = conjgrad!(P,RHS,denx,deny,denz,tmp2,tmp3,tmp4,dt,param,mesh,par_env)
    elseif pressureSolver == "FC_hypre"
        iter = FC_hypre_solver(P,RHS,denx,deny,denz,tmp4,param,mesh,par_env,jacob)
    elseif pressureSolver == "Secant"
        iter = Secant_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,outflow,param,mesh,par_env)
    elseif pressureSolver == "sparseSecant"
        iter = Secant_sparse_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,outflow,param,mesh,par_env,J,nstep)
    elseif pressureSolver == "hypreSecant"
        iter = Secant_jacobian_hypre!(P,uf,vf,wf,t,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,verts,tets,outflow,param,mesh,par_env,jacob)
    elseif pressureSolver == "NLsolve"
        iter = computeNLsolve!(P,uf,vf,wf,gradx,grady,gradz,band,den,dt,param,mesh,par_env)
    elseif pressureSolver == "Jacobi"
        Pois = Poisson(P,uf,vf,wf,denx,deny,denz,band,dt,param,par_env,mesh)
        iter = Jacobi!(Pois)
     else
        error("Unknown pressure solver $pressureSolver")
    end

    return iter
end


function lap!(L,P,denx,deny,denz,param,mesh)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    fill!(L,0.0)
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        L[i,j,k] = (
            (P[i+1,j,k]-P[i,j,k])/̂(denx[i+1,j,k]*dx^2)-(P[i,j,k]-P[i-1,j,k])/̂(denx[i,j,k]*dx^2) +
            (P[i,j+1,k]-P[i,j,k])/̂(deny[i,j+1,k]*dy^2)-(P[i,j,k]-P[i,j-1,k])/̂(deny[i,j,k]*dy^2) +
            (P[i,j,k+1]-P[i,j,k])/̂(denz[i,j,k+1]*dz^2)-(P[i,j,k]-P[i,j,k-1])/̂(denz[i,j,k]*dz^2) )
    end
    return nothing
end

# LHS of pressure poisson equation
function A!(LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,p,tets_arr,param,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack isroot,irank,nproc = par_env
    fill!(gradx,0.0)
    fill!(grady,0.0)
    fill!(gradz,0.0)

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env) # (overwrites BCs if periodic)

    @loop param for kk=kmin_-1:kmax_+1, jj=jmin_-1:jmax_+1, ii=imin_-1:imax_+2
        gradx[ii,jj,kk]=uf[ii,jj,kk]-dt/̂denx[ii,jj,kk]*̂(P[ii,jj,kk]-P[ii-1,jj,kk])/̂dx
    end

    @loop param for kk=kmin_-1:kmax_+1, jj=jmin_-1:jmax_+2, ii=imin_-1:imax_+1
        grady[ii,jj,kk]=vf[ii,jj,kk]-dt/̂deny[ii,jj,kk]*̂(P[ii,jj,kk]-P[ii,jj-1,kk])/̂dy
    end

    @loop param for kk=kmin_-1:kmax_+2, jj=jmin_-1:jmax_+1, ii=imin_-1:imax_+1
        gradz[ii,jj,kk]=wf[ii,jj,kk]-dt/̂denz[ii,jj,kk]*̂(P[ii,jj,kk]-P[ii,jj,kk-1])/̂dz
    end
    
    fill!(LHS,0.0)
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        LHS[i,j,k] = divg_cell(i,j,k,gradx,grady,gradz,band,dt,p,tets_arr,param,mesh)
    end
    return nothing
end


# Local A! matrix
function A!(i,j,k,LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,p,tets_arr,param,mesh,par_env)
    @unpack Nx,Ny,Nz,dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack isroot = par_env

    
    @loop param for kk = max(k - 1, 0):min(k + 1, Nz+1), jj = max(j - 1, 0):min(j + 1, Ny+1), ii = max(i - 1, 0):min(i + 2, Nx+2)
        gradx[ii,jj,kk]=uf[ii,jj,kk]-dt/̂denx[ii,jj,kk]*̂(P[ii,jj,kk]-P[ii-1,jj,kk])/̂dx
    end

    @loop param for kk = max(k - 1, 0):min(k + 1, Nz+1), jj = max(j - 1, 0):min(j + 2, Ny+2), ii = max(i - 1, 0):min(i + 1, Nx+1)
        grady[ii,jj,kk]=vf[ii,jj,kk] - dt/̂deny[ii,jj,kk]*̂(P[ii,jj,kk]-P[ii,jj-1,kk])/̂dy
    end

    @loop param for kk = max(k - 1, 0):min(k + 2, Nz+2), jj = max(j - 1, 0):min(j + 1, Ny+1), ii = max(i - 1, 0):min(i + 1, Nx+1)
        gradz[ii,jj,kk]=wf[ii,jj,kk] -dt/̂denz[ii,jj,kk]*̂(P[ii,jj,kk]-P[ii,jj,kk-1])/̂dz
    end

    LHS[i,j,k] = divg_cell(i,j,k,gradx,grady,gradz,band,dt,p,tets_arr,param,mesh)

    return nothing
end

function hyp_solve(solver_ref,precond_ref,parcsr_J, par_AP_old, par_P_new,par_env, solver_tag)
    @unpack comm = par_env
    if solver_tag == "PCG"
        # #! PCG
        # Create solver
        HYPRE_ParCSRPCGCreate(par_env.comm, solver_ref)
        solver = solver_ref[]

        # Set some parameters (See Reference Manual for more parameters)
        HYPRE_PCGSetMaxIter(solver, 100) # max iterations
        HYPRE_PCGSetTol(solver, 1e-7) # conv. tolerance
        HYPRE_PCGSetTwoNorm(solver, 1) # use the two norm as the stopping criteria
        # HYPRE_PCGSetRelChange(solver,0)
        # HYPRE_PCGSetPrintLevel(solver, 2) # prints out the iteration info
        # HYPRE_PCGSetLogging(solver, 1) # needed to get run info later
        HYPRE_ParCSRPCGSetup(solver, parcsr_J, par_AP_old, par_P_new)
        HYPRE_ParCSRPCGSolve(solver, parcsr_J, par_AP_old, par_P_new)
        HYPRE_ParCSRPCGDestroy(solver)
    elseif solver_tag == "AMG"    
        #! AMG
        # # Create solver
        HYPRE_BoomerAMGCreate(solver_ref)
        solver = solver_ref[]

        # Set some parameters (See Reference Manual for more parameters)
        # HYPRE_BoomerAMGSetPrintLevel(solver, 3) # print solve info + parameters
        HYPRE_BoomerAMGSetOldDefault(solver)    # Falgout coarsening with modified classical interpolaiton
        HYPRE_BoomerAMGSetRelaxType(solver, 0)  # G-S/Jacobi hybrid relaxation
        HYPRE_BoomerAMGSetRelaxOrder(solver, 1) # uses C/F relaxation
        HYPRE_BoomerAMGSetNumSweeps(solver, 3)  # Sweeeps on each level
        HYPRE_BoomerAMGSetMaxLevels(solver, 20) # maximum number of levels
        HYPRE_BoomerAMGSetTol(solver, 1e-7)     # conv. tolerance

        # Now setup and solve!
        HYPRE_BoomerAMGSetup(solver,  parcsr_J, par_AP_old, par_P_new)
        HYPRE_BoomerAMGSolve(solver,  parcsr_J, par_AP_old, par_P_new)
        HYPRE_BoomerAMGDestroy(solver)
    elseif solver_tag == "PCG-AMG"
        # ! PCG with AMG precond
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
        HYPRE_BoomerAMGSetRelaxType(precond, 3) # Sym G.S./Jacobi hybrid
        HYPRE_BoomerAMGSetNumSweeps(precond, 1)
        HYPRE_BoomerAMGSetTol(precond, 0.0) # conv. tolerance zero
        # HYPRE_BoomerAMGSetTol(precond, 1e-7) # conv. tolerance zero
        HYPRE_BoomerAMGSetMaxIter(precond, 1) # do only one iteration!

        # Set the PCG preconditioner
        HYPRE_PCGSetPrecond(solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond)

        # Now setup and solve!
        HYPRE_ParCSRPCGSetup(solver,parcsr_J, par_AP_old, par_P_new)
        HYPRE_ParCSRPCGSolve(solver, parcsr_J, par_AP_old, par_P_new)
        HYPRE_ParCSRPCGDestroy(solver)
        HYPRE_BoomerAMGDestroy(precond)
    elseif solver_tag == "LGMRES"
        #! LGMRES
        HYPRE_ParCSRLGMRESCreate(comm,solver_ref)
        solver = solver_ref[]

        HYPRE_LGMRESSetKDim(solver,20)
        HYPRE_LGMRESSetTol(solver, 1e-7) # conv. tolerance
        HYPRE_LGMRESSetMaxIter(solver,1000)
        # HYPRE_LGMRESSetPrintLevel(solver, 2) # print solve info
        HYPRE_LGMRESSetLogging(solver, 1) # needed to get run info later

        # Now set up the AMG preconditioner and specify any parameters
        HYPRE_BoomerAMGCreate(precond_ref)
        precond = precond_ref[]
        # HYPRE_BoomerAMGSetPrintLevel(precond, 1) # print amg solution info
        HYPRE_BoomerAMGSetCoarsenType(precond, 0)
        # HYPRE_BoomerAMGSetInterpType(precond, 17)
        HYPRE_BoomerAMGSetOldDefault(precond)
        HYPRE_BoomerAMGSetRelaxType(precond, 6) # Sym G.S./Jacobi hybrid
        HYPRE_BoomerAMGSetNumSweeps(precond, 2)
        HYPRE_BoomerAMGSetTol(precond, 0.0) # conv. tolerance zero
        HYPRE_BoomerAMGSetMaxIter(precond, 1) # do only one iteration!

        # # Set the FlexGMRES preconditioner
        HYPRE_LGMRESSetPrecond(solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond)

        # # Now setup and solve!
        HYPRE_ParCSRLGMRESSetup(solver,parcsr_J, par_AP_old, par_P_new)
        HYPRE_ParCSRLGMRESSolve(solver,parcsr_J, par_AP_old, par_P_new)
        num_iter = Ref{HYPRE_Int}(C_NULL)
        HYPRE_ParCSRLGMRESGetNumIterations(solver, num_iter)

        # HYPRE_ParCSRLGMRESGetNumIterations(solver)
        HYPRE_ParCSRLGMRESDestroy(solver)
        HYPRE_BoomerAMGDestroy(precond)
        return num_iter[]
    elseif solver_tag == "GMRES-AMG"
    # !GMRES
    # # # Create solver
        
        HYPRE_ParCSRFlexGMRESCreate(par_env.comm, solver_ref)
        solver = solver_ref[]

        # Set some parameters (See Reference Manual for more parameters)
        HYPRE_FlexGMRESSetKDim(solver,20) # restart
        HYPRE_FlexGMRESSetMaxIter(solver, 1000) # max iterations
        HYPRE_FlexGMRESSetTol(solver, 1e-7) # conv. tolerance
        # HYPRE_FlexGMRESSetPrintLevel(solver, 2) # print solve info
        HYPRE_FlexGMRESSetLogging(solver, 1) # needed to get run info later

        # Now set up the AMG preconditioner and specify any parameters
        HYPRE_BoomerAMGCreate(precond_ref)
        precond = precond_ref[]
        # HYPRE_BoomerAMGSetPrintLevel(precond, 1) # print amg solution info
        HYPRE_BoomerAMGSetCoarsenType(precond, 0)
        HYPRE_BoomerAMGSetInterpType(precond, 12)
        HYPRE_BoomerAMGSetOldDefault(precond)
        HYPRE_BoomerAMGSetRelaxType(precond, 3) # Sym G.S./Jacobi hybrid
        HYPRE_BoomerAMGSetNumSweeps(precond, 2)
        HYPRE_BoomerAMGSetTol(precond, 0.0) # conv. tolerance zero
        HYPRE_BoomerAMGSetMaxIter(precond, 1) # do only one iteration!

        # Set the FlexGMRES preconditioner
        HYPRE_FlexGMRESSetPrecond(solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond)

        # Now setup and solve!
        HYPRE_ParCSRFlexGMRESSetup(solver,parcsr_J, par_AP_old, par_P_new)
        HYPRE_ParCSRFlexGMRESSolve(solver,parcsr_J, par_AP_old, par_P_new)
        num_iter = Ref{HYPRE_Int}(C_NULL)
        HYPRE_ParCSRFlexGMRESGetNumIterations(solver, num_iter)
        HYPRE_ParCSRFlexGMRESDestroy(solver)
        HYPRE_BoomerAMGDestroy(precond)
        return num_iter[]
    end
        # HYPRE_ParCSRPCGDestroy(solver)
        # HYPRE_BoomerAMGDestroy(precond)
        # HYPRE_ParCSRFlexGMRESDestroy(solver)
        # HYPRE_ParCSRLGMRESDestroy(solver)
        
        # HYPRE_BoomerAMGDestroy(solver)

end


# Semi-Lagrangian pressure solvers

function compute_hypre_jacobian!(matrix,coeff_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS1,tmp4,p,tets_arr,par_env,mesh)
    @unpack  imin, imax, jmin,jmax,kmin,kmax,imin_, imax_, jmin_, jmax_,jmino_,imino_,jmaxo_,imaxo_,kmin_,kmax_,kmino_,kmaxo_,Nx,Nz,Ny = mesh
    
    delta = 1
    nrows = 1
    for k = kmin_:kmax_, j = jmin_:jmax_,i = imin_:imax_
        #define jacobian
        fill!(cols_,0)
        fill!(values_,0.0)
        nst = 0
        
        for kk = k-1:k+1 ,jj = j-1:j+1, ii = i-1:i+1
            if jj < jmin || jj > jmax || ii < imin || ii > imax || kk < kmin || kk > kmax
                continue
            else
                nst += 1
                add_perturb!(P,delta,ii,jj,kk,mesh,par_env)
                cols_[nst] = coeff_index[ii,jj,kk]
                A!(i,j,k,LHS1,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,p,tets_arr,param,mesh,par_env)
                values_[nst] = ((LHS1[i,j,k]
                - AP[i,j,k])
                /̂delta)
                remove_perturb!(P,delta,ii,jj,kk,mesh,par_env)
            end
        end

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
        rows_ = coeff_index[i,j,k]

        # Call function to set matrix values
        HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
    end
end

function Secant_jacobian_hypre!(P,uf,vf,wf,t,gradx,grady,gradz,band,dt,denx,deny,denz,LHS,AP,p_index,tmp4,P_k,AP_k,verts,tets,outflow,param,mesh,par_env,jacob)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env

    # HYPRE.Init()
    fill!(LHS,0.0)
    fill!(AP,0.0)
    fill!(p_index,0.0)

    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
    res_par = parallel_max_all(abs.(AP),par_env)
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = OffsetArray{Int32}(undef,1:27); fill!(cols_,0)
    values_ = OffsetArray{Float64}(undef,1:27); fill!(values_,0.0)
    
    compute_hypre_jacobian!(jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
    MPI.Barrier(comm)
    HYPRE_IJMatrixAssemble(jacob)


    parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
    parcsr_J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])

    # #! prepare Pressure vectors (P_old and P_new)
    AP_ref = Ref{HYPRE_IJVector}(C_NULL)
    HYPRE_IJVectorCreate(par_env.comm,p_min,p_max,AP_ref)
    AP_old = AP_ref[]
    HYPRE_IJVectorSetObjectType(AP_old,HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(AP_old)

    Pn_ref = Ref{HYPRE_IJVector}(C_NULL)
    HYPRE_IJVectorCreate(par_env.comm,p_min,p_max,Pn_ref)
    P_new = Pn_ref[]
    HYPRE_IJVectorSetObjectType(P_new, HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(P_new)

    for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
        row_ = p_index[i,j,k]
        HYPRE_IJVectorSetValues(P_new,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
        # HYPRE_IJVectorSetValues(P_new,1,pointer(Int32.([row_])),pointer(Float64.([P[i,j,k]])))
        HYPRE_IJVectorSetValues(AP_old, 1, pointer(Int32.([row_])), pointer(Float64.([AP[i,j,k]])))
    end

    MPI.Barrier(par_env.comm)


    HYPRE_IJVectorAssemble(AP_old)
    par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJVectorGetObject(AP_old, par_AP_ref)
    par_AP_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])

    HYPRE_IJVectorAssemble(P_new)
    par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJVectorGetObject(P_new, par_Pn_ref)
    par_P_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])


    # # #! create old Pressure and A(P) arrays
    copyto!(P_k,P)
    copyto!(AP_k,AP)

    # Iterate 
    iter=0
    while true
        iter += 1

        # if iter > 10
        # # if iter > 20 
        #     HYPRE_IJMatrixInitialize(jacob)
        #     compute_hypre_jacobian!(jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,p,tets_arr,par_env,mesh)

        #     HYPRE_IJMatrixAssemble(jacob)
        # end
        
        # #! reinit

        if iter > 1
            for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
                row_ = p_index[i,j,k]
                # HYPRE_IJVectorSetValues(P_new,1,pointer(Int32.([row_])),pointer(Float64.([P[i,j,k]])))
                HYPRE_IJVectorSetValues(P_new,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
                HYPRE_IJVectorSetValues(AP_old, 1, pointer(Int32.([row_])), pointer(Float64.([AP[i,j,k]])))
            end

            MPI.Barrier(par_env.comm)


            HYPRE_IJVectorAssemble(AP_old)
            par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
            HYPRE_IJVectorGetObject(AP_old, par_AP_ref)
            par_AP_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])

            HYPRE_IJVectorAssemble(P_new)
            par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
            HYPRE_IJVectorGetObject(P_new, par_Pn_ref)
            par_P_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])

        end

        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)
        MPI.Barrier(par_env.comm)

        hyp_iter = hyp_solve(solver_ref,precond_ref, parcsr_J, par_AP_old, par_P_new,par_env, "GMRES-AMG")

        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            int_x = zeros(1)
            HYPRE_IJVectorGetValues(P_new,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            # P_step[i,j,k] = int_x[1]
            if iter> 10
                P[i,j,k] -= 0.5*int_x[1]
            else
                P[i,j,k] -= int_x[1]
            end
        end

        MPI.Barrier(par_env.comm)

        MPI.Barrier(par_env.comm)
        P .-=parallel_mean_all(P,par_env)

        MPI.Barrier(par_env.comm)
        # if isroot; println("outflow gradient calculated within p iterations at iter ", iter); end
        outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,verts,tets,param,mesh,par_env)
        # if isroot; println("a op gradient calc within p iter at iter ", iter); end

        MPI.Barrier(par_env.comm)
        
        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)

        #! interpolate between the k and k+1 iterations
        # if t >109 && iter>1
        #     for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        #         P_temp = P_k[i,j,k] - AP_k[i,j,k]*(P[i,j,k]-P_k[i,j,k])/(AP[i,j,k]-AP_k[i,j,k])
        #         P_temp = min(max(P_k[i,j,k],P[i,j,k]),P_temp)
        #         P[i,j,k] = max(min(P_k[i,j,k],P[i,j,k]),P_temp)
        #     end
        #     outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,p,tets_arr,param,mesh,par_env)
        #     A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,p,tets_arr,param,mesh,par_env)
        # end

        # println("this is iter $iter")
        # outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,p,tets_arr,param,mesh,par_env)
        copyto!(P_k,P)
        copyto!(AP_k,AP)
        res_par = parallel_max_all(abs.(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
        
        if res_par < tol
            HYPRE_ParVectorDestroy(par_AP_old)
            HYPRE_ParVectorDestroy(par_P_new)
            return iter
        end
        # if iter >1 && t>109
        if iter % 10 == 0
        # if t == 6
            # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum(AP))
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env))
            # J = compute_sparse2D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,tmp2,tmp3,tmp4,mesh,par_env)
        end
    end    
end


function S_jacobian_hypre!(P,uf,vf,wf,t,gradx,grady,gradz,band,dt,denx,deny,denz,LHS,AP,p_index,tmp4,outflow,param,mesh,par_env,jacob)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env

    # HYPRE.Init()
    fill!(LHS,0.0)
    fill!(AP,0.0)
    fill!(p_index,0.0)
    AP_2 = OffsetArray{Float64}(undef,imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env)

    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)

    #!prep indices
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    MPI.Barrier(comm)

    cols_ = OffsetArray{Int32}(undef,1:27); fill!(cols_,0)
    values_ = OffsetArray{Float64}(undef,1:27); fill!(values_,0.0)
    
    tets_arr = Array{Float64}(undef, 3, 4, 5)
    p = Matrix{Float64}(undef, (3, 8))
    
    # if t==1 || t % 10==0
        compute_hypre_jacobian!(jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,p,tets_arr,par_env,mesh)
        MPI.Barrier(comm)
        HYPRE_IJMatrixAssemble(jacob)

    # end
    parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
    parcsr_J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])

    # #! prepare Pressure vectors (P_old and P_new)
    AP_ref = Ref{HYPRE_IJVector}(C_NULL)
    HYPRE_IJVectorCreate(par_env.comm,p_min,p_max,AP_ref)
    AP_old = AP_ref[]
    HYPRE_IJVectorSetObjectType(AP_old,HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(AP_old)

    Pn_ref = Ref{HYPRE_IJVector}(C_NULL)
    HYPRE_IJVectorCreate(par_env.comm,p_min,p_max,Pn_ref)
    P_new = Pn_ref[]
    HYPRE_IJVectorSetObjectType(P_new, HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(P_new)


    # Iterate 
    iter=0
    while true
        iter += 1

        # # # #! recompute the jacobian 
        # if iter % 10 == 0
        # # if iter >1
        #     HYPRE_IJMatrixInitialize(jacob)
        #     compute_hypre_jacobian!(jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,p,tets_arr,par_env,mesh)

        #     HYPRE_IJMatrixAssemble(jacob)
        # end
        
        #! reinitialize after iter 1
        if iter>1
            HYPRE_IJVectorInitialize(AP_old)
            HYPRE_IJVectorInitialize(P_new)
        end

        
        for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
            row_ = p_index[i,j,k]
            HYPRE_IJVectorSetValues(P_new,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            HYPRE_IJVectorSetValues(AP_old, 1, pointer(Int32.([row_])), pointer(Float64.([AP[i,j,k]])))
        end

        MPI.Barrier(par_env.comm)


        HYPRE_IJVectorAssemble(AP_old)
        par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(AP_old, par_AP_ref)
        par_AP_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])

        HYPRE_IJVectorAssemble(P_new)
        par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(P_new, par_Pn_ref)
        par_P_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])

        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)
        MPI.Barrier(par_env.comm)

        hyp_iter = hyp_solve(solver_ref,precond_ref, parcsr_J, par_AP_old, par_P_new,par_env, "LGMRES")
        
        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            int_x = zeros(1)
            HYPRE_IJVectorGetValues(P_new,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            P[i,j,k] -= int_x[1]
        end

        MPI.Barrier(par_env.comm)


        P .-=mean(P)

        MPI.Barrier(par_env.comm)
        P .-=parallel_mean_all(P,par_env)
        
        outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env)

        #update new Ap
        A!(AP_2,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)
        
        
        HYPRE_IJVectorInitialize(AP_old)
        HYPRE_IJVectorInitialize(P_new)
        

        
        for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
            row_ = p_index[i,j,k]
            HYPRE_IJVectorSetValues(P_new,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            HYPRE_IJVectorSetValues(AP_old, 1, pointer(Int32.([row_])), pointer(Float64.([AP_2[i,j,k]])))
        end

        MPI.Barrier(par_env.comm)


        HYPRE_IJVectorAssemble(AP_old)
        par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(AP_old, par_AP_ref)
        par_AP_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])

        HYPRE_IJVectorAssemble(P_new)
        par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(P_new, par_Pn_ref)
        par_P_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])

        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)
        MPI.Barrier(par_env.comm)

        hyp_iter = hyp_solve(solver_ref,precond_ref, parcsr_J, par_AP_old, par_P_new,par_env, "LGMRES")
        P_step = OffsetArray{Float64}(undef,imin_:imax_,jmin_:jmax_,kmin_:kmax_)
        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            int_x = zeros(1)
            HYPRE_IJVectorGetValues(P_new,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            P_step[i,j,k] = int_x[1]
            # P[i,j,k] -= int_x[1]
        end

        P .-= dot(P_step, (AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]./(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-2AP_2[imin_:imax_,jmin_:jmax_,kmin_:kmax_])))

        P .-=mean(P)

        MPI.Barrier(par_env.comm)
        P .-=parallel_mean_all(P,par_env)
        
        outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env)

        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)
        
        

        res_par = parallel_max_all(abs.(AP),par_env)
        @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,parallel_sum_all(AP,par_env))
        MPI.Barrier(par_env.comm)
        
        if res_par < tol
            HYPRE_ParVectorDestroy(par_AP_old)
            HYPRE_ParVectorDestroy(par_P_new)
            return iter
        end
        if iter % 1000 == 0 
            # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum(AP))
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,parallel_sum_all(AP,par_env))
            # J = compute_sparse2D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,tmp2,tmp3,tmp4,mesh,par_env)
        end
        # if iter == 400
        #     error("stop")
        # end
    end    
end

# Flux-Corrected solvers 
#! laplace operator containing face centered densities
function compute_lap_op!(matrix,coeff_index,cols_,values_,denx,deny,denz,par_env,mesh)
    @unpack  imin, imax, jmin,jmax,kmin,kmax,kmin_,kmax_,imin_, imax_, jmin_, jmax_,dx,dy,dz,Nx,Nz,Ny = mesh
    nrows = 1
    for k = kmin_:kmax_, j = jmin_:jmax_,i = imin_:imax_
        #define jacobian
        fill!(cols_,0)
        fill!(values_,0.0)
        nst = 0
        #! main diagonal
        nst+=1
        cols_[nst] = coeff_index[i,j,k]
        values_[nst] = -(1.0 / (denx[i, j, k] * dx^2) + 1.0 / (denx[i+1, j, k] * dx^2) + 
                        1.0 / (deny[i, j, k] * dy^2) + 1.0 / (deny[i, j+1, k] * dy^2) + 
                        1.0 / (denz[i, j, k] * dz^2) + 1.0 / (denz[i, j, k+1] * dz^2))

        #! x-dir off-diags
        if i-1 < imin
            values_[1] += 1.0/(denx[i,j,k]*dx^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i-1,j,k]
            values_[nst] = 1.0 / (denx[i,j,k] * dx^2)
        end

        if i+1 > imax
            values_[1] += 1.0/(denx[i+1,j,k]*dx^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i+1,j,k]
            values_[nst] = 1.0 / (denx[i+1,j,k] * dx^2)
        end

        #! y-dir off-diags
        if j-1 < jmin
            values_[1] += 1.0/(deny[i,j,k]*dy^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i,j-1,k]
            values_[nst] = 1.0 / (deny[i,j,k] * dy^2)
        end

        if j+1 > jmax
            values_[1] += 1.0/(deny[i,j+1,k]*dy^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i,j+1,k]
            values_[nst] = 1.0 / (deny[i,j+1,k] * dy^2)
        end

        #! z-dir off-diags
        if k-1 < kmin
            values_[1] += 1.0/(denz[i,j,k]*dz^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i,j,k-1]
            values_[nst] = 1.0 / (denz[i,j,k] * dz^2)
        end

        if k+1 > kmax
            values_[1] += 1.0/(denz[i,j,k+1]*dz^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i,j,k+1]
            values_[nst] = 1.0 / (denz[i,j,k+1] * dz^2)
        end

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
        rows_ = coeff_index[i,j,k]


        # Call function to set matrix values
        HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
    end
end

#! laplace operator containing face centered densities multiplied by negative 1
function compute_lap_op_neg!(matrix,coeff_index,cols_,values_,denx,deny,denz,par_env,mesh)
    @unpack  imin, imax, jmin,jmax,kmin,kmax,kmin_,kmax_,imin_, imax_, jmin_, jmax_,dx,dy,dz,Nx,Nz,Ny = mesh
    nrows = 1
    for k = kmin_:kmax_, j = jmin_:jmax_,i = imin_:imax_
        #define jacobian
        fill!(cols_,0)
        fill!(values_,0.0)
        nst = 0
        #! main diagonal
        nst+=1
        cols_[nst] = coeff_index[i,j,k]
        values_[nst] = (1.0 / (denx[i, j, k] * dx^2) + 1.0 / (denx[i+1, j, k] * dx^2) + 
                        1.0 / (deny[i, j, k] * dy^2) + 1.0 / (deny[i, j+1, k] * dy^2) + 
                        1.0 / (denz[i, j, k] * dz^2) + 1.0 / (denz[i, j, k+1] * dz^2))

        #! x-dir off-diags
        if i-1 < imin
            values_[1] -= 1.0/(denx[i,j,k]*dx^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i-1,j,k]
            values_[nst] = -1.0 / (denx[i,j,k] * dx^2)
        end

        if i+1 > imax
            values_[1] -= 1.0/(denx[i+1,j,k]*dx^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i+1,j,k]
            values_[nst] = -1.0 / (denx[i+1,j,k] * dx^2)
        end

        #! y-dir off-diags
        if j-1 < jmin
            values_[1] -= 1.0/(deny[i,j,k]*dy^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i,j-1,k]
            values_[nst] = -1.0 / (deny[i,j,k] * dy^2)
        end

        if j+1 > jmax
            values_[1] -= 1.0/(deny[i,j+1,k]*dy^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i,j+1,k]
            values_[nst] = -1.0 / (deny[i,j+1,k] * dy^2)
        end

        #! z-dir off-diags
        if k-1 < kmin
            values_[1] -= 1.0/(denz[i,j,k]*dz^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i,j,k-1]
            values_[nst] = -1.0 / (denz[i,j,k] * dz^2)
        end

        if k+1 > kmax
            values_[1] -= 1.0/(denz[i,j,k+1]*dz^2)
        else 
            nst+=1
            cols_[nst] = coeff_index[i,j,k+1]
            values_[nst] = -1.0 / (denz[i,j,k+1] * dz^2)
        end

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
        rows_ = coeff_index[i,j,k]
        # if i > imin && j > jmin
        # println(1/(dx^2*denx[i,j,k]))
        # println(1/(dy^2*deny[i,j,k]))
        #     println(cols_)
        #     println(values_)
        #     error("stop")
        # end

        # Call function to set matrix values
        HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
    end
end

#! laplace operator containing face centered densities with pressure reference point
function compute_lap_op_pref!(matrix,coeff_index,cols_,values_,denx,deny,denz,par_env,mesh)
    @unpack  imin, imax, jmin,jmax,kmin,kmax,kmin_,kmax_,imin_, imax_, jmin_, jmax_,dx,dy,dz,Nx,Nz,Ny = mesh
    nrows = 1
    for k = kmin_:kmax_, j = jmin_:jmax_,i = imin_:imax_
        #define jacobian
        fill!(cols_,0)
        fill!(values_,0.0)
        nst = 0
        if i == 15 && j == 110 && k==15
            nst+=1
            cols_[nst] = coeff_index[i,j,k]
            values_[1] = 1
        else
            #! main diagonal
            nst+=1
            cols_[nst] = coeff_index[i,j,k]
            values_[nst] = -(1.0 / (denx[i, j, k] * dx^2) + 1.0 / (denx[i+1, j, k] * dx^2) + 
                            1.0 / (deny[i, j, k] * dy^2) + 1.0 / (deny[i, j+1, k] * dy^2) + 
                            1.0 / (denz[i, j, k] * dz^2) + 1.0 / (denz[i, j, k+1] * dz^2))

            #! x-dir off-diags
            if i-1 < imin
                values_[1] += 1.0/(denx[i,j,k]*dx^2)
            else 
                nst+=1
                cols_[nst] = coeff_index[i-1,j,k]
                values_[nst] = 1.0 / (denx[i,j,k] * dx^2)
            end

            if i+1 > imax
                values_[1] += 1.0/(denx[i+1,j,k]*dx^2)
            else 
                nst+=1
                cols_[nst] = coeff_index[i+1,j,k]
                values_[nst] = 1.0 / (denx[i+1,j,k] * dx^2)
            end

            #! y-dir off-diags
            if j-1 < jmin
                values_[1] += 1.0/(deny[i,j,k]*dy^2)
            else 
                nst+=1
                cols_[nst] = coeff_index[i,j-1,k]
                values_[nst] = 1.0 / (deny[i,j,k] * dy^2)
            end

            if j+1 > jmax
                values_[1] += 1.0/(deny[i,j+1,k]*dy^2)
            else 
                nst+=1
                cols_[nst] = coeff_index[i,j+1,k]
                values_[nst] = 1.0 / (deny[i,j+1,k] * dy^2)
            end

            #! z-dir off-diags
            if k-1 < kmin
                values_[1] += 1.0/(denz[i,j,k]*dz^2)
            else 
                nst+=1
                cols_[nst] = coeff_index[i,j,k-1]
                values_[nst] = 1.0 / (denz[i,j,k] * dz^2)
            end

            if k+1 > kmax
                values_[1] += 1.0/(denz[i,j,k+1]*dz^2)
            else 
                nst+=1
                cols_[nst] = coeff_index[i,j,k+1]
                values_[nst] = 1.0 / (denz[i,j,k+1] * dz^2)
            end
        end
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
        rows_ = coeff_index[i,j,k]
        # if i > imin && j > jmin
        # println(1/(dx^2*denx[i,j,k]))
        # println(1/(dy^2*deny[i,j,k]))
        #     println(cols_)
        #     println(values_)
        #     error("stop")
        # end

        # Call function to set matrix values
        HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
    end
end

function FC_hypre_solver(P,RHS,denx,deny,denz,p_index,param,mesh,par_env,jacob)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env
    

    #! prep indices
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = OffsetArray{Int32}(undef,1:27); fill!(cols_,0)
    values_ = OffsetArray{Float64}(undef,1:27); fill!(values_,0.0)


    #! determine the laplacian to use
    compute_lap_op!(jacob,p_index,cols_,values_,denx,deny,denz,par_env,mesh)
    #compute_lap_op_pref!(jacob,p_index,cols_,values_,denx,deny,denz,par_env,mesh)
    # compute_lap_op_neg!(jacob,p_index,cols_,values_,denx,deny,denz,par_env,mesh)

    MPI.Barrier(comm)
    HYPRE_IJMatrixAssemble(jacob)
    parcsr_A_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_A_ref)
    parcsr_A = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_A_ref[])

    #! prepare Pressure vectors (P_old and P_new)
    RHS_ref = Ref{HYPRE_IJVector}(C_NULL)
    HYPRE_IJVectorCreate(par_env.comm,p_min,p_max,RHS_ref)
    RHS_hyp = RHS_ref[]
    HYPRE_IJVectorSetObjectType(RHS_hyp,HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(RHS_hyp)

    Pn_ref = Ref{HYPRE_IJVector}(C_NULL)
    HYPRE_IJVectorCreate(par_env.comm,p_min,p_max,Pn_ref)
    P_new = Pn_ref[]
    HYPRE_IJVectorSetObjectType(P_new, HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(P_new)


    for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
        row_ = p_index[i,j,k]
        HYPRE_IJVectorSetValues(P_new,1,pointer(Int32.([row_])),pointer(Float64.([P[i,j,k]])))
        HYPRE_IJVectorSetValues(RHS_hyp, 1, pointer(Int32.([row_])), pointer(Float64.([RHS[i,j,k]])))
    end
    
    #! if pressure reference point is used set here

    # row_ = p_index[15,110,15]
    # HYPRE_IJVectorSetValues(RHS_hyp, 1, pointer(Int32.([row_])), pointer(Float64.([0.0])))
    MPI.Barrier(par_env.comm)


    HYPRE_IJVectorAssemble(P_new)
    par_P_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJVectorGetObject(P_new, par_P_ref)
    par_P = convert(Ptr{HYPRE_ParVector}, par_P_ref[])

    HYPRE_IJVectorAssemble(RHS_hyp)
    par_RHS_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJVectorGetObject(RHS_hyp, par_RHS_ref)
    par_RHS = convert(Ptr{HYPRE_ParVector}, par_RHS_ref[])

    solver_ref = Ref{HYPRE_Solver}(C_NULL)
    precond_ref = Ref{HYPRE_Solver}(C_NULL)
    MPI.Barrier(par_env.comm)

    # iter = hyp_solve(solver_ref,precond_ref, parcsr_A, par_RHS, par_P ,par_env, "LGMRES")
    iter = hyp_solve(solver_ref,precond_ref, parcsr_A, par_RHS, par_P ,par_env, "GMRES-AMG")

    for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
        int_x = zeros(1)
        HYPRE_IJVectorGetValues(P_new,1,pointer(Int32.([p_index[i,j,k]])),int_x)
        P[i,j,k] = int_x[1]
    end

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env)
    return iter
end


function BiCGSTAB!(P,RHS,denx,deny,denz,r,p,v,t1,param,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack tol =  param
    @unpack isroot = par_env

    # Interior indices
    ix = imin_:imax_; iy = jmin_:jmax_;  iz = kmin_:kmax_
    # # Ghost cell indices
    gx = imino_:imaxo_; gy = jmino_:jmaxo_; gz = kmino_:kmaxo_    

    # r = @view r[gx,gy,gz]
    # p = @view p[gx,gy,gz]
    # v = @view v[gx,gy,gz]
    # t = @view t[gx,gy,gz]
    t = OffsetArray{Float64}(undef, gx,gy,gz)
    fill!(r,0.0)
    fill!(v,0.0)
    fill!(p,0.0)
    fill!(t,0.0)

    lap!(r,P,denx,deny,denz,param,mesh)
    r[ix,iy,iz] = RHS[ix,iy,iz] - r[ix,iy,iz] 

    Neumann!(r,mesh,par_env)
    update_borders!(r,mesh,par_env)
    
    omega = 1.0
    alpha = 1.0

    r_tld = r
    beta = 1.0
    phi_1 = sum(r_tld.*r)
    rnew = 0.0

    for i = 1:length(P)
        phi = sum(r_tld.*r)
        if i ==1 
            p = r
        else
            beta = (phi/phi_1)*(alpha/omega)
            p = r .+ beta*( p -omega*v)
        end
        lap!(v,p,denx,deny,denz,param,mesh)
        println("rold",sum(v.^2))
        alpha = phi / sum(r_tld.*v)
        
        s = r .- alpha * v
        lap!(t,s,denx,deny,denz,param,mesh)

        omega = sum(t.*s)/sum(t.*t)
        P = P .+ alpha*p .+ omega*s
        r = s .- omega*t
        rnew = sum(r.^2)
        println(rnew)
        if i == 5
            error("stop")
        end
        Neumann!(r,mesh,par_env)
        # Neumann!(t,mesh,par_env)
        # Neumann!(v,mesh,par_env)
        if sqrt(rnew) < tol
           return i
        end
        phi_1 = phi
        # r_tld = rnew
        
        
    end
    
    isroot && println("Failed to converged Poisson equation rsnew = $rnew")
    
    return length(RHS)
end

"""
GaussSeidel Poisson Solver
"""
function GaussSeidel!(P,RHS,uf,vf,wf,t,denx,deny,denz,dt,outflow,BC!,param,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack isroot = par_env
    @unpack tol = param
    maxIter=1000000
    iter = 0
    # Interior indices
    # ix = imin_:imax_; iy = jmin_:jmax_;  iz = kmin_:kmax_
    # # Ghost cell indices
    # gx = imino_:imaxo_; gy = jmino_:jmaxo_; gz = kmino_:kmaxo_
    # r  = OffsetArray{Float64}(undef, gx,gy,gz)
    # lap!(r,P,denx,deny,denz,param,mesh)
    # r[ix,iy,iz] = RHS[ix,iy,iz] - r[ix,iy,iz]
    # println(maximum(RHS))
    FD_outflowCorrection!(P,RHS,denx,deny,denz,uf,vf,wf,dt,outflow,param,mesh,par_env)
    # println(maximum(RHS))
    # error("stop")
    while true
        iter += 1
        max_update::Float64 = 0.0
        for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            Pnew = (-denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*RHS[i,j,k]*dx^2*dy^2*dz^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*dx^2*dy^2*P[i,j,k+1] + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k+1]*dx^2*dy^2*P[i,j,k-1] + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2*P[i,j+1,k] + denx[i,j,k]*denx[i+1,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2*P[i,j-1,k] + denx[i,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2*P[i+1,j,k] + denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2*P[i-1,j,k])/(denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*dx^2*dy^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k+1]*dx^2*dy^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2 + denx[i,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2 + denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2)
            max_update=max(max_update,abs(Pnew-P[i,j,k]))
            P[i,j,k] = Pnew 
        end
        # FD_outflowCorrection!(P,RHS,denx,deny,denz,uf,vf,wf,dt,outflow,param,mesh,par_env)

        update_borders!(P,mesh,par_env)
        Neumann!(P,mesh,par_env)
 
        max_update = parallel_max_all(max_update,par_env)
        # println(max_update)

        max_update < tol && return iter # Converged
        # Check if hit max iteration
        if iter == maxIter 
            isroot && println("Failed to converged Poisson equation max_upate = $max_update")
            return iter
        end
    end
end

"""
Conjugate gradient
"""
function conjgrad!(P,RHS,denx,deny,denz,r,p,Ap,dt,param,mesh,par_env)
    @unpack dx,dy,dz = mesh
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack tol = param
    @unpack irank,isroot = par_env
    maxIter = 1000000
    # Interior indices
    ix = imin_:imax_; iy = jmin_:jmax_;  iz = kmin_:kmax_

    fill!(r,0.0)
    fill!(p,0.0)
    fill!(Ap,0.0)
    # FD_outflowCorrection!(P,RHS,denx,deny,denz,uf,vf,wf,dt,outflow,param,mesh,par_env)
    lap!(r,P,denx,deny,denz,param,mesh)
    r[ix,iy,iz] = RHS[ix,iy,iz] - r[ix,iy,iz]
    Neumann!(r,mesh,par_env)
    update_borders!(r,mesh,par_env) # (overwrites BCs if periodic)
    p .= r
    rsold = parallel_sum_all(r[ix,iy,iz].^2,par_env)
    rsnew = 0.0
    for iter = 1:maxIter
        lap!(Ap,p,denx,deny,denz,param,mesh)

        sum = parallel_sum_all(p[ix,iy,iz].*Ap[ix,iy,iz],par_env)

        alpha = rsold /̂ sum

        P .+= alpha*p
        r[ix,iy,iz] .-= alpha * Ap[ix,iy,iz]
        
        rsnew = parallel_sum_all(r[ix,iy,iz].^2,par_env)

        if sqrt(rsnew) < tol
            return iter
        end
        p = r + (rsnew /̂ rsold) * p
        # FD_outflowCorrection!(P,RHS,denx,deny,denz,uf,vf,wf,dt,outflow,param,mesh,par_env)
        Neumann!(p,mesh,par_env)   
        update_borders!(p,mesh,par_env) # (overwrites BCs if periodic)
        rsold = rsnew

    end
    
    isroot && println("Failed to converged Poisson equation rsnew = $rsnew")
    
    return length(RHS)
end

#! non-parallelized semi-Lagrangian solvers

function computeJacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
    @unpack Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    J = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    dp = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    LHS1 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    LHS2 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    delta = 1.0

    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        fill!(LHS1,0.0)
        fill!(LHS2,0.0)
        fill!(dp,0.0)
        dp[i,j,k] += delta
        J[i,j,k] = (
            (A!(i,j,k,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)
            - A!(i,j,k,LHS2,uf,vf,wf,P.-dp,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env))
            ./̂2delta)
    end
    return J 
end


function n(i,j,k,Nx,Ny) 
    val = i + (j-1)*Nx + (k-1)*Nx*Ny
    # @show i,j,k,Ny,Nz,val
    return val
end 

function compute_sparse_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
    @unpack Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    J = OffsetArray{Float64}(undef,1:Nx*Ny*Nz,1:Nx*Ny*Nz)
    dp = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    LHS1 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    LHS2 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    delta = 1.0
    fill!(J,0.0)


    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        for kk=max(1,k-1):min(Nz,j+1), jj=max(1,j-1):min(Ny,j+1), ii=max(1,i-1):min(Nx,i+1)
            fill!(LHS1,0.0)
            fill!(LHS2,0.0)
            fill!(dp,0.0)
            dp[ii,jj,kk] += delta
            J[n(i,j,k,Nx,Ny),n(ii,jj,kk,Nx,Ny)] = (
                (A!(i,j,k,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)
                - A!(i,j,k,LHS2,uf,vf,wf,P.-dp,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env))
                ./̂2delta)
        end
    end
    return J 
end

function compute_sparse2D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,tmp2,tmp3,tmp4,mesh,par_env)
    @unpack Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    diags = OffsetArray{Float64}(undef,1:Nx*Ny*Nz,9)
    LHS1 = @view tmp3[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
    # LHS2 = @view tmp4[imin_:imax_,jmin_:jmax_,kmin_:kmax_]

    delta = 1.0
    fill!(diags,0.0)
    offset=[-(Nx+1), -Nx, -(Nx-1), -1, 0, 1, Nx-1, Nx, Nx+1]


    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        nNeigh = 0

        for kk = k ,jj = j-1:j+1, ii = i-1:i+1
            if jj < 1 || jj > Nx || ii < 1 || ii > Nx
                # Outside domain 
                nNeigh += 1
            elseif kk < 1 || kk > Nz
                nNeigh += 0
            else
                fill!(LHS1,0.0)
                P[ii,jj,kk] += delta
                J = ((A!(i,j,k,LHS1,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)
                    - AP[i,j,k])
                    ./̂delta)
                P[ii,jj,kk] -= delta
                nNeigh +=1 
                row = n(ii,jj,kk,Nx,Ny)-max(0,offset[nNeigh]) # Row in diagonal array
                diags[row,nNeigh] = J
            end
        end
    end

    J = spdiagm(
        offset[1] => diags[1:Nx*Ny*Nz-abs(offset[1]),1],
        offset[2] => diags[1:Nx*Ny*Nz-abs(offset[2]),2],
        offset[3] => diags[1:Nx*Ny*Nz-abs(offset[3]),3],
        offset[4] => diags[1:Nx*Ny*Nz-abs(offset[4]),4],
        offset[5] => diags[1:Nx*Ny*Nz-abs(offset[5]),5],
        offset[6] => diags[1:Nx*Ny*Nz-abs(offset[6]),6],
        offset[7] => diags[1:Nx*Ny*Nz-abs(offset[7]),7],
        offset[8] => diags[1:Nx*Ny*Nz-abs(offset[8]),8],
        offset[9] => diags[1:Nx*Ny*Nz-abs(offset[9]),9]
    )
    return J 
end


function compute_sparse3D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
    @unpack Nx,Ny,Nz = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    diags = OffsetArray{Float64}(undef,1:Nx*Ny*Nz,27)
    dp = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    LHS1 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    LHS2 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    # #initial calculation of gradientsa
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        gradx[i,j,k]=dt/denx[i,j,k]*(P[i,j,k]-P[i-1,j,k])/̂dx
    end
    
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        grady[i,j,k]=dt/deny[i,j,k]*(P[i,j,k]-P[i,j-1,k])/̂dy
    end

    @loop param for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        gradz[i,j,k]=dt/denz[i,j,k]*(P[i,j,k]-P[i,j,k-1])/̂dz
    end

    delta = 1.0
    fill!(diags,0.0)
    offset=[-Nx*Ny-Nx-1,-Nx*Ny-Nx,-Nx*Ny-Nx+1,-Nx*Ny-1,-Nx*Ny,-Nx*Ny+1,-Nx*Ny+Nx-1,
            -Nx*Ny+Nx,-Nx*Ny+Nx+1,-(Nx+1), -Nx, -(Nx-1), -1, 0, 1, Nx-1, Nx, Nx+1,
            Nx*Ny-Nx-1, Nx*Ny-Nx, Nx*Ny-Nx+1, Nx*Ny-1, Nx*Ny, Nx*Ny+1,Nx*Ny+Nx-1,
            Nx*Ny+Nx,Nx*Ny+Nx+1]


    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        nNeigh = 0

        
        for kk = k-1:k+1 ,jj = j-1:j+1, ii = i-1:i+1
            if jj < 1 || jj > Nx || ii < 1 || ii > Nx || kk < 1 || kk > Nz
                # Outside domain 
                nNeigh += 1
            else
                fill!(LHS1,0.0)
                fill!(LHS2,0.0)
                fill!(dp,0.0)
                dp[ii,jj,kk] += delta
                J = ((A!(i,j,k,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)
                    - AP[i,j,k])
                    ./̂delta)
                nNeigh +=1 
                row = n(ii,jj,kk,Ny,Nz)-max(0,offset[nNeigh]) # Row in diagonal array
                # println(n(ii,jj,kk,Ny,Nz))
                # println(max(0,offset[nNeigh])) 
                # @show i,j,ii,jj,nNeigh, row,n(ii,jj,kk,Ny,Nz),max(0,offset[nNeigh])
                #? can we store this as a sparse vector as well?
                diags[row,nNeigh] = J
            end
        end
    end
    J = spdiagm(
        offset[1] => diags[1:Nx*Ny*Nz-abs(offset[1]),1],
        offset[2] => diags[1:Nx*Ny*Nz-abs(offset[2]),2],
        offset[3] => diags[1:Nx*Ny*Nz-abs(offset[3]),3],
        offset[4] => diags[1:Nx*Ny*Nz-abs(offset[4]),4],
        offset[5] => diags[1:Nx*Ny*Nz-abs(offset[5]),5],
        offset[6] => diags[1:Nx*Ny*Nz-abs(offset[6]),6],
        offset[7] => diags[1:Nx*Ny*Nz-abs(offset[7]),7],
        offset[8] => diags[1:Nx*Ny*Nz-abs(offset[8]),8],
        offset[9] => diags[1:Nx*Ny*Nz-abs(offset[9]),9],
        offset[10] => diags[1:Nx*Ny*Nz-abs(offset[10]),10],
        offset[11] => diags[1:Nx*Ny*Nz-abs(offset[11]),11],
        offset[12] => diags[1:Nx*Ny*Nz-abs(offset[12]),12],
        offset[13] => diags[1:Nx*Ny*Nz-abs(offset[13]),13],
        offset[14] => diags[1:Nx*Ny*Nz-abs(offset[14]),14],
        offset[15] => diags[1:Nx*Ny*Nz-abs(offset[15]),15],
        offset[16] => diags[1:Nx*Ny*Nz-abs(offset[16]),16],
        offset[17] => diags[1:Nx*Ny*Nz-abs(offset[17]),17],
        offset[18] => diags[1:Nx*Ny*Nz-abs(offset[18]),18],
        offset[19] => diags[1:Nx*Ny*Nz-abs(offset[19]),19],
        offset[20] => diags[1:Nx*Ny*Nz-abs(offset[20]),20],
        offset[21] => diags[1:Nx*Ny*Nz-abs(offset[21]),21],
        offset[22] => diags[1:Nx*Ny*Nz-abs(offset[22]),22],
        offset[23] => diags[1:Nx*Ny*Nz-abs(offset[23]),23],
        offset[24] => diags[1:Nx*Ny*Nz-abs(offset[24]),24],
        offset[25] => diags[1:Nx*Ny*Nz-abs(offset[25]),25],
        offset[26] => diags[1:Nx*Ny*Nz-abs(offset[26]),26],
        offset[27] => diags[1:Nx*Ny*Nz-abs(offset[27]),27]
    )
    return J
 
end

function convert3d_1d(matrix)
    m1D = reshape(matrix,size(matrix,1)*size(matrix,2)*size(matrix,3),1)
    return vec(m1D)
end

function convert1d_3d(matrix,x,y,z)
    m3D = reshape(matrix,(x,y,z))
    return m3D
end

# Secant method
function Secant_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,outflow,param,mesh,par_env)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh

    AP = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    fill!(AP,0.0)
    outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env)

    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)
    J = computeJacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
    # Iterate 
    iter=0
    while true
        iter += 1

        # compute jacobian
        # J = computeJacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
        P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .-= 0.8AP./̂J

        P .-=mean(P)

        #Need to introduce outflow correction
        outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env)
        
        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)
        
        res = maximum(abs.(AP))
        if res < tol
            return iter
        end
        
        if iter % 1000 == 0
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res,sum(AP))
        end
    end    
end

function Secant_sparse_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,outflow,param,mesh,par_env,J,nstep)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env

    AP = @view tmp1[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
    # AP = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    fill!(AP,0.0)
    delP = OffsetArray{Float64}(undef, imax_*jmax_*kmax_,1); fill!(delP,0.0)
    outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env)

    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)
    if isnothing(J)
        J = compute_sparse2D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,tmp2,tmp3,tmp4,mesh,par_env)
        # J = compute_sparse_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
    end
    # J = compute_sparse2D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,tmp2,tmp3,tmp4,mesh,par_env)
    # Iterate 
    iter=0
    while true
        iter += 1

        if iter % 10 == 0
            # J0 = copy(J)
            # compute jacobian
        # J = compute_sparse2D_Jacobia n(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,tmp2,tmp3,tmp4,mesh,par_env)
            J = compute_sparse2D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,tmp2,tmp3,tmp4,mesh,par_env)
            # println(maximum(abs.(J-J0)))
        end


        # P0 = copy(convert3d_1d(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]))
        Pv = convert3d_1d(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
        APv = convert3d_1d(AP)
        Pstep = J\APv
        Pv -= Pstep
        P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = convert1d_3d(Pv,size(imin_:imax_)[1],size(jmin_:jmax_)[1],size(kmin_:kmax_)[1])

        P .-=mean(P)

        # println(P)
        #Need to introduce outflow correction
        outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env)
        # Pv = convert3d_1d(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_])

        #! potentially delete
        # delP = Pv .- P0
        # AP = convert3d_1d(AP)
        #calc J update
        # println(J)
        # println((APv - J\delP).*AP/AP.^2)
        # if nstep % 500 == 0 
        #     J += (APv - J*delP).*AP/AP.^2
        # end
        # error("stop")
        # P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = convert1d_3d(Pv,Nx/2,Ny/2,Nz)
        # AP = convert1d_3d(AP,Nx,Ny,Nz) 


        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)


        res = parallel_max_all(AP,par_env)

        if res < tol
            return iter
        end

        if iter == 100
            error("stop")
        end
        # if iter % 10 == 0
        #     @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res,sum(AP))
        #     # J = compute_sparse2D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,tmp2,tmp3,tmp4,mesh,par_env)
        # end
    end    
end

# NLsolve Library
function computeNLsolve!(P,uf,vf,wf,gradx,grady,gradz,band,den,dt,param,mesh,par_env)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Ghost cell indices
    gx = imino_:imaxo_; gy = jmino_:jmaxo_; gz = kmino_:kmaxo_

    LHS = OffsetArray{Float64}(undef, gx,gy,gz)


    function f!(LHS, P)
        A!(LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,den,mesh,par_env)
        # println("maxRes=",maximum(abs.(F)))
        return LHS
    end

    # Run solver
    out = nlsolve(f!,P,
        ftol = tol,
        method = :trust_region, #default 
        # method = :newton,
        # method = :anderson, # diverges
        # m=20,
        )

    # Get output
    P .= out.zero

    return out.iterations
end




struct Poisson{T<:AbstractArray,T16<:AbstractArray,parameters,par_env_struct,mesh_struct}
    p :: T # Pressure
    f :: Tuple{T,T,T} # flow (can we store three objects of type T here or do we need individual face velocity arrays)
    den :: Tuple{T,T,T}
    band :: T16
    step:: Float64
    z :: T # source
    j :: T # Jacobian
    r :: T # Jacobian residual
    n :: Int16 # iterations
    param :: parameters # param object
    par :: par_env_struct # parallel environement structure
    mesh :: mesh_struct # mesh structure
    function Poisson(p::T,uf::T,vf::T,wf::T,denx::T,deny::T,denz::T,_band::T16,dt::Float64,param::parameters,par::par_env_struct,mesh::mesh_struct) where {T,T16,parameters,par_env_struct,mesh_struct} #? is this where argument redundant
        #need to initilize all types 
        f = (uf,vf,wf)
        den = (denx,deny,denz)
        step,band = dt,_band
        #want to compute grad(u*-dt/rho) in each direction and store it in f 
        r = similar(p); fill!(r,0.0)
        z,j = copy(r),copy(r)
        n = Int16(0)
        new{T,T16,typeof(param),typeof(par),typeof(mesh)}(p,f,den,band,step,z,j,r,n,param,par,mesh)
    end
end


#want to define step and del operators
CI(a...) = CartesianIndex(a...)
step(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
step(i,I::CartesianIndex{N}) where N = step(i, Val{N}())

# partial derivative of scalar field
@inline del(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-step(a,I)]
# partial derivative of vector field
@inline del(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I,a]-u[I-step(a,I),a]
#might want to define a del function in the a_th direction that takes the correpsonding flow field, density and dt
@inline del(a,I::CartesianIndex{m},u::AbstractArray{T,n}, f::AbstractArray{T,d},den::AbstractArray{T,n},dt::Float64) where {T,n,m,d} = @inbounds u[I]-dt/̂den[I]*̂del(a,I,f)


@inline inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))

# non local A matrix for Poisson struct
function A!(P::Poisson{T}) where {T}
    param,mesh = P.param,P.mesh
    @unpack dx,dy,dz = mesh
    d_step = dx,dy,dz

    Neumann!(P.p,mesh,P.par)
    update_borders!(P.p,mesh,P.par) # (overwrites BCs if periodic)

    
    v_fields = (copy(P.f[1]), copy(P.f[2]), copy(P.f[3]))
    for ii in 1:length(P.f)
        for I in inside(P.f[ii])
            v_fields[ii][I] -= P.step/P.den[ii][I]*del(ii,I,P.p)/d_step[ii]
        end
    end
    # println("made it through")

    for I in inside(P.band)
        if abs(P.band[I]) <= 1
            tets, inds = cell2tets_withProject_uvwf(I[1],I[2],I[3],v_fields[1],v_fields[2],v_fields[3],P.step,mesh)
            if any(isnan,tets)
                error("Nan in tets at ", I)
            end
            v2 = dx*dy*dz
            v1 = tets_vol(tets)
            P.z[I] = (v2-v1) /̂ v2 /̂ P.step
        else 
            # Calculate divergence with finite differnce
            du_dx = (v_fields[1][I+step(1,I)]-v_fields[1][I])/(dx)
            dv_dy = (v_fields[2][I+step(2,I)]-v_fields[2][I])/(dy)
            dw_dz = (v_fields[3][I+step(3,I)]-v_fields[3][I])/(dz)
            P.z[I] = du_dx + dv_dy + dw_dz
        end
    end
end


# local A matrix that recieves Poisson struct and cartesian index
# A! matrix for the jacobian calculation at P+/-dP acts on Jacobian residual field
function A!(P::Poisson{T}, I::CartesianIndex{d},delta) where {T,d}
    param,mesh,par_env = P.param,P.mesh,P.par
    @unpack dx,dy,dz = mesh
    d_step = dx,dy,dz

    v_fields = (copy(P.f[1]), copy(P.f[2]), copy(P.f[3]))
    p = copy(P.p)
    p[I] += delta
    Neumann!(p,mesh,par_env)
    update_borders!(p,mesh,par_env) # (overwrites BCs if periodic)

    for ii in 1:length(v_fields)
        for Ii in inside(P.f[ii])
            v_fields[ii][Ii] -= P.step/P.den[ii][Ii]*del(ii,Ii,p)/d_step[ii]
        end
    end

    if abs(P.band[I]) <= 1
        tets, inds = cell2tets_withProject_uvwf(I[1],I[2],I[3],v_fields[1],v_fields[2],v_fields[3],P.step,mesh)
        if any(isnan,tets)
            error("Nan in tets at ", I)
        end
        v2 = dx*dy*dz
        v1 = tets_vol(tets)
        P.r[I] = (v2-v1) /̂ v2 /̂ P.step 
    else 
        # Calculate divergence with finite differnce
        du_dx = (v_fields[1][I+step(1,I)]-v_fields[1][I])/(dx)
        dv_dy = (v_fields[2][I+step(2,I)]-v_fields[2][I])/(dy)
        dw_dz = (v_fields[3][I+step(3,I)]-v_fields[3][I])/(dz)
        P.r[I] = du_dx + dv_dy + dw_dz
    end
end

function Jacobian!(P::Poisson)
    delta = 1.0
    ndelta = -1.0
    #! need to loop over p-field
    for I in inside(P.p)
        A!(P,I,delta)
        A_pos = P.r[I]
        fill!(P.r,0.0)
        A!(P,I,ndelta)
        A_neg = P.r[I]
        #! calc jacobian at each pt in mesh
        P.j[I] = (A_pos-A_neg)/̂2delta        
    end
end

function full_Jacobian!(P::Poisson)
    delta = 1.0
    ndelta = -1.0

    #! need to loop over p-field
    for I in inside(P.p)
        A!(P,I,delta)
        A_pos = P.r[I]
        fill!(P.r,0.0)
        A!(P,I,ndelta)
        A_neg = P.r[I]
        #! calc jacobian at each pt in mesh
        P.j[I] = (A_pos-A_neg)/̂2delta        
    end
end


function Jacobi!(P::Poisson,tol=1e-4)
    p,z,j,n = P.p,P.z,P.j,P.n
    #!apply outflow correction to P.p calc A! using P.z

    # calc A(P) as p.z
    A!(P)

    n = 0
    while true 
        n +=1
        # calc jacobian
        Jacobian!(P)

        for I in inside(p)
            P.p[I] -= 0.8z[I]/j[I]
        end

        # avoid drift
        p .-=mean(p)

        #!add outlfow correction to P.p

        # calc A(P)
        fill!(z,0.0)
        A!(P)

        # check residual to tol for new A(P)
        res = maximum(abs.(z))
     
        if res < tol
            return n
        end
        if n % 1000 == 0
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",n,res,sum(z))
        end  
    end
end

function guided_Jacobi!(P::Poisson,tol=1e-4)
    p,z,e,n = P.p,P.z,P.e,P.n
    # calc A(P) as p.z
    A!(P)

    n = 0
    alpha = similar(p)
    fill!(alpha,0.0)
    while true 
        n +=1
        # calc jacobian
        Jacobian!(P)

        #! compute alpha at each point in the domain

        for I in inside(p)
            alpha[I] = arm_gold(P,I)
        end
        # println(alpha)
        for I in inside(p)
            P.p[I] -= alpha[I]*z[I]/e[I]
        end

        # avoid drift
        p .-=mean(p)

        # calc A(P)
        fill!(z,0.0)
        A!(P)

        # check residual to tol for new A(P)
        res = maximum(abs.(z))
     
        if res < tol
            return n
        end
        if n % 500 == 0
            println(alpha)
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",n,res,sum(z))
        end  
    end
end

#! need to think about how to construct the domain wide line search for the armijo-goldstein condition
function arm_gold(P::Poisson, I::CartesianIndex,del = 0.5, c=1000)
    f,x,d = P.z,P.p,P.e
    g = copy(d)
    f1 = copy(f)
    t = 1.0
    delta = -t*d
    A!(P,I,delta[I])
    while f[I] < f1[I] + c*t*g[I]^2
        t *= del
        delta = t*d
        A!(P,I,delta[I])

    end
    return t
end