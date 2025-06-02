using JSON


# Solve Poisson equation: δP form
function pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,gradx,grady,gradz,verts,tets,BC!,jacob,b,x)
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
    iter = poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,verts,tets,jacob,b,x)

    return iter
end

function poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,verts,tets,jacob,b,x)
    @unpack pressureSolver = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    if pressureSolver == "FC_hypre"
        iter = FC_hypre_solver(P,RHS,denx,deny,denz,tmp4,param,mesh,par_env,jacob,b,x)
    elseif pressureSolver == "hypreSecant"
        iter = Secant_jacobian_hypre!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,verts,tets,param,mesh,par_env,jacob,b,x)
    elseif pressureSolver == "hypreSecantLS"
        iter = Secant_jacobian_hypre_linesearch!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,verts,tets,param,mesh,par_env,jacob,b,x)
    elseif pressureSolver == "Ostrowski"
        iter = Ostrowski(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,verts,tets,param,mesh,par_env,jacob,b,x)
    elseif pressureSolver == "SOR"
        iter = SOR(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,verts,tets,param,mesh,par_env,jacob,b,x)
    elseif pressureSolver == "SecantSOR"
        iter = Secant_SOR(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,verts,tets,param,mesh,par_env,jacob,b,x)
    elseif pressureSolver == "res_iteration"
        iter = res_iteration(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,verts,tets,param,mesh,par_env) 
    else
        error("Unknown pressure solver $pressureSolver")
    end

    return iter
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
        grady[ii,jj,kk]=vf[ii,jj,kk] - dt/̂deny[ii,jj,kk]*̂(P[ii,jj,kk]-P[ii,jj-1,kk])/̂dy
    end

    @loop param for kk=kmin_-1:kmax_+2, jj=jmin_-1:jmax_+1, ii=imin_-1:imax_+1
        gradz[ii,jj,kk]=wf[ii,jj,kk] -dt/̂denz[ii,jj,kk]*̂(P[ii,jj,kk]-P[ii,jj,kk-1])/̂dz
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
        # HYPRE_LGMRESSetTol(solver, 1e-9) # conv. tolerance
        HYPRE_LGMRESSetAbsoluteTol(solver,1e-9)
        HYPRE_LGMRESSetMaxIter(solver,1000)
        # HYPRE_LGMRESSetPrintLevel(solver, 2) # print solve info
        HYPRE_LGMRESSetLogging(solver, 1) # needed to get run info later

        # Now set up the AMG preconditioner and specify any parameters
        HYPRE_BoomerAMGCreate(precond_ref)
        precond = precond_ref[]
        # HYPRE_BoomerAMGSetPrintLevel(precond, 1) # print amg solution info
        HYPRE_BoomerAMGSetCoarsenType(precond, 0)
        HYPRE_BoomerAMGSetInterpType(precond, 6)
        HYPRE_BoomerAMGSetOldDefault(precond)
        # HYPRE_BoomerAMGSetRelaxType(precond, 3) # Sym G.S./Jacobi hybrid
        HYPRE_BoomerAMGSetNumSweeps(precond, 2)
        HYPRE_BoomerAMGSetTol(precond, 0.0) # conv. tolerance zero
        HYPRE_BoomerAMGSetMaxIter(precond, 1) # do only one iteration!

        # Set the FlexGMRES preconditioner
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
        
        HYPRE_ParCSRGMRESCreate(par_env.comm, solver_ref)
        solver = solver_ref[]

        # Set some parameters (See Reference Manual for more parameters)
        HYPRE_GMRESSetKDim(solver,20) # restart
        HYPRE_GMRESSetMaxIter(solver, 1000) # max iterations
        HYPRE_GMRESSetTol(solver, 1e-9) # conv. tolerance
        # HYPRE_FlexGMRESSetPrintLevel(solver, 2) # print solve info
        HYPRE_GMRESSetLogging(solver, 1) # needed to get run info later

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
        HYPRE_GMRESSetPrecond(solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond)

        # Now setup and solve!
        HYPRE_ParCSRGMRESSetup(solver,parcsr_J, par_AP_old, par_P_new)
        HYPRE_ParCSRGMRESSolve(solver,parcsr_J, par_AP_old, par_P_new)
        num_iter = Ref{HYPRE_Int}(C_NULL)
        HYPRE_ParCSRGMRESGetNumIterations(solver, num_iter)
        HYPRE_ParCSRGMRESDestroy(solver)
        HYPRE_BoomerAMGDestroy(precond)
        return num_iter[]
    end
end


# Semi-Lagrangian pressure solvers
function compute_hypre_jacobian!(dynamic_dP,jacobi_iter,matrix,coeff_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS1,LHS2,p,tets_arr,par_env,mesh)
    @unpack  imin, imax, jmin,jmax,kmin,kmax,imin_, imax_, jmin_, jmax_,jmino_,imino_,jmaxo_,imaxo_,kmin_,kmax_,kmino_,kmaxo_,Nx,Nz,Ny = mesh
    if dynamic_dP
        if isodd(jacobi_iter)
            delta = 1*10.0^(-ceil(jacobi_iter/2))
        elseif iseven(jacobi_iter)
            delta = 1*10.0^(jacobi_iter/2+1)
        end
    else 
        delta = 5
    end
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
                remove_perturb!(P,delta,ii,jj,kk,mesh,par_env)
                add_perturb!(P,-delta,ii,jj,kk,mesh,par_env)
                cols_[nst] = coeff_index[ii,jj,kk]
                A!(i,j,k,LHS2,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,p,tets_arr,param,mesh,par_env)
                remove_perturb!(P,-delta,ii,jj,kk,mesh,par_env)
                values_[nst] = ((LHS1[i,j,k]
                - LHS2[i,j,k])
                /̂2delta)
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
        # HYPRE.assemble!(mat_assembler,[rows_], cols_[1:ncols],values_[:,1:ncols])
        HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
    end
end

function line_search(P_k,AP_k,uf,vf,wf,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,JdP,jacob,x,mesh,param,par_env)
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # Compute J⋅dP for line search
    fill!(JdP,0.0)
    ncols_arr = zeros(Int32,1)
    for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
        # Get non-zero columns on this row 
        row = p_index[i,j,k]
        HYPRE_IJMatrixGetRowCounts(jacob, 1, pointer(Int32.([row])), ncols_arr)
        ncols = ncols_arr[1]
        cols = zeros(Int32,  ncols)
        Jvals = zeros(Float64,ncols)
        dPvals = zeros(Float64,ncols)
        # Note returns cols (with 0 index) and vals
        HYPRE_IJMatrixGetValues(jacob, -1, ncols_arr,pointer(Int32.([row])),cols,Jvals)
        cols .+= 1 # shift column index to be 1 based
        HYPRE_IJVectorGetValues(x,ncols,cols,dPvals)
        JdP[i,j,k] = dot(Jvals,dPvals)
    end


    # Line search to find largest damping parameter
    c=0.01
    λ = 1
    mydP = zeros(1)
    line_iter = 0
    AP_mag    = mag(  AP[imin:imax,jmin:jmax,kmin:kmax],par_env)
    JdP_mag   = mag( JdP[imin:imax,jmin:jmax,kmin:kmax],par_env)
    while true
        line_iter += 1
        P_k .+= λ*dP
        A!(AP_k,uf,vf,wf,P_k,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        APnew_mag = mag(AP_k[imin:imax,jmin:jmax,kmin:kmax],par_env)
        # Check if λ is small enough
        if APnew_mag <= AP_mag - c*λ*JdP_mag
            # println("lambda = ",λ)
            break
        end
        # Reduce λ
        λ /= 2
        # Max number of iter
        if line_iter == 4
            println("Iter=$iter: Reached $line_iter subiterations on line search λ=$λ")
            break
        end
    end
end

function recompute_jacobian(res_par,res_par_old,jacobi_iter,dynamic_dP,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP_k,LHS,tmp4,verts,tets,par_env,mesh)
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    if abs(res_par_old - res_par) > 1e2  #|| iter % 20 == 0
        println("res diff= ",abs(res_par_old - res_par))
        @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum_res)
        # iter += 1
        jacobi_iter += 1
        println("number of jacobian computes = $jacobi_iter")
        dynamic_dP = true
        HYPRE_IJMatrixInitialize(jacob)
        compute_hypre_jacobian!(dynamic_dP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP_k,LHS,tmp4,verts,tets,par_env,mesh)
        HYPRE_IJMatrixAssemble(jacob)
        parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
        dynamic_dP = false
    else
        P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .=P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        res_par_old = parallel_max_all(abs.(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
        sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        jacobi_iter = 1
        iter+=1
        @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par_old,sum_res)
    end

    return jacobi_iter
end

function Secant_jacobian_hypre_linesearch!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,LHS,AP,p_index,tmp4,P_k,AP_k,dP,JdP,verts,tets,param,mesh,par_env,jacob,b,x)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env

    # HYPRE.Init()
    fill!(LHS,0.0)
    fill!(AP,0.0)
    fill!(p_index,0.0)
    
    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)

    res_par_old = parallel_max_all(abs.(AP),par_env)
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = Vector{Int32}(undef,27); fill!(cols_,0)
    values_ = Matrix{Float64}(undef,1,27); fill!(values_,0.0)
    
    iter=0
    jacobi_iter = 1 
    dynamic_dP = false
    HYPRE_IJMatrixInitialize(jacob)
    compute_hypre_jacobian!(dynamic_dP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
    HYPRE_IJMatrixAssemble(jacob)

    parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
    J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])

    # Iterate
    iter += 1
        
    while true
        P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .= P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        
        HYPRE_IJVectorInitialize(b)
        HYPRE_IJVectorInitialize(x)
        # if iter > 1
        for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
            row_ = p_index[i,j,k]
            HYPRE_IJVectorSetValues(x,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            HYPRE_IJVectorSetValues(b, 1, pointer(Int32.([row_])), pointer(Float64.([AP[i,j,k]])))
        end
        MPI.Barrier(par_env.comm)
        HYPRE_IJVectorAssemble(b)
        par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(b, par_AP_ref)
        par_b_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])
        HYPRE_IJVectorAssemble(x)
        par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(x, par_Pn_ref)
        par_x_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])


        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)

        hyp_iter = hyp_solve(solver_ref,precond_ref, J, par_b_old, par_x_new, par_env, "LGMRES")
        int_x = zeros(1)
        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            HYPRE_IJVectorGetValues(x,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            # P_k[i,j,k] -= int_x[1]
            dP[i,j,k] = -int_x[1]
        end
        # println("computing J dP")
        # Compute J⋅dP for line search
        update_borders!(dP,mesh,par_env)
        fill!(JdP,0.0)
        ncols_arr = zeros(Int32,1)
        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            # Get non-zero columns on this row 
            row = p_index[i,j,k]
            HYPRE_IJMatrixGetRowCounts(jacob, 1, pointer(Int32.([row])), ncols_arr)
            ncols = ncols_arr[1]
            cols = zeros(Int32,  ncols)
            Jvals = zeros(Float64,ncols)
            dPvals = zeros(Float64,ncols)
            # Note returns cols (with 0 index) and vals
            HYPRE_IJMatrixGetValues(jacob, -1, ncols_arr,pointer(Int32.([row])),cols,Jvals)
            cols .+= 1 # shift column index to be 1 based
            HYPRE_IJVectorGetValues(x,ncols,cols,dPvals)
            # dPvals[row,cols] = dP[row,cols]
            # error("stop")
            JdP[i,j,k] = dot(Jvals,dPvals)
        end

        # println("starting line search")
        # Line search to find largest damping parameter
        c=0.01
        λ = 1
        mydP = zeros(1)
        line_iter = 0
        AP_mag    = mag(  AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        JdP_mag   = mag( JdP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        while true
            line_iter += 1
            P_k .+= λ*dP
            A!(AP_k,uf,vf,wf,P_k,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
            APnew_mag = mag(AP_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
            # Check if λ is small enough
            if APnew_mag <= AP_mag - c*λ*JdP_mag
                # println("lambda = ",λ)
                break
            end
            # Reduce λ
            λ /= 2
            # Max number of iter
            if line_iter == 4
                println("Iter=$iter: Reached $line_iter subiterations on line search λ=$λ")
                break
            end
        end
        
        # account for drift
        P_k .-=parallel_mean_all(P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        #update new Ap
        A!(AP_k,uf,vf,wf,P_k,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        # pressure_VTK(iter,P,AP,sfx,sfy,sfz,dir,pvd_pressure,param,mesh,par_env)
        
        res_par = parallel_max_all(abs.(AP_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
        sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        # println("initial residual = $res_par_old")
        # println("new residual residual = $res_par")
        # if res_par_old < res_par
        if res_par_old - res_par < -1e2
            # println(abs(res_par_old - res_par))   
            # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum_res)
            
            println("number of jacobian computes = $jacobi_iter")
            dynamic_dP = true
            HYPRE_IJMatrixInitialize(jacob)
            compute_hypre_jacobian!(dynamic_dP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP_k,LHS,tmp4,verts,tets,par_env,mesh)
            HYPRE_IJMatrixAssemble(jacob)
            jacobi_iter +=1
            parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
            dynamic_dP = false
        else 
            P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .=P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
            res_par_old = parallel_max_all(abs.(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
            sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
            # if jacobi_iter > 0; println("Jacobian recomputed $jacobi_iter times"); end
            # jacobi_iter = 1
            iter += 1
            # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par_old,sum_res)
        end
        if iter == 1000; println("P-solve did not converge to tol");return iter; end

        sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        if res_par_old < tol; return iter; end

    end    
end

function Secant_jacobian_hypre!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,LHS,AP,p_index,tmp4,P_k,AP_k,tmp7,tmp8,verts,tets,param,mesh,par_env,jacob,b,x)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env

    # HYPRE.Init()
    fill!(LHS,0.0)
    fill!(AP,0.0)
    fill!(p_index,0.0)
    
    # pvd_pressure,dir = pVTK_init(param,par_env)
    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)

    res_par_old = parallel_max_all(abs.(AP),par_env)
    # println("Initial res = $res_par_old")
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = Vector{Int32}(undef,27); fill!(cols_,0)
    values_ = Matrix{Float64}(undef,1,27); fill!(values_,0.0)
    
    iter=0
    jacobi_iter = 1 
    # pressure_VTK(iter,P,AP,sfx,sfy,sfz,dir,pvd_pressure,param,mesh,par_env)
    dynamic_dP = false
    max_δP = zeros(1)
    max_δP_k = [res_par_old]
    HYPRE_IJMatrixInitialize(jacob)
    compute_hypre_jacobian!(dynamic_dP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
    HYPRE_IJMatrixAssemble(jacob)

    parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
    J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])

    # Iterate
    iter += 1
        
    while true
        P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .= P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        
        HYPRE_IJVectorInitialize(b)
        HYPRE_IJVectorInitialize(x)
        # if iter > 1
        for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
            row_ = p_index[i,j,k]
            HYPRE_IJVectorSetValues(x,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            HYPRE_IJVectorSetValues(b, 1, pointer(Int32.([row_])), pointer(Float64.([AP[i,j,k]])))
        end
        MPI.Barrier(par_env.comm)
        HYPRE_IJVectorAssemble(b)
        par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(b, par_AP_ref)
        par_b_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])
        HYPRE_IJVectorAssemble(x)
        par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(x, par_Pn_ref)
        par_x_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])


        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)

        hyp_iter = hyp_solve(solver_ref,precond_ref, J, par_b_old, par_x_new, par_env, "LGMRES")

        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            int_x = zeros(1)
            HYPRE_IJVectorGetValues(x,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            P_k[i,j,k] -= int_x[1]
            if abs(maximum(int_x[1])) > max_δP[1]
                max_δP[1] = int_x[1]
            end
        end

        # account for drift
        P_k .-=parallel_mean_all(P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)

        #update new Ap
        A!(AP_k,uf,vf,wf,P_k,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        # pressure_VTK(iter,P,AP,sfx,sfy,sfz,dir,pvd_pressure,param,mesh,par_env)
        
        
        res_par = parallel_max_all(abs.(AP_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
        sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        # println("checking residual for potential recompute")
        # if res_par_old < res_par
        if abs(res_par_old - res_par) > 1e2    
            # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum_res)
            
            println("number of jacobian recomputes = $jacobi_iter")
            dynamic_dP = true
            HYPRE_IJMatrixInitialize(jacob)
            compute_hypre_jacobian!(dynamic_dP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP_k,LHS,tmp4,verts,tets,par_env,mesh)
            HYPRE_IJMatrixAssemble(jacob)
            jacobi_iter +=1
            parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
            dynamic_dP = false
        else 
            P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .=P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
            res_par_old = parallel_max_all(abs.(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
            sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
            # if jacobi_iter > 0; println("Jacobian recomputed $jacobi_iter times"); end
            # jacobi_iter = 1
            iter += 1
            # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g max_δP = %12.3g \n",iter,res_par_old,sum_res,max_δP[1])
        end
        
        sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        max_δP_k[1] = max_δP[1]
    
        # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g max_δP = %12.3g \n",iter,res_par,sum_res,max_δP[1])
        # if iter % 10 == 0 ;@printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum_res); end

        if res_par_old < tol; return iter; end

    end    
end
function Secant_jacobian_hypre2!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,LHS,AP,p_index,tmp4,P_k,AP_k,verts,tets,param,mesh,par_env,jacob,b,x_vec)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env

    # HYPRE.Init()
    fill!(LHS,0.0)
    fill!(AP,0.0)
    fill!(P,0.0)
    fill!(p_index,0.0)
    
    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
    pvd_pressure,dir = pVTK_init(param,par_env)
    res_par = parallel_max_all(abs.(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
    sum_res = parallel_sum_all((AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
    # println("inside P solve max div = ", res_par)
    # println("inside P solve sum div = ", sum_res)
    
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = Vector{Int32}(undef,27); fill!(cols_,0)
    values_ = Matrix{Float64}(undef,1,27); fill!(values_,0.0)
    
    dynamic_dP = false
    max_δP = zeros(1)
    jacobi_iter = 1
    HYPRE_IJMatrixInitialize(jacob)
    compute_hypre_jacobian!(dynamic_dP,max_δP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
    HYPRE_IJMatrixAssemble(jacob)

    parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
    J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])

    # Iterate
    iter=0
    # if res_par < tol; return iter; end
    # pressure_VTK(iter,P,AP,sfx,sfy,sfz,dir,pvd_pressure,param,mesh,par_env)
    while true

        iter += 1
        HYPRE_IJVectorInitialize(b)
        HYPRE_IJVectorInitialize(x_vec)
        
        for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
            row_ = p_index[i,j,k]
            HYPRE_IJVectorSetValues(x_vec,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            HYPRE_IJVectorSetValues(b, 1, pointer(Int32.([row_])), pointer(Float64.([AP[i,j,k]])))
        end

        MPI.Barrier(par_env.comm)
        HYPRE_IJVectorAssemble(b)
        par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(b, par_AP_ref)
        par_b_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])
        HYPRE_IJVectorAssemble(x_vec)
        par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(x_vec, par_Pn_ref)
        par_x_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])


        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)

        # hyp_iter = hyp_solve(solver_ref,precond_ref, J, b, x, par_env, "LGMRES")     
        hyp_iter = hyp_solve(solver_ref,precond_ref, J, par_b_old, par_x_new, par_env, "LGMRES")
        # HYPRE_IJVectorPrint(x_vec,"P_sol")
        # error("stop")
        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            int_x = zeros(1)
            HYPRE_IJVectorGetValues(x_vec,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            P[i,j,k] -= int_x[1]
            if abs(maximum(int_x[1])) > max_δP[1]
                # println("max dP of $(int_x[1]) at $i,$j,$k")
                max_δP[1] = int_x[1]
            end
        end

        # account for drift
        P .-=parallel_mean_all(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)

        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        # pressure_VTK(iter,P,AP,sfx,sfy,sfz,dir,pvd_pressure,param,mesh,par_env)
        res_par = parallel_max_all(abs.(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
        sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g max_δP = %12.3g \n",iter,res_par,sum_res,max_δP[1])
        
        # error("stop")
        max_δP = zeros(1)
        if res_par < tol; return iter; end

    end    
end


function nonZero_getter!(p_index,i,j,k,cols,cols_ijk,mesh)
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    nst = 0
    for kk = k-1:k+1 ,jj = j-1:j+1, ii = i-1:i+1
        if jj < jmin || jj > jmax || ii < imin || ii > imax || kk < kmin || kk > kmax
            continue
        else
            nst += 1
            cols_ijk[nst] = (ii,jj,kk)
            cols[nst] = p_index[ii,jj,kk]
        end


        for st in 1:nst
            ind = st + argmin(cols[st:nst], dims=1)[1] - 1
            tmpi = cols[st]
            cols[st] = cols[ind]
            cols[ind] = tmpi
            tmpijk = cols_ijk[st]
            cols_ijk[st] = cols_ijk[ind]
            cols_ijk[ind] = tmpijk
        end
        
    end
end

function SOR(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,LHS,AP,p_index,tmp4,tmp5,tmp6,tmp7,tmp8,verts,tets,param,mesh,par_env,jacob,b,x)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env

    fill!(tmp5,0.0)
    fill!(tmp6,0.0)

    max_iter = 1e4
    ω = 0.5
    # set up work matrices
    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
    res_par = parallel_max_all(abs.(AP),par_env)
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = Vector{Int32}(undef,27); fill!(cols_,0)
    values_ = Matrix{Float64}(undef,1,27); fill!(values_,0.0)

    δP = tmp5
    δP_old = tmp6
    max_δP = zeros(1)
    dynamic_dP = false
    #? compute Jacobian (probably do not need it to be HYPRE matrix)
    HYPRE_IJMatrixInitialize(jacob)
    compute_hypre_jacobian!(dynamic_dP,max_δP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
    HYPRE_IJMatrixAssemble(jacob)

    parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
    J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])
    iter = 0
    j_vec = zeros(Nx*Ny*Nz)
    

    while iter<=max_iter
        iter+=1
        for k = kmin_:kmax_, j = jmin_:jmax_,i = imin_:imax_
            HYPRE_IJMatrixGetValues(jacob,1,pointer(Int32.([Nx*Ny*Nz])),pointer(Int32.([p_index[i,j,k]])),pointer(Int32.(Vector(1:(Nx*Ny*Nz)))),j_vec)
            j_index = Int(p_index[i,j,k])
            δP[i,j,k] = 1/j_vec[j_index]*( -sum(j_vec[1:(Nx*Ny*Nz)].*vec(δP[1:imax_,1:jmax_,1:kmax_])) + j_vec[j_index]*δP[i,j,k] + AP[i,j,k])
            P[i,j,k] -= (δP_old[i,j,k]*(1-ω) + ω*δP[i,j,k])
        end
        
        P .-=parallel_mean_all(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        res_par = parallel_max_all(abs.(AP),par_env)
        sum_res = parallel_sum_all(abs.(AP),par_env)
        δP_old =δP
        if res_par<tol
            return iter
        end
        @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum_res)

    end
end

function Secant_SOR(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,LHS,AP,p_index,tmp4,δP,δP_old,tmp7,tmp8,verts,tets,param,mesh,par_env,jacob,b,x)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env

    # HYPRE.Init()
    fill!(LHS,0.0)
    fill!(AP,0.0)
    fill!(p_index,0.0)
    fill!(δP,0.0)
    fill!(δP_old,0.0)

    # pvd_pressure,dir = pVTK_init(param,par_env)
    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
    
    res_par_old = parallel_max_all(abs.(AP),par_env)
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = Vector{Int32}(undef,27); fill!(cols_,0)
    cols_ijk = Vector{Tuple{Int32,Int32,Int32}}(undef,27); fill!(cols_ijk,(0,0,0))
    values_ = Matrix{Float64}(undef,1,27); fill!(values_,0.0)

    ω = 1.0 
    dynamic_dP = false
    jacobi_iter = 0
    HYPRE_IJMatrixInitialize(jacob)
    compute_hypre_jacobian!(dynamic_dP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
    HYPRE_IJMatrixAssemble(jacob)

    parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
    J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])

    # Iterate
    iter = 0     
    while true
        iter += 1
        HYPRE_IJVectorInitialize(b)
        HYPRE_IJVectorInitialize(x)
        
        for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
            row_ = p_index[i,j,k]
            HYPRE_IJVectorSetValues(x,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            HYPRE_IJVectorSetValues(b, 1, pointer(Int32.([row_])), pointer(Float64.([AP[i,j,k]])))
        end

        MPI.Barrier(par_env.comm)
        HYPRE_IJVectorAssemble(b)
        par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(b, par_AP_ref)
        par_b_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])
        HYPRE_IJVectorAssemble(x)
        par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(x, par_Pn_ref)
        par_x_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])


        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)

        
        hyp_iter = hyp_solve(solver_ref,precond_ref, J, par_b_old, par_x_new, par_env, "LGMRES")
        
        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            int_x = zeros(1)
            HYPRE_IJVectorGetValues(x,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            #! update δP with Newton step 
            δP[i,j,k] = int_x[1]

            # #! Jacobi update using previous Newton step as δP_old
            # fill!(values_,0.0)
            # fill!(cols_,0)
            # fill!(cols_ijk,(0,0,0))

            # nonZero_getter!(p_index,i,j,k,cols_,cols_ijk,mesh)
            # HYPRE_IJMatrixGetValues(jacob,1,pointer(Int32.([27])),pointer(Int32.([p_index[i,j,k]])),pointer(Int32.(cols_)),values_)
            # j_index = Int(p_index[i,j,k])
            # ijk_ind = findfirst(val -> val == j_index, cols_)

            # δP[i,j,k] = 1/values_[ijk_ind]*( -sum(values_*[δP[ind[1],ind[2],ind[3]] for ind in cols_ijk[:]]) + values_[ijk_ind]*δP[i,j,k] + AP[i,j,k])

            # P[i,j,k] -= (δP_old[i,j,k]*(1-ω) + ω*δP[i,j,k])
        end

        for k = kmin_:kmax_, j = jmin_:jmax_,i = imin_:imax_
            fill!(values_,0.0)
            fill!(cols_,0)
            fill!(cols_ijk,(0,0,0))
            #! grab non-zero cols and store in cols_
            nonZero_getter!(p_index,i,j,k,cols_,cols_ijk,mesh)
            #! grab corresponding values from jacobian and store in values_
            HYPRE_IJMatrixGetValues(jacob,1,pointer(Int32.([27])),pointer(Int32.([p_index[i,j,k]])),pointer(Int32.(cols_)),values_)
            j_index = Int(p_index[i,j,k])
            ijk_ind = findfirst(val -> val == j_index, cols_)
            δP[i,j,k] = 1/values_[ijk_ind]*( -sum(values_*[δP[ind[1],ind[2],ind[3]] for ind in cols_ijk[:]]) + values_[ijk_ind]*δP[i,j,k] + AP[i,j,k])    
            P[i,j,k] -= (δP_old[i,j,k]*(1-ω) + ω*δP[i,j,k])
        end

        # account for drift
        P .-=parallel_mean_all(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)

        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)

        res_par = parallel_max_all(abs.(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
        sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
    
        δP_old  = δP
    
        @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum_res)
        # if iter % 10 == 0 ;@printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum_res); end

        if res_par < tol; return iter; end

    end    
end

function Ostrowski(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,LHS,AP,p_index,tmp4,δP,AP2,P_k,AP_k,verts,tets,param,mesh,par_env,jacob,b,x)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env
    
    fill!(LHS,0.0)
    fill!(AP,0.0)
    fill!(p_index,0.0)
    fill!(δP,0.0)
    fill!(AP2,0.0)


    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
    res_par_old = parallel_max_all(abs.(AP),par_env)
    
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = Vector{Int32}(undef,27); fill!(cols_,0)
    values_ = Matrix{Float64}(undef,1,27); fill!(values_,0.0)

    dynamic_dP = false
    jacobi_iter = 1
    HYPRE_IJMatrixInitialize(jacob)
    compute_hypre_jacobian!(dynamic_dP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
    HYPRE_IJMatrixAssemble(jacob)

    parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
    J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])

    iter = 1
    while true
        P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .= P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        # iter +=1
        HYPRE_IJVectorInitialize(b)
        HYPRE_IJVectorInitialize(x)
        # if iter > 1
        for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
            row_ = p_index[i,j,k]
            HYPRE_IJVectorSetValues(x,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            HYPRE_IJVectorSetValues(b, 1, pointer(Int32.([row_])), pointer(Float64.([AP[i,j,k]])))
        end
        MPI.Barrier(par_env.comm)
        HYPRE_IJVectorAssemble(b)
        par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(b, par_AP_ref)
        par_b_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])
        HYPRE_IJVectorAssemble(x)
        par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(x, par_Pn_ref)
        par_x_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])


        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)

        # hyp_iter = hyp_solve(solver_ref,precond_ref, J, b, x, par_env, "LGMRES")     
        hyp_iter = hyp_solve(solver_ref,precond_ref, J, par_b_old, par_x_new, par_env, "LGMRES")
        
        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            int_x = zeros(1)
            HYPRE_IJVectorGetValues(x,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            P_k[i,j,k] -= int_x[1]
        end

        #! second point
        A!(AP2,uf,vf,wf,P_k,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        res_par = parallel_max_all(abs.(AP2[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
        
        HYPRE_IJVectorInitialize(b)
        HYPRE_IJVectorInitialize(x)
        # if iter > 1
        for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
            row_ = p_index[i,j,k]
            HYPRE_IJVectorSetValues(x,1,pointer(Int32.([row_])),pointer(Float64.([0.0])))
            HYPRE_IJVectorSetValues(b, 1, pointer(Int32.([row_])), pointer(Float64.([AP2[i,j,k]])))
        end
        MPI.Barrier(par_env.comm)
        HYPRE_IJVectorAssemble(b)
        par_AP_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(b, par_AP_ref)
        par_b_old = convert(Ptr{HYPRE_ParVector}, par_AP_ref[])
        HYPRE_IJVectorAssemble(x)
        par_Pn_ref = Ref{Ptr{Cvoid}}(C_NULL)
        HYPRE_IJVectorGetObject(x, par_Pn_ref)
        par_x_new = convert(Ptr{HYPRE_ParVector}, par_Pn_ref[])


        solver_ref = Ref{HYPRE_Solver}(C_NULL)
        precond_ref = Ref{HYPRE_Solver}(C_NULL)

             
        hyp_iter = hyp_solve(solver_ref,precond_ref, J, par_b_old, par_x_new, par_env, "LGMRES")
        
        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            int_x = zeros(1)
            HYPRE_IJVectorGetValues(x,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            δP[i,j,k] = int_x[1]
        end


        β = 1.0

        P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .-= dot(((AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-β*AP2[imin_:imax_,jmin_:jmax_,kmin_:kmax_])./(eps().+AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-(β-2)*AP2[imin_:imax_,jmin_:jmax_,kmin_:kmax_])), δP[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
        P_k .-=parallel_mean_all(P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        
        A!(AP_k,uf,vf,wf,P_k,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        sum_res = parallel_sum_all(AP_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
        res_par = parallel_max_all(abs.(AP_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
        # if res_par_old < res_par
        if res_par_old - res_par < -1e2
            # println("res diff= ",abs(res_par_old - res_par))
            # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum_res)
            # iter += 1
            jacobi_iter += 1
            println("number of jacobian computes = $jacobi_iter")
            dynamic_dP = true
            HYPRE_IJMatrixInitialize(jacob)
            compute_hypre_jacobian!(dynamic_dP,jacobi_iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP_k,LHS,tmp4,verts,tets,par_env,mesh)
            HYPRE_IJMatrixAssemble(jacob)
            parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
            dynamic_dP = false
        else
            P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .=P_k[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
            res_par_old = parallel_max_all(abs.(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
            sum_res = parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
            jacobi_iter = 1
            iter+=1
            # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par_old,sum_res)
        end
        # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum_res)
        if res_par_old < tol; return iter; end 
    end
end 


function jacob_single(jacob,LHS1,LHS2,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts_arr,tets_arr,param,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    delta = 5
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
        add_perturb!(P,delta,i,j,k,mesh,par_env)
        A!(i,j,k,LHS1,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts_arr,tets_arr,param,mesh,par_env)
        remove_perturb!(P,delta,i,j,k,mesh,par_env)
        add_perturb!(P,-delta,i,j,k,mesh,par_env)
        A!(i,j,k,LHS2,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts_arr,tets_arr,param,mesh,par_env)
        remove_perturb!(P,-delta,i,j,k,mesh,par_env)
        jacob[i,j,k] = ((LHS1[i,j,k]
        - LHS2[i,j,k])
        /̂2delta)
    end
    return nothing 
end


# residual iteration method with accompanying functions
function weighted_sum(arrs::Vector{Array{Float64,3}}, weights::Vector{Float64})
    result = zeros(size(arrs[1]))
    for i in 1:length(arrs)
        result .+= weights[i] .* arrs[i]
    end
    return result
end

function anderson_accel(Fhist)
    m = length(Fhist)
    
    # flatten residual into a column vector
    Fmat = hcat([vec(F) for F in Fhist]...)
    # construct constraint matrix
    A = Fmat
    b = zeros(size(A,1))
    C = ones(1,m)
    d = [1.0]

    KKT = [A' * A C'; C zeros(1,1)]
    rhs = [A' * b; d]

    α_aug = KKT \ rhs
    α = α_aug[1:end-1]
    return α

end

function res_iteration(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,AP,Gn,AP2,jacob,verts,tets,param,mesh,par_env) 
    @unpack Nx,Ny,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_,dx,dy,dz = mesh
    @unpack tol = param

    Fhist = Vector{Array{Float64,3}}()
    Phist = Vector{Array{Float64,3}}()

    # Gn = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmaxo_,kmino_:kmaxo_);fill!(Gn,0.0)

    max_iter = 100000
    ω = 0.8
    m = 50
    iter = 0

    jacob_single(jacob,AP,AP2,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
    while iter < max_iter
        iter += 1
        # m = iter
        # evaluate objective function
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        # calculate values for jacobi step
        # if iter % 100 == 0  
        #     jacob_single(jacob,AP,AP2,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)
        # end
        # define variables for anderson acceleration
        Fn = - ω * AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]./jacob[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        Gn[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+ Fn

        if length(Fhist) >= m
            popfirst!(Fhist)
            popfirst!(Phist)
        end
        
        push!(Fhist,Fn[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
        push!(Phist,Gn[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
        
        if length(Fhist) > 1
            α = anderson_accel(Fhist)
            # P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = weighted_sum(Phist,α)
            Pnew = weighted_sum(Phist,α)
            β = 1.0
            P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = β * Pnew + (1-β) * P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]

        else
            P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = Gn[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        end
        
        res_norm = maximum(abs.(AP))
        if iter % 100 == 0
            println("residual at iter $iter = $(maximum(abs.(AP)))")
        end
        if res_norm < tol
            break
        end
    end

    return iter

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

        # HYPRE.assemble!(matrix,[rows_],cols_[1:ncols],values_[:,1:ncols])
        HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
    end
end

function FC_hypre_solver(P,RHS,denx,deny,denz,p_index,param,mesh,par_env,jacob,b_vec,x_vec)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    @unpack comm,nprocx,nprocy,nprocz,nproc,irank,iroot,isroot,irankx,iranky,irankz = par_env
    
    #! prep indices
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = Vector{Int32}(undef,27); fill!(cols_,0)
    values_ = Matrix{Float64}(undef,1,27); fill!(values_,0.0)

    HYPRE_IJVectorInitialize(x_vec)
    HYPRE_IJVectorInitialize(b_vec)

    for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
        row_ = p_index[i,j,k]
        HYPRE_IJVectorSetValues(b_vec,1, pointer(Int32.([row_])), pointer(Float64.([RHS[i,j,k]])))
        HYPRE_IJVectorSetValues(x_vec,1, pointer(Int32.([row_])), pointer(Float64.([0.0])))
    end

    HYPRE_IJVectorAssemble(x_vec)
    par_x_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJVectorGetObject(x_vec, par_x_ref)
    par_x = convert(Ptr{HYPRE_ParVector}, par_x_ref[])

    HYPRE_IJVectorAssemble(b_vec)
    par_b_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJVectorGetObject(b_vec, par_b_ref)
    par_b = convert(Ptr{HYPRE_ParVector}, par_b_ref[])

    
    # RHS_assembler = HYPRE.start_assemble!(b)
    # x_assembler = HYPRE.start_assemble!(x)

    # for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
    #     row_ = p_index[i,j,k]
    #     HYPRE.assemble!(RHS_assembler,[row_],[RHS[i,j,k]])
    #     HYPRE.assemble!(x_assembler,[row_],[0.0])
    # end
    
    # b = HYPRE.finish_assemble!(RHS_assembler)
    # x = HYPRE.finish_assemble!(x_assembler)
    
    # J_assembler = HYPRE.start_assemble!(jacob)

    # compute_lap_op!(J_assembler,p_index,cols_,values_,denx,deny,denz,par_env,mesh)
    
    # par_A = HYPRE.finish_assemble!(J_assembler)
    # HYPRE_IJMatrixPrint(par_A,"par_A")
    compute_lap_op!(jacob,p_index,cols_,values_,denx,deny,denz,par_env,mesh)

    HYPRE_IJMatrixAssemble(jacob)
    parcsr_A_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_A_ref)
    par_A = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_A_ref[])
    

    solver_ref = Ref{HYPRE_Solver}(C_NULL)
    precond_ref = Ref{HYPRE_Solver}(C_NULL)

    #! HYPRE.jl solver setup (not as expansive and more tricky to converge)
    # precond = HYPRE.BoomerAMG(; MaxIter = 1, RelaxType = 3,CoarsenType = 0, InterpType = 12,NumSweeps= 2)
    # solver = HYPRE.GMRES(comm; Tol= tol, KDim = 20, Precond=precond)
    # HYPRE.solve!(solver,x,J,b)
    # iter = HYPRE.GetNumIterations(solver)

    #! HYPRE.LibHYPRE c function call solver
    iter = hyp_solve(solver_ref,precond_ref, par_A, par_b, par_x, par_env, "GMRES-AMG")
    # iter = hyp_solve(solver_ref,precond_ref, par_A, par_b, par_x, par_env, "LGMRES")
    maxdP = zeros(1)
    
    for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
        int_x = zeros(1)
        HYPRE.HYPRE_IJVectorGetValues(x_vec,1,pointer(Int32.([p_index[i,j,k]])),int_x)
        P[i,j,k] = int_x[1]
    end
    
    # account for drift
    P .-=parallel_mean_all(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
    
    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env)
    
    return iter
end