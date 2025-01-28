using JSON

# Solve Poisson equation: δP form
function pressure_solver!(P,uf,vf,wf,sfx,sfy,sfz,t,dt,band,VF,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,gradx,grady,gradz,verts,tets,BC!,jacob,b,x)
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
    iter = poisson_solve!(P,RHS,uf,vf,wf,sfx,sfy,sfz,t,gradx,grady,gradz,band,dt,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,verts,tets,jacob,b,x)

    return iter
end

function poisson_solve!(P,RHS,uf,vf,wf,sfx,sfy,sfz,t,gradx,grady,gradz,band,dt,param,mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,verts,tets,jacob,b,x)
    @unpack pressureSolver = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    if pressureSolver == "FC_hypre"
        iter = FC_hypre_solver(P,RHS,denx,deny,denz,tmp4,param,mesh,par_env,jacob,b,x)
    elseif pressureSolver == "hypreSecant"
        iter = Secant_jacobian_hypre!(P,uf,vf,wf,sfx,sfy,sfz,t,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,verts,tets,param,mesh,par_env,jacob,b,x)
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
        HYPRE_LGMRESSetTol(solver, 1e-9) # conv. tolerance
        HYPRE_LGMRESSetMaxIter(solver,1000)
        # HYPRE_LGMRESSetPrintLevel(solver, 2) # print solve info
        HYPRE_LGMRESSetLogging(solver, 1) # needed to get run info later

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
end


# Semi-Lagrangian pressure solvers
function compute_hypre_jacobian!(iter,matrix,coeff_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS1,LHS2,p,tets_arr,par_env,mesh)
    @unpack  imin, imax, jmin,jmax,kmin,kmax,imin_, imax_, jmin_, jmax_,jmino_,imino_,jmaxo_,imaxo_,kmin_,kmax_,kmino_,kmaxo_,Nx,Nz,Ny = mesh
    # if iter == 0
    delta = 1e5
    # else
        # delta = 1e-1
    # end
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

function Secant_jacobian_hypre!(P,uf,vf,wf,sfx,sfy,sfz,t,gradx,grady,gradz,band,dt,denx,deny,denz,LHS,AP,p_index,tmp4,P_k,AP_k,verts,tets,param,mesh,par_env,jacob,b,x)
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
    res_par = parallel_max_all(abs.(AP),par_env)
    p_min,p_max = prepare_indices(p_index,par_env,mesh)

    cols_ = Vector{Int32}(undef,27); fill!(cols_,0)
    values_ = Matrix{Float64}(undef,1,27); fill!(values_,0.0)
    

    # J_assembler = HYPRE.start_assemble!(jacob)
    # compute_hypre_jacobian!(J_assembler,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
    # J = HYPRE.finish_assemble!(J_assembler)

    iter=0
    HYPRE_IJMatrixInitialize(jacob)
    compute_hypre_jacobian!(iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
    HYPRE_IJMatrixAssemble(jacob)

    parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
    J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])

    # #! used to grab jacobian rows at (12,13,11) & (14,13,11)
    # #! output JSON dictionary is the input for the jacobian_check.jl function
    # jacobians = Dict(
    # "12,11,12" => Vector{Tuple{String, Float64}}(),
    # "12,15,12" => Vector{Tuple{String, Float64}}()
    # )
    # count = 0
    # for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
    #     count+=1
    #     int_x1 = zeros(1)
    #     HYPRE_IJMatrixGetValues(jacob,1,pointer(Int32.([1])),pointer(Int32.([p_index[12,11,12]])),pointer(Int32.([p_index[i,j,k]])),int_x1)
    #     if int_x1[1] != 0.0
    #         push!(jacobians["12,11,12"], ("$i,$j,$k",int_x1[1]))
    #     end
    #     int_x2 = zeros(1)
    #     HYPRE_IJMatrixGetValues(jacob,1,pointer(Int32.([1])),pointer(Int32.([p_index[12,15,12]])),pointer(Int32.([p_index[i,j,k]])),int_x2)
    #     if int_x2[1] != 0.0
    #         push!(jacobians["12,15,12"], ("$i,$j,$k",int_x2[1]))
    #     end
    # end

    # open("jacob_comp_dict_24tets.json","w") do file
    #     JSON.print(file,jacobians)
    # end

    # Iterate     
    while true
        iter += 1

        #? save for recomputing jacobian
        # if iter == 2
        #     HYPRE_IJMatrixInitialize(jacob)
        #     compute_hypre_jacobian!(iter,jacob,p_index,cols_,values_,P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,LHS,tmp4,verts,tets,par_env,mesh)
        #     HYPRE_IJMatrixAssemble(jacob)
        #     parcsr_J_ref = Ref{Ptr{Cvoid}}(C_NULL)
        #     HYPRE_IJMatrixGetObject(jacob, parcsr_J_ref)
        #     J = convert(Ptr{HYPRE_ParCSRMatrix}, parcsr_J_ref[])
        # end
        #! reinit
        # b_assembler = HYPRE.start_assemble!(b)
        # x_assembler = HYPRE.start_assemble!(x)
        
        # b = HYPRE.finish_assemble!(b_assembler)
        # x = HYPRE.finish_assemble!(x_assembler)

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

        # hyp_iter = hyp_solve(solver_ref,precond_ref, J, b, x, par_env, "LGMRES")
     
        hyp_iter = hyp_solve(solver_ref,precond_ref, J, par_b_old, par_x_new, par_env, "LGMRES")
        for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
            int_x = zeros(1)
            HYPRE_IJVectorGetValues(x,1,pointer(Int32.([p_index[i,j,k]])),int_x)
            # P_step[i,j,k] = int_x[1]
            if iter> 10
                P[i,j,k] -= 0.5*int_x[1]
            else
                P[i,j,k] -= int_x[1]
            end
        end
        # account for drift
        P .-=parallel_mean_all(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)

        

        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,verts,tets,param,mesh,par_env)

        res_par = parallel_max_all(abs.(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
        
        if res_par < tol || iter == 50
            return iter
        end

        if iter % 10 == 0
        # if t == 6
            # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,sum(AP))
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res_par,parallel_sum_all(AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env))
            # J = compute_sparse2D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,AP,tmp2,tmp3,tmp4,mesh,par_env)
        end
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

    # println("RHS at 12,13,11 = ",RHS[12,13,11])
    # println("RHS at 14,13,11 = ",RHS[14,13,11])
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
    iter = hyp_solve(solver_ref,precond_ref, par_A, par_b, par_x, par_env, "LGMRES")
    
    for k in kmin_:kmax_,j in jmin_:jmax_,i in imin_:imax_
        int_x = zeros(1)
        HYPRE.HYPRE_IJVectorGetValues(x_vec,1,pointer(Int32.([p_index[i,j,k]])),int_x)
        
        P[i,j,k] = int_x[1]
    end
    
    # account for drift
    P .-=parallel_mean_all(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
    
    # println("P at 12,13,11 = ",P[12,13,11])
    # println("P at 14,13,11 = ",P[14,13,11])
    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env)
    
    return iter
end