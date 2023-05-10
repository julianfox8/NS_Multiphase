using NLsolve

# Solve Poisson equation: δP form (Ax=0)
function semi_lag_pressure_solver!(P,uf,vf,wf,dt,param,mesh,par_env)

    iter = semi_lag_poisson_solve!(P,uf,vf,wf,dt,param,mesh,par_env)

    return iter
end

function semi_lag_poisson_solve!(P,uf,vf,wf,dt,param,mesh,par_env)
    @unpack pressureSolver = param


    if pressureSolver == "Secant"
        iter = Secant!(P,uf,vf,wf,dt,param,mesh,par_env)
    elseif pressureSolver == "NLsolve"
        iter = computeNLsolve(P,uf,vf,wf,dt,param,mesh,par_env)
    else
        error("Unknown pressure solver $pressureSolver")
    end

    return iter
end

"""
Secant Poisson Solver
"""


# LHS of pressure poisson equation
function A!(LHS,uf,vf,wf,P,gx,gy,gz,dt,param,mesh,par_env,tet_volumes=nothing)
    @unpack rho = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    gradientx = OffsetArray{Float64}(undef, gx,gy,gz)
    gradienty = OffsetArray{Float64}(undef, gx,gy,gz)
    gradientz = OffsetArray{Float64}(undef, gx,gy,gz)

    fill!(gradientx,0.0)
    fill!(gradienty,0.0)
    fill!(gradientz,0.0)
    fill!(LHS,0.0)
    println(P)
    println(uf)
    

    #suspect that the correct gradient is being calculate due to loop
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        gradientx[i,j,k]=(P[i,j,k]-P[i-1,j,k])/dx
    end

    for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        gradienty[i,j,k]=(P[i,j,k]-P[i,j-1,k])/dy
    end

    for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        gradientz[i,j,k]=(P[i,j,k]-P[i,j,k-1])/dz
    end

    uf1 = uf-dt/rho*gradientx
    vf1 = vf-dt/rho*gradienty
    wf1 = wf-dt/rho*gradientz
    println(uf1)


    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        tets, inds = cell2tets_withProject_uvwf(i,j,k,uf1,vf1,wf1,dt,mesh)
        if any(isnan,tets)
            error("Nan in tets at ", i,j,k)
        end
        v2 = dx*dy*dz
        v1 = tets_vol(tets)
        if tet_volumes !== nothing
            tet_volumes[i,j,k] = v1
        end
        LHS[i,j,k] = (v2-v1) /̂ v2 /̂ dt
        # LHS = semi_lag_divergence(uf,vf,wf,dt,mesh)
    end
    return nothing
end

# NLsolve Library
function computeNLsolve(P,uf,vf,wf,dt,param,mesh,par_env)
    @unpack rho,tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Ghost cell indices
    gx = imino_:imaxo_; gy = jmino_:jmaxo_; gz = kmino_:kmaxo_

    LHS = OffsetArray{Float64}(undef, gx,gy,gz)


    function f!(LHS, P)
        A!(LHS,uf,vf,wf,P,gx,gy,gz,dt,param,mesh,par_env)
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

# Secant method
function Secant!(P,uf,vf,wf,dt,param,mesh,par_env)
    @unpack rho,tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh

    
    maxIter = 5000
    # Interior indices
    ix = imin_:imax_; iy = jmin_:jmax_;  iz = kmin_:kmax_
    # Ghost cell indices
    gx = imino_:imaxo_; gy = jmino_:jmaxo_; gz = kmino_:kmaxo_

    # # Start with randon Pressures 
    P0=randn(size(P))
    P1=randn(size(P))


    P0 = OffsetArray(P0,0:Nx+1,0:Ny+1,0:Nz+1)
    P1 = OffsetArray(P1,0:Nx+1,0:Ny+1,0:Nz+1)


    # P0 = OffsetArray{Float64}(undef, gx,gy,gz)
    # P1 = OffsetArray{Float64}(undef, gx,gy,gz)

    #Preallocate LHS arrays

    # keep around for investigating tet volumes
    # tet_volumes = OffsetArray{Float64}(undef, gx,gy,gz)
    # tet_volumes1 = OffsetArray{Float64}(undef, gx,gy,gz)

    LHS0 = OffsetArray{Float64}(undef, gx,gy,gz)
    LHS1 = OffsetArray{Float64}(undef, gx,gy,gz)
    # Compute LHS for which LHS(P)=0
    A!(LHS0,uf,vf,wf,P0,gx,gy,gz,dt,param,mesh,par_env)
    A!(LHS1,uf,vf,wf,P1,gx,gy,gz,dt,param,mesh,par_env)


    Neumann!(P0,mesh,par_env)
    Neumann!(P1,mesh,par_env)

    update_borders!(P0,mesh,par_env)
    update_borders!(P1,mesh,par_env)



    # Iterate 
    iter=0
    while true
        iter += 1

        # if tet_volumes1 !== tet_volumes
        #     println("tet_volumes from P1 and P0 are not the same for iter ", iter)
        # end

        # Compute new P 
        # printArray("LSH1",LHS1,par_env)
        # printArray("LSH0",LHS0,par_env)
        P2 = P1 .-LHS1.*(P1.-P0) ./̂ (LHS1.-LHS0)
        if any(isnan,P2)
            error("P2 has nan")
        end
        # println("first check on ",iter)
        # If P2==P1 adjust
        for k=iz, j=iy, i=ix
            if P2[i,j,k]==P1[i,j,k] && abs(LHS1[i,j,k]) > tol
                P2[i,j,k]+=randn()
            end
        end

        # What values do I really need to apply Neumann and update borders
        Neumann!(P2,mesh,par_env)
        update_borders!(P2,mesh,par_env)
        # Neumann!(P0,mesh,par_env,param)
        # Neumann!(P1,mesh,par_env,param)
        # update_borders!(P0,mesh,par_env)
        # update_borders!(P1,mesh,par_env)

        # Shift P's 
        P0 = P1 
        P1 = P2
        # Recompute LHS for P0 and P1
        A!(LHS0,uf,vf,wf,P0,gx,gy,gz,dt,param,mesh,par_env)
        A!(LHS1,uf,vf,wf,P1,gx,gy,gz,dt,param,mesh,par_env)

        # Check if converged
        if maximum(abs.(LHS1)) < tol 
            # println("Converged at iteration ",iter," with divergence of ", maximum(abs.(LHS1)))
            # println(P1)
            return iter
        end

        # # Output
        # if rem(iter,1000)==1
        #     println("Iter = ",iter,"  Error = ",maximum(abs.(LHS1)))

        # end

        if iter == maxIter
            println("Failed to converged Poisson equation max_upate = ", maximum(abs.(LHS1)))
            break
        end
    end    
    
end



