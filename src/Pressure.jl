

# Solve Poisson equation: δP form
function pressure_solver!(P,uf,vf,wf,dt,band,param,mesh,par_env)
    @unpack rho = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh


    RHS = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    gradx = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    grady = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    gradz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    #LHS = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)

    sig=0.1
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # RHS
        RHS[i,j,k]= rho/dt * ( 
            ( uf[i+1,j,k] - uf[i,j,k] )/(dx) +
            ( vf[i,j+1,k] - vf[i,j,k] )/(dy) +
            ( wf[i,j,k+1] - wf[i,j,k] )/(dz) )
    end

    iter = poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env)

    return iter
end



function poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env)
    @unpack pressureSolver = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh


    if pressureSolver == "GaussSeidel"
        iter = GaussSeidel!(P,RHS,param,mesh,par_env)
    elseif pressureSolver == "ConjugateGradient"
        iter = conjgrad!(P,RHS,param,mesh,par_env)
    elseif pressureSolver == "Secant"
        iter = Secant_jacobian!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env)
    elseif pressureSolver == "NLsolve"
        iter = computeNLsolve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env)
    else
        error("Unknown pressure solver $pressureSolver")
    end

    return iter
end


function lap!(L,P,param,mesh)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        L[i,j,k] = (
            (P[i-1,j,k] - 2P[i,j,k] + P[i+1,j,k]) / dx^2 +
            (P[i,j-1,k] - 2P[i,j,k] + P[i,j+1,k]) / dy^2 +
            (P[i,j,k-1] - 2P[i,j,k] + P[i,j,k+1]) / dz^2 )
    end
    return nothing
end

# LHS of pressure poisson equation
function A!(RHS,LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
    @unpack rho= param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # println(uf)
    fill!(gradx,0.0)
    fill!(grady,0.0)
    fill!(gradz,0.0)

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env) # (overwrites BCs if periodic)

    #suspect that the correct gradient is being calculate due to loop
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        gradx[i,j,k]=(P[i,j,k]-P[i-1,j,k])/dx
    end

    for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        grady[i,j,k]=(P[i,j,k]-P[i,j-1,k])/dy
    end

    for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        gradz[i,j,k]=(P[i,j,k]-P[i,j,k-1])/dz
    end

    uf1 = uf-dt/rho*gradx
    vf1 = vf-dt/rho*grady
    wf1 = wf-dt/rho*gradz



    fill!(LHS,0.0)

    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        if abs(band[i,j,k]) <= 1
            tets, inds = cell2tets_withProject_uvwf(i,j,k,uf1,vf1,wf1,dt,mesh)
            if any(isnan,tets)
                error("Nan in tets at ", i,j,k)
            end
            v2 = dx*dy*dz
            v1 = tets_vol(tets)
            LHS[i,j,k] = (v2-v1) /̂ v2 /̂ dt
        else 
            # Calculate divergence with finite differnce
            du_dx = ( uf1[i+1,j,k] - uf1[i,j,k] )/(dx)
            dv_dy = ( vf1[i,j+1,k] - vf1[i,j,k] )/(dy)
            dw_dz = ( wf1[i,j,k+1] - wf1[i,j,k] )/(dz)
            LHS[i,j,k] = du_dx + dv_dy + dw_dz
        end
    end
    return nothing
end

#local A! matrix
function A!(i,j,k,RHS,LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
    @unpack rho= param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    fill!(gradx,0.0)
    fill!(grady,0.0)
    fill!(gradz,0.0)

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env) # (overwrites BCs if periodic)
    
    #probably dont need to calculate every pt but need a 3x3 stencil for velocity projection with i,j,k being in a corner
    #maybe want to use diff finite difference approx
    # for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
    #     gradx[i,j,k]=(P[i,j,k]-P[i-1,j,k])/̂dx
    # end
    
    # for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
    #     grady[i,j,k]=(P[i,j,k]-P[i,j-1,k])/̂dy
    # end

    # for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
    #     gradz[i,j,k]=(P[i,j,k]-P[i,j,k-1])/̂dz
    # end
    # println(dz)
    # println(P[i,j,k])
    # println(P[i,j,k-1])
    # println(gradz[i,j,k])
    # println(gradx)

    for ii = i:i+1
        gradx[ii,j,k]=(P[ii,j,k]-P[ii-1,j,k])/̂dx
    end
    for jj = j:j+1
        grady[i,jj,k]=(P[i,jj,k]-P[i,jj-1,k])/̂dy
    end
    for kk = k:k+1
        gradz[i,j,kk]=(P[i,j,kk]-P[i,j,kk-1])/̂dz
        if any(isnan,gradz)
            println(dz)
            println(P[i,j,k])
            println(P[i,j,k-1])
            println(gradz[i,j,k])
        end

    end
    # if any(isnan,gradz)
    #     println(gradz)
    #     println(gradx)
    #     error("Nan in gradz at ", i,j,k)
    # end
    uf1 = uf-dt/̂rho*gradx
    vf1 = vf-dt/̂rho*grady
    wf1 = wf-dt/̂rho*gradz



    if abs(band[i,j,k]) <= 1
        tets, inds = cell2tets_withProject_uvwf(i,j,k,uf1,vf1,wf1,dt,mesh)
        if any(isnan,tets)
            error("Nan in tets at ", i,j,k)
        end
        v2 = dx*dy*dz
        v1 = tets_vol(tets)
        LHS[i,j,k] = (v2-v1) /̂ v2 /̂ dt
    else 
            # Calculate divergence with finite differnce
        du_dx = ( uf1[i+1,j,k] - uf1[i,j,k] )/̂(dx)
        dv_dy = ( vf1[i,j+1,k] - vf1[i,j,k] )/̂(dy)
        dw_dz = ( wf1[i,j,k+1] - wf1[i,j,k] )/̂(dz)
        LHS[i,j,k] = du_dx + dv_dy + dw_dz
        if any(isnan,du_dx)
            println(du_dx)
            error("Nan in du_dx at ", i,j,k)
        end
        if any(isnan,dv_dy)
            println(dv_dy)
            error("Nan in dv_dy at ", i,j,k)
        end
        if any(isnan,dw_dz)
            println(dw_dz)
            error("Nan in dw_dz at ", i,j,k)
        end
    end
    return LHS[i,j,k]
end



#//* want to implement bubble rising in 2-D so provide outflow correction for top of container
#! currently correcting for example 3 flow out the right side
function outflowCorrection!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
    @unpack jmin_,jmax_ = mesh
    @unpack tol = param
    iter = 0; maxIter=100
    d = sum(AP)
    while abs(d) >tol*1e-1
        iter += 1

        #check divergence
        correction = d./̂((jmax_-jmin_))
        #? maybe only need to correct for lower half of right side
        uf[end,0:5,:] .-= 0.5correction
        A!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
        d = sum(AP)

        if iter == maxIter
            @warn("outflowCorrection did not converge!")
            return
        end
    end
end

function computeJacobian(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env)
    @unpack Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    J = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    dp = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    LHS1 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    LHS2 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    println(dp)

    delta = 1.0
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        fill!(LHS1,0.0)
        fill!(LHS2,0.0)
        # fill!(dp,0.0)
        dp[i,j,k] += delta
        println(dp)
        error("dp wrong")
        # println(A!(i,j,k,RHS,LHS2,uf,vf,wf,P.-delta,dt,gradx,grady,gradz,band,param,mesh,par_env))
        # println(A!(i,j,k,RHS,LHS2,uf,vf,wf,P.+delta,dt,gradx,grady,gradz,band,param,mesh,par_env))
        # println(P.+delta)
        # error("stop")
        #? do we want to add/subtract delta at each point in P or just P @ i,j,k
        J[i,j,k] = (
            (A!(i,j,k,RHS,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,param,mesh,par_env)
            - A!(i,j,k,RHS,LHS2,uf,vf,wf,P.-dp,dt,gradx,grady,gradz,band,param,mesh,par_env))
            ./̂2delta)
    end
    return J 
end

# NLsolve Library
function computeNLsolve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Ghost cell indices
    gx = imino_:imaxo_; gy = jmino_:jmaxo_; gz = kmino_:kmaxo_

    LHS = OffsetArray{Float64}(undef, gx,gy,gz)

    # println(uf)
    function f!(LHS, P)
        A!(RHS,LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
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
function Secant_jacobian!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    AP = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    outflowCorrection!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)

    A!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
    # Iterate 
    iter=0
    while true
        iter += 1

   
        # compute jacobian
        J = computeJacobian(P,RHS,uf,vf,wf,gradx,grady,gradz,band,dt,param,mesh,par_env)

        P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .-= AP./̂J

        P .-=mean(P)
        # println(mean(P))
        # println(P)
        # error("stop")
        #compute new Ap
        # A!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)


        #Need to introduce outflow correction
        outflowCorrection!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
        #update new Ap
        A!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
        if iter == 100
            println(AP)
            println(sum(AP))
            error("stop")
        end
        
        res = maximum(abs.(AP))
        if res < tol
            return P
        end

        @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res,sum(AP))

    end    
    
end


"""
GaussSeidel Poisson Solver
"""
function GaussSeidel!(P,RHS,param,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack isroot = par_env
    @unpack tol = param
    maxIter=1000
    iter = 0

    while true
        iter += 1
        max_update=0.0
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            Pnew = ( (RHS[i,j,k]
                    - (P[i-1,j,k]+P[i+1,j,k])/dx^2
                    - (P[i,j-1,k]+P[i,j+1,k])/dy^2 
                    - (P[i,j,k-1]+P[i,j,k+1])/dz^2) 
                    / (-2.0/dx^2 - 2.0/dy^2 - 2.0/dz^2) )
            max_update=max(max_update,abs(Pnew-P[i,j,k]))
            P[i,j,k] = Pnew
        end
        update_borders!(P,mesh,par_env)
        Neumann!(P,mesh,par_env)
        # Check if converged
        if iter == 10
            print(max_update)
        end
        max_update = parallel_max_all(max_update,par_env)
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
function conjgrad!(P,RHS,param,mesh,par_env)
    @unpack dx,dy,dz = mesh
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack tol = param
    @unpack irank,isroot = par_env

    # Interior indices
    ix = imin_:imax_; iy = jmin_:jmax_;  iz = kmin_:kmax_
    # Ghost cell indices
    gx = imino_:imaxo_; gy = jmino_:jmaxo_; gz = kmino_:kmaxo_
    
    # Allocat work arrays (with ghost cells for comm)
    r  = OffsetArray{Float64}(undef, gx,gy,gz)
    p  = OffsetArray{Float64}(undef, gx,gy,gz)
    Ap = OffsetArray{Float64}(undef, gx,gy,gz)

    lap!(r,P,param,mesh)

    r[ix,iy,iz] = RHS.parent - r[ix,iy,iz]
    Neumann!(r,mesh,par_env)
    update_borders!(r,mesh,par_env) # (overwrites BCs if periodic)
    p .= r
    rsold = parallel_sum_all(r[ix,iy,iz].^2,par_env)
    rsnew = 0.0
    for iter = 1:length(RHS)
        lap!(Ap,p,param,mesh)
        sum = parallel_sum_all(p[ix,iy,iz].*Ap[ix,iy,iz],par_env)
        alpha = rsold / sum
        P .+= alpha*p
        r -= alpha * Ap
        rsnew = parallel_sum_all(r[ix,iy,iz].^2,par_env)
        if sqrt(rsnew) < tol
            return iter
        end
        p = r + (rsnew / rsold) * p
        Neumann!(p,mesh,par_env)   
        update_borders!(p,mesh,par_env) # (overwrites BCs if periodic)
        rsold = rsnew
    end
    isroot && println("Failed to converged Poisson equation rsnew = $rsnew")

    return length(RHS)
end
