

# Solve Poisson equation: δP form

function pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mesh,par_env,denx,deny,denz)

    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    RHS = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    gradx = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    grady = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    gradz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    #LHS = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)

    # @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
    #     # RHS
    #     RHS[i,j,k]= 1/dt * ( 
    #         ( uf[i+1,j,k] - uf[i,j,k] )*(denx[i+1,j,k]-denx[i,j,k])/(2dx) +
    #         ( vf[i,j+1,k] - vf[i,j,k] )*(deny[i,j+1,k]-deny[i,j,k])/(2dy) +
    #         ( wf[i,j,k+1] - wf[i,j,k] )*(denz[i,j,k+1]-denz[i,j,k])/(2dz) )
    # end
    
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # RHS
        RHS[i,j,k]=  ( 
            ( uf[i+1,j,k] - uf[i,j,k] )/(dx) +
            ( vf[i,j+1,k] - vf[i,j,k] )/(dy) +
            ( wf[i,j,k+1] - wf[i,j,k] )/(dz) )
    end
    iter = poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,VF,dt,param,mesh,par_env,denx,deny,denz)

    return iter
end



function poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,VF,dt,param,mesh,par_env,denx,deny,denz)
    @unpack pressureSolver = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh


    if pressureSolver == "GaussSeidel"
        iter = GaussSeidel!(P,RHS,uf,vf,wf,denx,deny,denz,dt,param,mesh,par_env)
    elseif pressureSolver == "ConjugateGradient"
        iter = conjgrad!(P,RHS,param,mesh,par_env)

    elseif pressureSolver == "Secant"
        iter = Secant_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,param,mesh,par_env)

    elseif pressureSolver == "NLsolve"
        iter = computeNLsolve!(P,uf,vf,wf,gradx,grady,gradz,band,den,dt,param,mesh,par_env)
    else
        error("Unknown pressure solver $pressureSolver")
    end

    return iter
end


function lap!(L,P,param,mesh)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    fill!(L,0.0)
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        L[i,j,k] = (
            (P[i-1,j,k] - 2P[i,j,k] + P[i+1,j,k]) /̂ dx^2 +
            (P[i,j-1,k] - 2P[i,j,k] + P[i,j+1,k]) /̂ dy^2 +
            (P[i,j,k-1] - 2P[i,j,k] + P[i,j,k+1]) /̂ dz^2 )
    end
    return nothing
end

# LHS of pressure poisson equation

function A!(LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env)

    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh


    fill!(gradx,0.0)
    fill!(grady,0.0)
    fill!(gradz,0.0)

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env) # (overwrites BCs if periodic)

    #suspect that the correct gradient is being calculate due to loop
    #! need cell centered densities
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        gradx[i,j,k]=dt/denx[i,j,k]*(P[i,j,k]-P[i-1,j,k])/dx
    end

    for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        grady[i,j,k]=dt/deny[i,j,k]*(P[i,j,k]-P[i,j-1,k])/dy
    end

    for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        gradz[i,j,k]=dt/denz[i,j,k]*(P[i,j,k]-P[i,j,k-1])/dz
    end


    uf1 = uf-gradx
    vf1 = vf-grady
    wf1 = wf-gradz



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
function A!(i,j,k,LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    fill!(gradx,0.0)
    fill!(grady,0.0)
    fill!(gradz,0.0)

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env) # (overwrites BCs if periodic)
    
    #probably dont need to calculate every pt but need a 3x3 stencil for velocity projection with i,j,k being in a corner
    #maybe want to use diff finite difference approx
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        gradx[i,j,k]=dt/denx[i,j,k]*(P[i,j,k]-P[i-1,j,k])/̂dx
    end
    
    for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        grady[i,j,k]=dt/deny[i,j,k]*(P[i,j,k]-P[i,j-1,k])/̂dy
    end

    for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        gradz[i,j,k]=dt/denz[i,j,k]*(P[i,j,k]-P[i,j,k-1])/̂dz
    end

    #? might want to use these smaller loops
    # for ii = i:i+1
    #     gradx[ii,j,k]=(P[ii,j,k]-P[ii-1,j,k])/̂dx
    # end
    # for jj = j:j+1
    #     grady[i,jj,k]=(P[i,jj,k]-P[i,jj-1,k])/̂dy
    # end
    # for kk = k:k+1
    #     gradz[i,j,kk]=(P[i,j,kk]-P[i,j,kk-1])/̂dz
    # end

    uf1 = uf-gradx
    vf1 = vf-grady
    wf1 = wf-gradz



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
    end
    return LHS[i,j,k]
end

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
            (A!(i,j,k,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env)
            - A!(i,j,k,LHS2,uf,vf,wf,P.-dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env))
            ./̂2delta)
    end
    return J 
end


# Secant method
function Secant_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,param,mesh,par_env)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh
    AP = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    fill!(AP,0.0)
    # outflowCorrection!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)

    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env)


    # Iterate 
    iter=0
    while true
        iter += 1

   
        # compute jacobian
        J = computeJacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)

        P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .-= 0.5AP./̂J

        P .-=mean(P)

        # #     #Need to introduce outflow correction
        # outflowCorrection!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
        # end
            #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env)

        
        res = maximum(abs.(AP))
        if res < tol
            return iter
        end

        # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res,sum(AP))


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



"""
GaussSeidel Poisson Solver
"""
function GaussSeidel!(P,RHS,uf,vf,wf,denx,deny,denz,dt,param,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack isroot = par_env
    @unpack tol = param
    maxIter=100
    iter = 0

    # println(vf[:, jmin_+1, :]) 
    # println(vf[:, jmax_-1, :])
    #apply outflow correction
    outflowCorrection!(RHS,P,uf,vf,wf,denx,deny,denz,param,mesh,par_env)
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # RHS
        RHS[i,j,k]=  ( 
            ( uf[i+1,j,k] - uf[i,j,k] )/(dx) +
            ( vf[i,j+1,k] - vf[i,j,k] )/(dy) +
            ( wf[i,j,k+1] - wf[i,j,k] )/(dz) )
    end
    
    while true
        iter += 1
        max_update=0.0
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            
            Pnew = ( (RHS[i,j,k]
                    - (P[i-1,j,k]-2P[i,j,k]+P[i+1,j,k])/̂dx^2
                    - (P[i,j-1,k]-2P[i,j,k]+P[i,j+1,k])/̂dy^2 
                    - (P[i,j,k-1]-2P[i,j,k]+P[i,j,k+1])/̂dz^2) 
                    /̂ (-2.0/dx^2 - 2.0/dy^2 - 2.0/dz^2) )
            max_update=max(max_update,abs(Pnew-P[i,j,k]))
            # println(Pnew)
            P[i,j,k] = Pnew
        end
        # for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        #     res_factor = (-denx[i,j,k] * denx[i+1,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k] * denz[i,j,k+1] * RHS[i,j,k] * dx^2 * dy^2 * dz^2)
        #     Pk_pos = (denx[i,j,k] * denx[i+1,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k] * dt * dx^2 * dy^2)
        #     Pk_neg = (denx[i,j,k] * denx[i+1,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k+1] * dt * dx^2 * dy^2)
        #     Pj_pos = (denx[i,j,k] * denx[i+1,j,k] * deny[i,j,k] * denz[i,j,k] * denz[i,j,k+1] * dt * dx^2 * dz^2)
        #     Pj_neg = (denx[i,j,k] * denx[i+1,j,k] * deny[i,j+1,k] * denz[i,j,k] * denz[i,j,k+1] * dt * dx^2 * dz^2)
        #     Pi_pos = (denx[i,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k] * denz[i,j,k+1] * dt * dy^2 * dz^2)
        #     Pi_neg = (denx[i+1,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k] * denz[i,j,k+1] * dt * dy^2 * dz^2)
        #     Pnew = (res_factor + Pk_pos*P[i,j,k+1] + Pk_neg*P[i,j,k-1] + Pj_pos*P[i,j+1,k] + Pj_neg*P[i,j-1,k] + Pi_pos*P[i+1,j,k] + Pi_neg*P[i-1,j,k])/̂
        #             dt*(Pk_pos + Pk_neg + Pj_pos + Pj_neg + Pi_pos + Pi_neg)
        #     # println(Pnew)
        #     max_update=max(max_update,abs(Pnew-P[i,j,k]))
        #     P[i,j,k] = Pnew 
        # end
        # # error("stop")
        update_borders!(P,mesh,par_env)
        Neumann!(P,mesh,par_env)
        println("Max update = ",max_update)
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
        alpha = rsold /̂ sum
        P .+= alpha*p
        r -= alpha * Ap
        rsnew = parallel_sum_all(r[ix,iy,iz].^2,par_env)
        if sqrt(rsnew) < tol
            return iter
        end
        p = r + (rsnew /̂ rsold) * p
        Neumann!(p,mesh,par_env)   
        update_borders!(p,mesh,par_env) # (overwrites BCs if periodic)
        rsold = rsnew

    end
    
    isroot && println("Failed to converged Poisson equation rsnew = $rsnew")
    
    return length(RHS)
end
