
# Solve Poisson equation: δP form
function pressure_solver!(P,uf,vf,wf,dt,param,mesh,par_env)
    @unpack rho = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    RHS = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    sig=0.1
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # Derivatives 
        duf_dx   = ( uf[i+1,j,k] - uf[i,j,k] )/(dx)
        dvf_dy   = ( vf[i,j+1,k] - vf[i,j,k] )/(dy)
        dwf_dz   = ( wf[i,j,k+1] - wf[i,j,k] )/(dz)

        # RHS
        RHS[i,j,k]= rho/dt * ( duf_dx + dvf_dy + dwf_dz )
    end

    iter = poisson_solve!(P,RHS,param,mesh,par_env)

    return iter
end

function poisson_solve!(P,RHS,param,mesh,par_env)

    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    # Interior indices
    ix = imin_:imax_; iy = jmin_:jmax_;  iz = kmin_:kmax_

    #iter = GaussSeidel!(P,RHS,param,mesh,par_env)
    #printArray("P - Gauss Seidel",P[ix,iy,iz],par_env)
    iter = conjgrad!(P,RHS,param,mesh,par_env)
    #printArray("P - conjgrad",P[ix,iy,iz],par_env)

    return iter
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
        for i=imin_:imax_, j=jmin_:jmax_, k=kmin_:kmax_
            Pnew = ( (RHS[i,j,k]
                    - (P[i-1,j,k]+P[i+1,j,k])/dx^2
                    - (P[i,j-1,k]+P[i,j+1,k])/dy^2 
                    - (P[i,j,k-1]+P[i,j,k+1])/dz^2) 
                    / (-2.0/dx^2 - 2.0/dy^2 - 2.0/dz^2) )
            max_update=max(max_update,abs(Pnew-P[i,j,k]))
            P[i,j,k] = Pnew
        end
        update_borders!(P,mesh,par_env)
        pressure_BC!(P,mesh,par_env)
        # Check if converged
        max_update = parallel_max(max_update,par_env,recvProcs="all")
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
function lap!(L,P,mesh)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    for i=imin_:imax_, j=jmin_:jmax_, k=kmin_:kmax_
        L[i,j,k] = (
            (P[i-1,j,k] - 2P[i,j,k] + P[i+1,j,k]) / dx^2 +
            (P[i,j-1,k] - 2P[i,j,k] + P[i,j+1,k]) / dy^2 +
            (P[i,j,k-1] - 2P[i,j,k] + P[i,j,k+1]) / dz^2 )
    end
    return nothing
end

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


    lap!(r,P,mesh)
    r[ix,iy,iz] = RHS.parent - r[ix,iy,iz]
    update_borders!(r,mesh,par_env)
    pressure_BC!(r,mesh,par_env)
    p = copy(r)
    rsold = parallel_sum(r[ix,iy,iz].^2,par_env,recvProcs="all")
    rsnew = 0.0
    for iter = 1:length(RHS)
        #println(" ===================== Iteration $i ===================")
        #printArray("p",p[gx,gy,gz],par_env)
        lap!(Ap,p,mesh)
        #printArray("Ap",Ap[ix,iy,iz],par_env)

        alpha = rsold / parallel_sum(p[ix,iy,iz] .* Ap[ix,iy,iz],par_env,recvProcs="all")
        P[:,:,:] += alpha * p
        r -= alpha * Ap
        rsnew = parallel_sum(r[ix,iy,iz].^2,par_env,recvProcs="all")
        if sqrt(rsnew) < tol
            return iter
        end
        p = r + (rsnew / rsold) * p
        update_borders!(p,mesh,par_env)
        pressure_BC!(p,mesh,par_env)   
        rsold = rsnew
    end
    isroot && println("Failed to converged Poisson equation rsnew = $rsnew")

    return nothing
end



""" 
Apply BC's on pressure
"""
function pressure_BC!(A,mesh,par_env)

    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack nprocx,nprocy,nprocz,irankx,iranky,irankz = par_env

    irankx == 0        ? A[imin_-1,:,:]=A[imin_,:,:] : nothing # Left 
    irankx == nprocx-1 ? A[imax_+1,:,:]=A[imax_,:,:] : nothing # Right
    iranky == 0        ? A[:,jmin_-1,:]=A[:,jmin_,:] : nothing # Bottom
    iranky == nprocy-1 ? A[:,jmax_+1,:]=A[:,jmax_,:] : nothing # Top
    irankz == 0        ? A[:,:,kmin_-1]=A[:,:,kmin_] : nothing # Back
    irankz == nprocz-1 ? A[:,:,kmax_+1]=A[:,:,kmax_] : nothing # Front
    
    return nothing
end