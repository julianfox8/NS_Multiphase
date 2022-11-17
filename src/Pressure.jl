
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

    poisson_solve!(P,RHS,param,mesh,par_env)

    return nothing
end

function poisson_solve!(P,RHS,param,mesh,par_env)

    GaussSeidel!(P,RHS,param,mesh,par_env)

    return nothing
end

"""
Serial GaussSeidel Poisson Solver
"""
function GaussSeidel!(P,RHS,param,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack isroot = par_env
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
        max_update < 1e-10 && return iter # Converged
        # Check if hit max iteration
        if iter == maxIter 
            isroot && println("Failed to converged Poisson equation max_upate = $max_update")
            return nothing 
        end
    end
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