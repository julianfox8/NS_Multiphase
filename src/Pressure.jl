
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
