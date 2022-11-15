
# Solve Poisson equation: δP form
function pressure_solver!(P,us,vs,ws,dt,param,mesh,par_env)
    @unpack rho = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    RHS = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    sig=0.1
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # Derivatives 
        dus_dx   = ( us[i+1,j,k] - us[i-1,j,k] )/(2dx)
        dvs_dy   = ( vs[i,j+1,k] - vs[i,j-1,k] )/(2dy)
        dws_dz   = ( ws[i,j,k+1] - ws[i,j,k-1] )/(2dz)

        # RHS
        RHS[i,j,k]= rho/dt * ( dus_dx + dvs_dy + dws_dz )
    end

    poisson_solve!(P,RHS,param,mesh,par_env)

    return nothing
end
