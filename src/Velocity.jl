
function corrector!(uf,vf,wf,P,dt,param,mesh)
    @unpack rho_liq = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1
        dp_dx = ( P[i,j,k] - P[i-1,j,k] )/(dx)
        uf[i,j,k] = uf[i,j,k] - dt/rho_liq * dp_dx
    end
    for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_
        dp_dy = ( P[i,j,k] - P[i,j-1,k] )/(dy)
        vf[i,j,k] = vf[i,j,k] - dt/rho_liq * dp_dy
    end
    for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_
        dp_dz = ( P[i,j,k] - P[i,j,k-1] )/(dz)
        wf[i,j,k] = wf[i,j,k] - dt/rho_liq * dp_dz
    end

    return nothing
end

function interpolateFace!(u,v,w,uf,vf,wf,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1
        uf[i,j,k] = 0.5*(u[i-1,j,k] + u[i,j,k])
    end
    for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_
        vf[i,j,k] = 0.5*(v[i,j-1,k] + v[i,j,k])
    end
    for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_
        wf[i,j,k] = 0.5*(w[i,j,k-1] + w[i,j,k])
    end
    return nothing
end

function interpolateCenter!(u,v,w,us,vs,ws,uf,vf,wf,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # Copy BCs
    u[:,:,:]=us[:,:,:]
    v[:,:,:]=vs[:,:,:]
    w[:,:,:]=ws[:,:,:]

    # Perform interpolation
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
        u[i,j,k] = 0.5*(uf[i,j,k] + uf[i+1,j,k])
    end
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
        v[i,j,k] = 0.5*(vf[i,j,k] + vf[i,j+1,k])
    end
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
        w[i,j,k] = 0.5*(wf[i,j,k] + wf[i,j,k+1])
    end
    return nothing
end

function divergence(uf,vf,wf,dt,band,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    
    divg = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_

        # Check if near interface
        if abs(band[i,j,k]) <= 1
            # Calculate divergence with semi-lagrangian scheme
            tets, inds = cell2tets_withProject_uvwf(i,j,k,uf,vf,wf,dt,mesh)
            if any(isnan,tets)
                error("Nan in tets at ", i,j,k)
            end
            v2 = dx*dy*dz
            v1 = tets_vol(tets)
            divg[i,j,k] = (v2-v1) /̂ v2 /̂ dt

        else
            # Calculate divergence with finite differnce
            du_dx = ( uf[i+1,j,k] - uf[i,j,k] )/(dx)
            dv_dy = ( vf[i,j+1,k] - vf[i,j,k] )/(dy)
            dw_dz = ( wf[i,j,k+1] - wf[i,j,k] )/(dz)
                
            # Divergence
            divg[i,j,k] = du_dx + dv_dy + dw_dz
        end
    end

    return divg
end

function semi_lag_divergence(uf,vf,wf,dt,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    
    divg = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        tets, inds = cell2tets_withProject_uvwf(i,j,k,uf,vf,wf,dt,mesh)
        if any(isnan,tets)
            error("Nan in tets at ", i,j,k)
        end
        v2 = dx*dy*dz
        v1 = tets_vol(tets)
        divg[i,j,k] = (v2-v1) /̂ v2 /̂ dt
    end

    return divg
end

# function semi_lag_divergence(uf,vf,wf,dt,mesh)
#     tets, inds = cell2tets_withProject_uvwf(i,j,k,uf,vf,wf,dt,mesh)
#     if any(isnan,tets)
#         error("Nan in tets at ", i,j,k)
#     end
#     v2 = dx*dy*dz
#     v1 = tets_vol(tets)
#     divg[i,j,k] = (v2-v1) /̂ v2 /̂ dt

#     return divg
# end