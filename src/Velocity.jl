

function corrector!(uf,vf,wf,P,dt,denx,deny,denz,mesh)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    for k = kmin_-1:kmax_+1, j = jmin_-1:jmax_+1, i = imin_-1:imax_+2
        dp_dx = ( P[i,j,k] - P[i-1,j,k] )/(dx)
        uf[i,j,k] = uf[i,j,k] - dt/denx[i,j,k] * dp_dx
    end
    for k = kmin_-1:kmax_+1, j = jmin_-1:jmax_+2, i = imin_-1:imax_+1
        dp_dy = ( P[i,j,k] - P[i,j-1,k] )/(dy)
        vf[i,j,k] = vf[i,j,k] - dt/deny[i,j,k] * dp_dy
    end
    for k = kmin_-1:kmax_+2, j = jmin_-1:jmax_+1, i = imin_-1:imax_+1
        dp_dz = ( P[i,j,k] - P[i,j,k-1] )/(dz)
        wf[i,j,k] = wf[i,j,k] - dt/denz[i,j,k] * dp_dz
    end

    return nothing
end

function interpolateFace!(u,v,w,uf,vf,wf,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    for k = kmin_-1:kmax_+1, j = jmin_-1:jmax_+1, i = imin_-1:imax_+2
        uf[i,j,k] = 0.5*(u[i-1,j,k] + u[i,j,k])
    end
    for k = kmin_-1:kmax_+1, j = jmin_-1:jmax_+2, i = imin_-1:imax_+1
        vf[i,j,k] = 0.5*(v[i,j-1,k] + v[i,j,k])
    end
    for k = kmin_-1:kmax_+2, j = jmin_-1:jmax_+1, i = imin_-1:imax_+1
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

function divergence!(divg,uf,vf,wf,dt,band,mesh,param,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    fill!(divg,0.0)
    tets_arr = Array{Float64}(undef, 3, 4, 5)
    p = Matrix{Float64}(undef, (3, 8))
    @loop param for  k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
        divg[i,j,k] = divg_cell(i,j,k,uf,vf,wf,band,dt,p,tets_arr,param,mesh)
    end

    return divg
end

# Divergence in a computational cell ∇⋅u
function divg_cell(i,j,k,uf,vf,wf,band,dt,p,tets_arr,param,mesh)
    @unpack dx,dy,dz = mesh
    @unpack pressure_scheme = param

    if pressure_scheme == "semi-lagrangian" &&abs(band[i,j,k]) <= 1
        # Calculate divergence with semi-Lagrangian
        cell2tets_uvwf_A!(i,j,k,uf,vf,wf,dt,p,tets_arr,mesh)
        v2 = dx*dy*dz
        v1 = tets_vol(tets_arr)
        divg = (v2-v1) /̂ v2 /̂ dt

    else
        # Calculate divergence with finite differnce
        du_dx = ( uf[i+1,j,k] - uf[i,j,k] )/(dx)
        dv_dy = ( vf[i,j+1,k] - vf[i,j,k] )/(dy)
        dw_dz = ( wf[i,j,k+1] - wf[i,j,k] )/(dz)
        divg = du_dx + dv_dy + dw_dz
    end
    return divg
end 