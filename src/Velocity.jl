

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

# Divergence computed in entire domain
function divergence!(divg,uf,vf,wf,dt,band,verts,tets,param,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    fill!(divg,0.0)
    @loop param for  k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
        divg[i,j,k] = divg_cell(i,j,k,uf,vf,wf,band,dt,verts,tets,param,mesh)
    end
    
    return nothing
end

# Divergence in a computational cell ∇⋅u 
# - Main divergence operator used throughout code
function divg_cell(i,j,k,uf,vf,wf,band,dt,verts,tets,param,mesh)
    @unpack dx,dy,dz = mesh
    @unpack pressure_scheme = param

    if pressure_scheme == "semi-lagrangian" && abs(band[i,j,k]) <= 1
        # Calculate divergence with semi-Lagrangian
        tetsign = cell2tets!(verts,tets,i,j,k,param,mesh,project_verts=true,uf=uf,vf=vf,wf=wf,dt=dt)
        v2 = dx*dy*dz
        v1 = tetsign * tets_vol(tets)
        divg = (v2-v1) / v2 / dt
    else
        # Calculate divergence with finite volume/difference (and SL at edge of band)
        if pressure_scheme == "semi-lagrangian" && band[i-1,j,k] <= 1
            Fxm = SLfluxVol(1,i  ,j,k,verts,tets,uf,vf,wf,dt,param,mesh)/dt
        else
            Fxm = dy*dz*uf[i,j,k]
        end
        if pressure_scheme == "semi-lagrangian" && band[i+1,j,k] <= 1
            Fxp = SLfluxVol(1,i+1,j,k,verts,tets,uf,vf,wf,dt,param,mesh)/dt
        else
            Fxp = dy*dz*uf[i+1,j,k]
        end
        if pressure_scheme == "semi-lagrangian" && band[i,j-1,k] <= 1
            Fym = SLfluxVol(2,i,j  ,k,verts,tets,uf,vf,wf,dt,param,mesh)/dt
        else
            Fym = dx*dz*vf[i,j,k]
        end
        if pressure_scheme == "semi-lagrangian" && band[i,j+1,k] <= 1
            Fyp = SLfluxVol(2,i,j+1,k,verts,tets,uf,vf,wf,dt,param,mesh)/dt
        else
            Fyp = dx*dz*vf[i,j+1,k]
        end
        if pressure_scheme == "semi-lagrangian" && band[i,j,k-1] <= 1
            Fzm = SLfluxVol(3,i,j,k  ,verts,tets,uf,vf,wf,dt,param,mesh)/dt
        else
            Fzm = dx*dy*wf[i,j,k]
        end
        if pressure_scheme == "semi-lagrangian" && band[i,j,k+1] <= 1
            Fzp = SLfluxVol(3,i,j,k+1,verts,tets,uf,vf,wf,dt,param,mesh)/dt
        else
            Fzp = dx*dy*wf[i,j,k+1]
        end
        divg = (
            Fxp - Fxm +
            Fyp - Fym +
            Fzp - Fzm
        ) / (dx*dy*dz)
    end
    return divg
end 