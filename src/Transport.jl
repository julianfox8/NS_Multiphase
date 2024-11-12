
function transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,Fux,Fuy,Fuz,Fvx,Fvy,Fvz,Fwx,Fwy,Fwz,VFnew,Curve,dt,param,mesh,par_env,BC!,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,t,verts,tets,inds,vInds)
    @unpack gravity,pressure_scheme,VFlo,VFhi = param
    @unpack irankx,isroot = par_env
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    
    # Compute interface normal 
    computeNormal!(nx,ny,nz,VF,param,mesh,par_env)
    
    # Compute PLIC reconstruction 
    computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)

    # Preallocate for cutTet
    nLevel=100
    nThread = Threads.nthreads()
    vert = Array{Float64}(undef, 3, 8,nLevel,nThread)
    vert_ind = Array{Int64}(undef, 3, 8, 2,nLevel,nThread)
    d = Array{Float64}(undef, 4,nThread)
    newtet = Array{Float64}(undef, 3, 4,nThread)

    # Compute interface curvature
    fill!(Curve,0.0)
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        compute_curvature!(i,j,k,Curve,VF,nx,ny,nz,param,mesh)
    end

    # Compute surface tension force
    compute_sf!(sfx,sfy,sfz,VF,Curve,param,mesh)

    ########################
    #    Convective term   #
    ########################

    # Compute convective fluxes : x faces
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        uface = 0.5*(u[i-1,j,k] + u[i,j,k])
        vface = 0.5*(v[i-1,j,k] + v[i,j,k])
        wface = 0.5*(w[i-1,j,k] + w[i,j,k])
        if abs(band[i-1,j,k]) <= 1 && abs(band[i,j,k]) <= 1 
            # No flux needed : Both cells transported with SL
            Fux[i,j,k] = 0
            Fvx[i,j,k] = 0
            Fwx[i,j,k] = 0
        elseif abs(band[i-1,j,k]) <=1 || abs(band[i,j,k]) <= 1
            # Compute flux with SL : One of cells transported with SL
            vol = SLfluxVol(1,i,j,k,verts,tets,uf,vf,wf,dt,param,mesh)/dt
            Fux[i,j,k] = -uface * vol
            Fvx[i,j,k] = -vface * vol
            Fwx[i,j,k] = -wface * vol
        else
            # Compute flux with FD : Neither cell transported with SL 
            Fux[i,j,k] = -uface * dy*dz*uf[i,j,k]
            Fvx[i,j,k] = -vface * dy*dz*uf[i,j,k]
            Fwx[i,j,k] = -wface * dy*dz*uf[i,j,k]
        end
    end

    # Compute convective fluxes : y faces
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        uface = 0.5*(u[i,j-1,k] + u[i,j,k])
        vface = 0.5*(v[i,j-1,k] + v[i,j,k])
        wface = 0.5*(w[i,j-1,k] + w[i,j,k])
        if abs(band[i,j-1,k]) <= 1 && abs(band[i,j,k]) <= 1 
            # No flux needed : Both cells transported with SL
            Fuy[i,j,k] = 0
            Fvy[i,j,k] = 0
            Fwy[i,j,k] = 0
        elseif abs(band[i,j-1,k]) <=1 || abs(band[i,j,k]) <= 1
            # Compute flux with SL : One of cells transported with SL
            vol = SLfluxVol(2,i,j,k,verts,tets,uf,vf,wf,dt,param,mesh)/dt
            Fuy[i,j,k] = -uface * vol
            Fvy[i,j,k] = -vface * vol
            Fwy[i,j,k] = -wface * vol
        else
            # Compute flux with FD : Neither cell transported with SL 
            Fuy[i,j,k] = -uface * dx*dz*vf[i,j,k]
            Fvy[i,j,k] = -vface * dx*dz*vf[i,j,k]
            Fwy[i,j,k] = -wface * dx*dz*vf[i,j,k]
        end
    end
    
    # Compute convective fluxes : z faces
    @loop param for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        uface = 0.5*(u[i,j,k-1] + u[i,j,k])
        vface = 0.5*(v[i,j,k-1] + v[i,j,k])
        wface = 0.5*(w[i,j,k-1] + w[i,j,k])
        if abs(band[i,j,k-1]) <= 1 && abs(band[i,j,k]) <= 1 
            # No flux needed : Both cells transported with SL
            Fuz[i,j,k] = 0
            Fvz[i,j,k] = 0
            Fwz[i,j,k] = 0
        elseif abs(band[i,j,k-1]) <=1 || abs(band[i,j,k]) <= 1
            # Compute flux with SL : One of cells transported with SL
            vol = SLfluxVol(3,i,j,k,verts,tets,uf,vf,wf,dt,param,mesh)/dt
            Fuz[i,j,k] = -uface * vol
            Fvz[i,j,k] = -vface * vol
            Fwz[i,j,k] = -wface * vol
        else
            # Compute flux with FD : Neither cell transported with SL 
            Fuz[i,j,k] = -uface * dx*dy*wf[i,j,k]
            Fvz[i,j,k] = -vface * dx*dy*wf[i,j,k]
            Fwz[i,j,k] = -wface * dx*dy*wf[i,j,k]
        end
    end
    
    # Perform transport
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        
        # Calculate inertia near or away from the interface
        if abs(band[i,j,k]) <= 1
            # Semi-Lagrangian near interface 
            # ------------------------------
            # Form projected cell and break into tets using face velocities
            ntets = 5
            tetsign = cell2tets!(verts,tets,i,j,k,param,mesh; 
                project_verts=true,uf=uf,vf=vf,wf=wf,dt=dt,
                compute_indices=true,inds=inds,vInds=vInds)

            if pressure_scheme == "finite-difference"
                # Add correction tets 
                tets,inds,ntets = add_correction_tets(ntets,verts,tets,inds,i,j,k,uf,vf,wf,dt,param,mesh)
            end
            
            # Compute VF in semi-Lagrangian cell 
            vol  = 0.0
            vLiq = 0.0
            vU   = 0.0
            vV   = 0.0
            vW   = 0.0
            for tet in eachindex(view(tets,1,1,1:ntets))
                tetVol, tetvLiq, tetvU, tetvV, tetvW, maxlvl = cutTet(tets[:,:,tet],inds[:,:,tet],
                                    u,v,w,
                                    false,false,false,nx,ny,nz,D,mesh,
                                    1,vert,vert_ind,d,newtet)
                vol  += tetsign * tetVol
                vLiq += tetsign * tetvLiq
                vU   += tetsign * tetvU
                vV   += tetsign * tetvV
                vW   += tetsign * tetvW
            end
            VFnew[i,j,k] = vLiq/vol
            us[i,j,k] = vU/vol
            vs[i,j,k] = vV/vol
            ws[i,j,k] = vW/vol
        else
            # Apply fluxes away from interface
            # --------------------------------
            # VF (doesn't change)
            VFnew[i,j,k] = VF[i,j,k]
        
            # u: x-velocity
            us[i,j,k] = u[i,j,k] + dt/(dx*dy*dz) * (
                    Fux[i+1,j,k] - Fux[i,j,k] +
                    Fuy[i,j+1,k] - Fuy[i,j,k] + 
                    Fuz[i,j,k+1] - Fuz[i,j,k] )
            # v: y-velocity           
            vs[i,j,k] = v[i,j,k] + dt/(dx*dy*dz) * (
                    Fvx[i+1,j,k] - Fvx[i,j,k] +
                    Fvy[i,j+1,k] - Fvy[i,j,k] + 
                    Fvz[i,j,k+1] - Fvz[i,j,k] )

            # w: z-velocity
            ws[i,j,k] = w[i,j,k] + dt/(dx*dy*dz) * (
                    Fwx[i+1,j,k] - Fwx[i,j,k] +
                    Fwy[i,j+1,k] - Fwy[i,j,k] + 
                    Fwz[i,j,k+1] - Fwz[i,j,k] )
        end# band conditional
    end

    #######################################
    # Viscous & Surface Tension & Gravity #
    #######################################

    # Compute viscous fluxes : x faces
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        dudx = (u[i,j,k] - u[i-1,j,k])/dx
        dvdx = (v[i,j,k] - v[i-1,j,k])/dx
        dwdx = (w[i,j,k] - w[i-1,j,k])/dx
        Fux[i,j,k] = dy*dz*( viscx[i,j,k]/̂denx[i,j,k]*dudx ) 
        Fvx[i,j,k] = dy*dz*( viscx[i,j,k]/̂denx[i,j,k]*dvdx ) 
        Fwx[i,j,k] = dy*dz*( viscx[i,j,k]/̂denx[i,j,k]*dwdx ) 
    end
    # Compute viscous fluxes : y faces
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        dudy = (u[i,j,k] - u[i,j-1,k])/dy
        dvdy = (v[i,j,k] - v[i,j-1,k])/dy
        dwdy = (w[i,j,k] - w[i,j-1,k])/dy
        Fuy[i,j,k] = dx*dz*( viscy[i,j,k]/̂deny[i,j,k]*dudy ) 
        Fvy[i,j,k] = dx*dz*( viscy[i,j,k]/̂deny[i,j,k]*dvdy ) 
        Fwy[i,j,k] = dx*dz*( viscy[i,j,k]/̂deny[i,j,k]*dwdy ) 
    end
    # Compute viscous fluxes : z faces
    @loop param for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        dudz = (u[i,j,k] - u[i,j,k-1])/dz
        dvdz = (v[i,j,k] - v[i,j,k-1])/dz
        dwdz = (w[i,j,k] - w[i,j,k-1])/dz
        Fuz[i,j,k] = dx*dy*( viscz[i,j,k]/̂denz[i,j,k]*dudz ) 
        Fvz[i,j,k] = dx*dy*( viscz[i,j,k]/̂denz[i,j,k]*dvdz ) 
        Fwz[i,j,k] = dx*dy*( viscz[i,j,k]/̂denz[i,j,k]*dwdz ) 
    end
    
    # Apply viscous fluxes, surface tension force, and gravitational force
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # u: x-velocity
        us[i,j,k] = us[i,j,k] + dt/(dx*dy*dz) * (
                Fux[i+1,j,k] - Fux[i,j,k] +
                Fuy[i,j+1,k] - Fuy[i,j,k] + 
                Fuz[i,j,k+1] - Fuz[i,j,k]) +
                dt*sfx[i,j,k]/̂(0.5*(denx[i+1,j,k]+denx[i,j,k]))
        # v: y-velocity           
        vs[i,j,k] = vs[i,j,k] + dt/(dx*dy*dz) * (
                Fvx[i+1,j,k] - Fvx[i,j,k] +
                Fvy[i,j+1,k] - Fvy[i,j,k] + 
                Fvz[i,j,k+1] - Fvz[i,j,k]) +
                dt*(sfy[i,j,k]/̂(0.5*(deny[i,j+1,k]+deny[i,j,k])) - gravity)
        # w: z-velocity
        ws[i,j,k] = ws[i,j,k] + dt/(dx*dy*dz) * (
                Fwx[i+1,j,k] - Fwx[i,j,k] +
                Fwy[i,j+1,k] - Fwy[i,j,k] + 
                Fwz[i,j,k+1] - Fwz[i,j,k]) +
                dt*sfz[i,j,k]/̂(0.5*(denz[i,j,k+1]+denz[i,j,k]))
    end

    # Finish updating VF 
    VF .= VFnew
    # Apply boundary conditions
    Neumann!(VF,mesh,par_env)
    BC!(us,vs,ws,t,mesh,par_env)

    # Update Processor boundaries (overwrites BCs if periodic)
    update_VF_borders!(VF,mesh,par_env)
    update_borders!(us,mesh,par_env)
    update_borders!(vs,mesh,par_env)
    update_borders!(ws,mesh,par_env)

    return nothing
end