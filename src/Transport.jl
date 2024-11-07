
function transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,Fx,Fy,Fz,VFnew,Curve,dt,param,mesh,par_env,BC!,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,t,verts,tets,inds,vInds)
    @unpack gravity,pressure_scheme,VFlo,VFhi = param
    @unpack irankx,isroot = par_env
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Create band around interface 
    computeBand!(band,VF,param,mesh,par_env)
    
    # Compute interface normal 
    computeNormal!(nx,ny,nz,VF,param,mesh,par_env)
    
    # Compute PLIC reconstruction 
    computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)

    # Transport velocity and volume fraction 
    fill!(VFnew,0.0)
    fill!(Curve,0.0)

    # Preallocate for cutTet
    nLevel=100
    nThread = Threads.nthreads()
    vert = Array{Float64}(undef, 3, 8,nLevel,nThread)
    vert_ind = Array{Int64}(undef, 3, 8, 2,nLevel,nThread)
    d = Array{Float64}(undef, 4,nThread)
    newtet = Array{Float64}(undef, 3, 4,nThread)

    # compute surface tension
    fill!(Curve,0.0)
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        compute_curvature!(i,j,k,Curve,VF,nx,ny,nz,param,mesh)
    end

    compute_sf!(sfx,sfy,sfz,VF,Curve,mesh,param)
    # Loop overdomain
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        
        # Calculate inertia near or away from the interface
        if abs(band[i,j,k]) <= 1
            # Semi-Lagrangian near interface 
            # ------------------------------
            # Form projected cell and break into tets using face velocities
            tetsign = cell2tets!(verts,tets,i,j,k,mesh; 
                project_verts=true,uf=uf,vf=vf,wf=wf,dt=dt,
                compute_indices=true,inds=inds,vInds=vInds)

            if pressure_scheme == "finite-difference"
                # Add correction tets 
                error("Fix correction tets for odd/even cell2tets")
                tets,inds = add_correction_tets(tets,inds,i,j,k,uf,vf,wf,dt,mesh)
            end
            
            # Compute VF in semi-Lagrangian cell 
            vol  = 0.0
            vLiq = 0.0
            vU   = 0.0
            vV   = 0.0
            vW   = 0.0
            for tet in eachindex(view(tets,1,1,:))
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
            # Finite-differences for intertia away from interface 
            # --------------------------------------
            # VF (doesn't change)
            VFnew[i,j,k] = VF[i,j,k]
        
            # u: x-velocity
            for ii = i:i+1 # Loop over faces 
                uface = 0.5*(u[ii-1,j,k] + u[ii,j,k])
                Fx[ii,j,k] = dy*dz*( - uf[ii,j,k]*uface ) # uf*uf or uf*uface ???
            end           
            for jj = j:j+1 # Loop over faces 
                uface = 0.5*(u[i,jj-1,k] + u[i,jj,k])
                Fy[i,jj,k] = dx*dz*( - vf[i,jj,k]*uface )
            end          
            for kk = k:k+1 # Loop over faces 
                uface = 0.5*(u[i,j,kk-1] + u[i,j,kk])
                Fz[i,j,kk] = dx*dy*( - wf[i,j,kk]*uface )
            end
            us[i,j,k] = u[i,j,k] + dt/(dx*dy*dz) * (
                    Fx[i+1,j,k] - Fx[i,j,k] +
                    Fy[i,j+1,k] - Fy[i,j,k] + 
                    Fz[i,j,k+1] - Fz[i,j,k]
                )

            # v: y-velocity           
            for ii = i:i+1 # Loop over faces 
                vface = 0.5*(v[ii-1,j,k] + v[ii,j,k])
                Fx[ii,j,k] = dy*dz*( - uf[ii,j,k]*vface ) # uf*uf or uf*uface ???
            end           
            for jj = j:j+1 # Loop over faces 
                vface = 0.5*(v[i,jj-1,k] + v[i,jj,k])
                Fy[i,jj,k] = dx*dz*( - vf[i,jj,k]*vface )
            end       
            for kk = k:k+1 # Loop over faces 
                vface = 0.5*(v[i,j,kk-1] + v[i,j,kk])
                Fz[i,j,kk] = dx*dy*( - wf[i,j,kk]*vface )
            end
            vs[i,j,k] = v[i,j,k] + dt/(dx*dy*dz) * (
                    Fx[i+1,j,k] - Fx[i,j,k] +
                    Fy[i,j+1,k] - Fy[i,j,k] + 
                    Fz[i,j,k+1] - Fz[i,j,k]
                )

            # w: z-velocity
            for ii = i:i+1 # Loop over faces 
                wface = 0.5*(w[ii-1,j,k] + w[ii,j,k])
                Fx[ii,j,k] = dy*dz*( - uf[ii,j,k]*wface ) # uf*uf or uf*uface ??
            end       
            for jj = j:j+1 # Loop over faces 
                wface = 0.5*(w[i,jj-1,k] + w[i,jj,k])
                Fy[i,jj,k] = dx*dz*( - vf[i,jj,k]*wface )
            end           
            for kk = k:k+1 # Loop over faces 
                wface = 0.5*(w[i,j,kk-1] + w[i,j,kk])
                Fz[i,j,kk] = dx*dy*( - wf[i,j,kk]*wface )
            end
            ws[i,j,k] = w[i,j,k] + dt/(dx*dy*dz) * (
                    Fx[i+1,j,k] - Fx[i,j,k] +
                    Fy[i,j+1,k] - Fy[i,j,k] + 
                    Fz[i,j,k+1] - Fz[i,j,k]
                )
        end# band conditional
    end

    # Loop overdomain
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # u: x-velocity
        for ii = i:i+1 # Loop over faces 
            dudx = (u[ii,j,k] - u[ii-1,j,k])/dx
            Fx[ii,j,k] = dy*dz*( viscx[ii,j,k]/̂denx[ii,j,k]*dudx ) 
        end
        for jj = j:j+1# Loop over faces 
            dudy = (u[i,jj,k] - u[i,jj-1,k])/dy
            Fy[i,jj,k] = dx*dz*( viscy[i,jj,k]/̂deny[i,jj,k]*dudy )
        end
        for kk = k:k+1# Loop over faces 
            dudz = (u[i,j,kk] - u[i,j,kk-1])/dz
            Fz[i,j,kk] = dx*dy*( viscz[i,j,kk]/̂denz[i,j,kk]*dudz )
        end
        us[i,j,k] = us[i,j,k] + dt/(dx*dy*dz) * (
                Fx[i+1,j,k] - Fx[i,j,k] +
                Fy[i,j+1,k] - Fy[i,j,k] + 
                Fz[i,j,k+1] - Fz[i,j,k]) +
                # dt*sfx[i,j,k]/̂(denx[i,j,k])
                dt*sfx[i,j,k]/̂(0.5*(denx[i+1,j,k]+denx[i,j,k]))

        # v: y-velocity
        for ii = i:i+1 # Loop over faces 
            dvdx = (v[ii,j,k] - v[ii-1,j,k])/dx
            Fx[ii,j,k] = dy*dz*( viscx[ii,j,k]/̂denx[ii,j,k]*dvdx) 
        end
        for jj = j:j+1# Loop over faces 
            dvdy = (v[i,jj,k] - v[i,jj-1,k])/dy
            Fy[i,jj,k] = dx*dz*( viscy[i,jj,k]/̂deny[i,jj,k]*dvdy )
        end
        for kk = k:k+1# Loop over faces 
            dvdz = (v[i,j,kk] - v[i,j,kk-1])/dz
            Fz[i,j,kk] = dx*dy*( viscz[i,j,kk]/̂denz[i,j,kk]*dvdz )
        end
        vs[i,j,k] = vs[i,j,k] + dt/(dx*dy*dz) * (
                Fx[i+1,j,k] - Fx[i,j,k] +
                Fy[i,j+1,k] - Fy[i,j,k] + 
                Fz[i,j,k+1] - Fz[i,j,k]) +
                # dt*(sfy[i,j,k]/̂deny[i,j,k] -gravity)
                dt*(sfy[i,j,k]/̂(0.5*(deny[i,j+1,k]+deny[i,j,k])) - gravity)

        # w: z-velocity
        for ii = i:i+1 # Loop over faces 
            dwdx = (w[ii,j,k] - w[ii-1,j,k])/dx
            Fx[ii,j,k] = dy*dz*(viscx[ii,j,k]/̂denx[ii,j,k]*dwdx ) 
        end
        for jj = j:j+1 # Loop over faces 
            dwdy = (w[i,jj,k] - w[i,jj-1,k])/dy
            Fy[i,jj,k] = dx*dz*( viscy[i,jj,k]/̂deny[i,jj,k]*dwdy )
        end
        for kk = k:k+1# Loop over faces 
            dwdz = (w[i,j,kk] - w[i,j,kk-1])/dz
            Fz[i,j,kk] = dx*dy*( viscz[i,j,kk]/̂denz[i,j,kk]*dwdz )
        end
        ws[i,j,k] = ws[i,j,k] + dt/(dx*dy*dz) * (
                Fx[i+1,j,k] - Fx[i,j,k] +
                Fy[i,j+1,k] - Fy[i,j,k] + 
                Fz[i,j,k+1] - Fz[i,j,k]) +
                # dt*sfz[i,j,k]/̂denz[i,j,k] 
                dt*sfz[i,j,k]/̂(0.5*(denz[i,j,k+1]+denz[i,j,k]))
    end # Domain loop

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