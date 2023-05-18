function transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,Fx,Fy,Fz,VFnew,Curve,dt,param,mesh,par_env,BC!)
    @unpack rho_liq,mu = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Create band around interface 
    computeBand!(band,VF,param,mesh,par_env)
    
    # Compute interface normal 
    computeNormal!(nx,ny,nz,VF,param,mesh,par_env)
    
    # Compute PLIC reconstruction 
    computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)

    # # Compute surface tension using curvature and normal vectors
    # compute_curvature!(Curve,VF,param,mesh)
    # # println(nx)
    # # println(Curve)

    # #maybe use temp arrays
    # sfx = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    # sfy = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    # sfz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    # compute_sf!(sfx,sfy,sfz,VF,Curve,mesh,param)

    # Transport velocity and volume fraction 
    fill!(VFnew,0.0)

    # Preallocate for cutTet
    nLevel=100
    nThread = Threads.nthreads()
    vert = Array{Float64}(undef, 3, 8,nLevel,nThread)
    vert_ind = Array{Int64}(undef, 3, 8, 2,nLevel,nThread)
    d = Array{Float64}(undef, 4,nThread)
    newtet = Array{Float64}(undef, 3, 4,nThread)

    #need to introduce a way to determine whether gas or liquid
    #we can determine whether we or in the bubble dependent on the liquid volume fraction
    rho = rho_liq


    #need to introduce gravity term into the transport





    # Loop overdomain
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_

        # #need a method to determine the fluid properties along the interface
        # rho = rho_liq*VF[i,j,k] +rho_gas*(1-VF[i,j,k])
        # mu = 1/(VF[i,j,k]/mu_liq+((1-VF[i,j,k])/mu_gas))


        # Calculate inertia near or away from the interface
        # Check if near interface
        if abs(band[i,j,k]) <= 1

            compute_curvature!(i,j,k,Curve,VF,nx,ny,nz,param,mesh)

            # Semi-Lagrangian near interface 
            # ------------------------------
            # From projected cell and break into tets using face velocities
            tets,inds = cell2tets_withProject_uvwf(i,j,k,uf,vf,wf,dt,mesh)
            
            # Compute VF in semi-Lagrangian cell 
            vol  = 0.0
            vLiq = 0.0
            vU   = 0.0
            vV   = 0.0
            vW   = 0.0
            for tet=1:5
                tetVol, tetvLiq, tetvU, tetvV, tetvW, maxlvl = cutTet(tets[:,:,tet],inds[:,:,tet],
                                    u,v,w,
                                    false,false,false,nx,ny,nz,D,mesh,
                                    1,vert,vert_ind,d,newtet)
                vol += tetVol
                vLiq += tetvLiq
                vU   += tetvU
                vV   += tetvV
                vW   += tetvW
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
            fill!(Fx,0.0) 
            for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 
                uface = 0.5*(u[i-1,j,k] + u[i,j,k])
                Fx[i,j,k] = dy*dz*( - uf[i,j,k]*uface ) # uf*uf or uf*uface ???
            end
            fill!(Fy,0.0)
            for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
                uface = 0.5*(u[i,j-1,k] + u[i,j,k])
                Fy[i,j,k] = dx*dz*( - vf[i,j,k]*uface )
            end
            fill!(Fz,0.0)
            for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
                uface = 0.5*(u[i,j,k-1] + u[i,j,k])
                Fz[i,j,k] = dx*dy*( - wf[i,j,k]*uface )
            end
            for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
                us[i,j,k] = u[i,j,k] + dt/(dx*dy*dz) * (
                    Fx[i+1,j,k] - Fx[i,j,k] +
                    Fy[i,j+1,k] - Fy[i,j,k] + 
                    Fz[i,j,k+1] - Fz[i,j,k]
                )
            end

            # v: y-velocity
            fill!(Fx,0.0)
            for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 
                vface = 0.5*(v[i-1,j,k] + v[i,j,k])
                Fx[i,j,k] = dy*dz*( - uf[i,j,k]*vface ) # uf*uf or uf*uface ???
            end
            fill!(Fy,0.0)
            for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
                vface = 0.5*(v[i,j-1,k] + v[i,j,k])
                Fy[i,j,k] = dx*dz*( - vf[i,j,k]*vface  )
            end
            fill!(Fz,0.0)
            for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
                vface = 0.5*(v[i,j,k-1] + v[i,j,k])
                Fz[i,j,k] = dx*dy*( - wf[i,j,k]*vface  )
            end
            for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
                vs[i,j,k] = v[i,j,k] + dt/(dx*dy*dz) * (
                    Fx[i+1,j,k] - Fx[i,j,k] +
                    Fy[i,j+1,k] - Fy[i,j,k] + 
                    Fz[i,j,k+1] - Fz[i,j,k]
                )
            end


            # w: z-velocity
            fill!(Fx,0.0)
            for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 
                wface = 0.5*(w[i-1,j,k] + w[i,j,k])
                Fx[i,j,k] = dy*dz*( - uf[i,j,k]*wface  ) # uf*uf or uf*uface ???
            end
            fill!(Fy,0.0)
            for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
                wface = 0.5*(w[i,j-1,k] + w[i,j,k])
                Fy[i,j,k] = dx*dz*( - vf[i,j,k]*wface  )
            end
            fill!(Fz,0.0)
            for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
                wface = 0.5*(w[i,j,k-1] + w[i,j,k])
                Fz[i,j,k] = dx*dy*( - wf[i,j,k]*wface )
            end
            for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
                ws[i,j,k] = w[i,j,k] + dt/(dx*dy*dz) * (
                    Fx[i+1,j,k] - Fx[i,j,k] +
                    Fy[i,j+1,k] - Fy[i,j,k] + 
                    Fz[i,j,k+1] - Fz[i,j,k]
                )
            end
        end #end band conditional
                
        ## Need to introduce viscous and surface tension effects(how to add surface tension effects)
        sfx = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
        sfy = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
        sfz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
        fill!(Fx,0.0) 
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 
            dudx = (u[i,j,k] - u[i-1,j,k])/dx
            Fx[i,j,k] = dy*dz*( mu/rho*dudx ) 
        end
        fill!(Fy,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
            dudy = (u[i,j,k] - u[i,j-1,k])/dy
            Fy[i,j,k] = dx*dz*( mu/rho*dudy )
        end
        fill!(Fz,0.0)
        for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
            dudz = (u[i,j,k] - u[i,j,k-1])/dz
            Fz[i,j,k] = dx*dy*( mu/rho*dudz )
        end
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            us[i,j,k] = us[i,j,k] + dt/(dx*dy*dz) * (
                Fx[i+1,j,k] - Fx[i,j,k] +
                Fy[i,j+1,k] - Fy[i,j,k] + 
                Fz[i,j,k+1] - Fz[i,j,k] -
                sfx[i,j,k]

            )
        end

        # v: y-velocity
        fill!(Fx,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 
            dvdx = (v[i,j,k] - v[i-1,j,k])/dx
            Fx[i,j,k] = dy*dz*( mu/rho*dvdx) 
        end
        fill!(Fy,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
            dvdy = (v[i,j,k] - v[i,j-1,k])/dy
            Fy[i,j,k] = dx*dz*( mu/rho*dvdy )
        end
        fill!(Fz,0.0)
        for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
            dvdz = (v[i,j,k] - v[i,j,k-1])/dz
            Fz[i,j,k] = dx*dy*( mu/rho*dvdz )
        end
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            vs[i,j,k] = vs[i,j,k] + dt/(dx*dy*dz) * (
                Fx[i+1,j,k] - Fx[i,j,k] +
                Fy[i,j+1,k] - Fy[i,j,k] + 
                Fz[i,j,k+1] - Fz[i,j,k] -
                sfy[i,j,k] 
            )
        end


        # w: z-velocity
        fill!(Fx,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 

            dwdx = (w[i,j,k] - w[i-1,j,k])/dx
            Fx[i,j,k] = dy*dz*(mu/rho*dwdx ) # uf*uf or uf*uface ???
        end
        fill!(Fy,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
            dwdy = (w[i,j,k] - w[i,j-1,k])/dy
            Fy[i,j,k] = dx*dz*( mu/rho*dwdy )
        end
        fill!(Fz,0.0)
        for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
            dwdz = (w[i,j,k] - w[i,j,k-1])/dz
            Fz[i,j,k] = dx*dy*( mu/rho*dwdz )
        end
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            ws[i,j,k] = ws[i,j,k] + dt/(dx*dy*dz) * (
                Fx[i+1,j,k] - Fx[i,j,k] +
                Fy[i,j+1,k] - Fy[i,j,k] + 
                Fz[i,j,k+1] - Fz[i,j,k] -
                sfz[i,j,k]
            )
        end
        
    end # Domain loop

    # Finish updating VF 
    VF .= VFnew

    # Apply boundary conditions
    Neumann!(VF,mesh,par_env)
    BC!(us,vs,ws,mesh,par_env)

    # Update Processor boundaries (overwrites BCs if periodic)
    update_borders!(VF,mesh,par_env)
    update_borders!(us,mesh,par_env)
    update_borders!(vs,mesh,par_env)
    update_borders!(ws,mesh,par_env)

    return nothing
end