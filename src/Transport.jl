
function transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,Fx,Fy,Fz,VFnew,Curve,dt,param,mesh,par_env,BC!,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz)
    @unpack gravity,pressure_scheme = param

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




    fill!(Curve,0.0)
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # if abs(band[i,j,k]) <= 1
        compute_curvature!(i,j,k,Curve,VF,nx,ny,nz,param,mesh)
        # end
    end
    

    # Loop overdomain
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_



        ## //? do we want to move allocation of surface tension here?
        compute_sf!(sfx,sfy,sfz,VF,Curve,mesh,param)

        # Calculate inertia near or away from the interface
        # Check if near interface
        if abs(band[i,j,k]) <= 1
        # if abs(band[i,j,k]) <= 3

            # compute_curvature!(i,j,k,Curve,VF,nx,ny,nz,param,mesh)
            # Semi-Lagrangian near interface 
            # ------------------------------
            # From projected cell and break into tets using face velocities
            tets,inds = cell2tets_withProject_uvwf(i,j,k,uf,vf,wf,dt,mesh)

            if pressure_scheme == "finite-difference"
                # Add correction tets 
                tets,inds = add_correction_tets(tets,inds,i,j,k,uf,vf,wf,dt,mesh)
            end
            
            # Compute VF in semi-Lagrangian cell 
            vol  = 0.0
            vLiq = 0.0
            vU   = 0.0
            vV   = 0.0
            vW   = 0.0
            for tet in eachindex(view(tets,1,1,:))
                # @show tet
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
            # println("made it")

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

        # if VFnew[i,j,k] ==1
        #     println(i,j,k)
        #     # println("u-star with inertia ", us[5,5,1])
        #     println("u-star with inertia ", VFnew[6,5,1])
        # end
    end

    # Loop overdomain
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_

        fill!(Fx,0.0) 
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 
            dudx = (u[i,j,k] - u[i-1,j,k])/dx
            Fx[i,j,k] = dy*dz*( viscx[i,j,k]/̂denx[i,j,k]*dudx ) 
        end
        # if j ==4 && i ==4
        #     println("new term ", Fx[5,5,1])
        # end
        fill!(Fy,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
            dudy = (u[i,j,k] - u[i,j-1,k])/dy
            Fy[i,j,k] = dx*dz*( viscy[i,j,k]/̂deny[i,j,k]*dudy )
        end
        fill!(Fz,0.0)
        for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
            dudz = (u[i,j,k] - u[i,j,k-1])/dz
            Fz[i,j,k] = dx*dy*( viscz[i,j,k]/̂denz[i,j,k]*dudz )
        end

        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            us[i,j,k] = us[i,j,k] + dt/(dx*dy*dz) * (
                Fx[i+1,j,k] - Fx[i,j,k] +
                Fy[i,j+1,k] - Fy[i,j,k] + 
                Fz[i,j,k+1] - Fz[i,j,k]) +
                dt*sfx[i,j,k]
            

        end

        # v: y-velocity
        fill!(Fx,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 
            dvdx = (v[i,j,k] - v[i-1,j,k])/dx
            Fx[i,j,k] = dy*dz*( viscx[i,j,k]/denx[i,j,k]*dvdx) 
        end
        fill!(Fy,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
            dvdy = (v[i,j,k] - v[i,j-1,k])/dy
            Fy[i,j,k] = dx*dz*( viscy[i,j,k]/deny[i,j,k]*dvdy )
        end
        fill!(Fz,0.0)
        for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
            dvdz = (v[i,j,k] - v[i,j,k-1])/dz
            Fz[i,j,k] = dx*dy*( viscz[i,j,k]/denz[i,j,k]*dvdz )
        end
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            vs[i,j,k] = vs[i,j,k] + dt/(dx*dy*dz) * (
                Fx[i+1,j,k] - Fx[i,j,k] +
                Fy[i,j+1,k] - Fy[i,j,k] + 
                Fz[i,j,k+1] - Fz[i,j,k]) +
                dt*(sfy[i,j,k] - gravity)
        end


        # w: z-velocity
        fill!(Fx,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 

            dwdx = (w[i,j,k] - w[i-1,j,k])/dx
            Fx[i,j,k] = dy*dz*(viscx[i,j,k]/denx[i,j,k]*dwdx ) # uf*uf or uf*uface ???
        end
        fill!(Fy,0.0)
        for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
            dwdy = (w[i,j,k] - w[i,j-1,k])/dy
            Fy[i,j,k] = dx*dz*( viscy[i,j,k]/deny[i,j,k]*dwdy )
        end
        fill!(Fz,0.0)
        for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
            dwdz = (w[i,j,k] - w[i,j,k-1])/dz
            Fz[i,j,k] = dx*dy*( viscz[i,j,k]/denz[i,j,k]*dwdz )
        end
 
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            ws[i,j,k] = ws[i,j,k] + dt/(dx*dy*dz) * (
                Fx[i+1,j,k] - Fx[i,j,k] +
                Fy[i,j+1,k] - Fy[i,j,k] + 
                Fz[i,j,k+1] - Fz[i,j,k]) +
                dt*sfz[i,j,k] 
            
        end
        
    end # Domain loop
    # println("u-star with other terms ", us[5,5,1])
    # if VF != VFnew
    #     println(VF)
    #     println(VFnew)
    # end

    # println(Curve)
    # error("check")

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