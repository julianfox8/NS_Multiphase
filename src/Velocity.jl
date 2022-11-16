function predictor!(us,vs,ws,u,v,w,dt,param,mesh,par_env,mask)
    @unpack rho,mu = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh


    # # u: x-velocity
    # for i in imin_:imax_+1, j in jmin_:jmax_+1, k in kmin_:kmax_+1 # Loop over faces 
    #     if mask[i-1,j,k] == false && mask[i,j,k] == false
    #         uface = 0.5*(u[i-1]+u[i])
    #         dudx = (u[i] - u[i-1])/dx
    #         Fx = - uface*uface + mu/rho
    #     end
    # end
    

    # Predictor step for u
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        if !mask[i,j,k]
            # Derivatives 
            du_dx   = ( u[i+1,j,k] - u[i-1,j,k] )/(2dx)
            du_dy   = ( u[i,j+1,k] - u[i,j-1,k] )/(2dy)
            du_dz   = ( u[i,j,k+1] - u[i,j,k-1] )/(2dz)
            dv_dx   = ( v[i+1,j,k] - v[i-1,j,k] )/(2dx)
            dv_dy   = ( v[i,j+1,k] - v[i,j-1,k] )/(2dy)
            dv_dz   = ( v[i,j,k+1] - v[i,j,k-1] )/(2dz)
            dw_dx   = ( w[i+1,j,k] - w[i-1,j,k] )/(2dx)
            dw_dy   = ( w[i,j+1,k] - w[i,j-1,k] )/(2dy)
            dw_dz   = ( w[i,j,k+1] - w[i,j,k-1] )/(2dz)
            d²u_dx² = ( u[i-1,j,k] - 2u[i,j,k] + u[i+1,j,k] )/(dx^2)
            d²u_dy² = ( u[i,j-1,k] - 2u[i,j,k] + u[i,j+1,k] )/(dy^2)
            d²u_dz² = ( u[i,j,k-1] - 2u[i,j,k] + u[i,j,k+1] )/(dz^2)
            d²v_dx² = ( v[i-1,j,k] - 2v[i,j,k] + v[i+1,j,k] )/(dx^2)
            d²v_dy² = ( v[i,j-1,k] - 2v[i,j,k] + v[i,j+1,k] )/(dy^2)
            d²v_dz² = ( v[i,j,k-1] - 2v[i,j,k] + v[i,j,k+1] )/(dz^2)
            d²w_dx² = ( w[i-1,j,k] - 2w[i,j,k] + w[i+1,j,k] )/(dx^2)
            d²w_dy² = ( w[i,j-1,k] - 2w[i,j,k] + w[i,j+1,k] )/(dy^2)
            d²w_dz² = ( w[i,j,k-1] - 2w[i,j,k] + w[i,j,k+1] )/(dz^2)

            # Predictor step for u
            us[i,j,k] = u[i,j,k] + dt * (
                - ( u[i,j,k]*du_dx + v[i,j,k]*du_dy + w[i,j,k]*du_dz ) 
                + mu/rho * ( d²u_dx² + d²u_dy² + d²u_dz²)
            )

            # Predictor step for v
            vs[i,j,k] = v[i,j,k] + dt * (
                - ( u[i,j,k]*dv_dx + v[i,j,k]*dv_dy + w[i,j,k]*dv_dz )
                + mu/rho * ( d²v_dx² + d²v_dy² + d²v_dz²)
            )

            # Predictor step for w
            ws[i,j,k] = w[i,j,k] + dt * (
                - ( u[i,j,k]*dw_dx + v[i,j,k]*dw_dy + w[i,j,k]*dw_dz )
                + mu/rho * ( d²w_dx² + d²w_dy² + d²w_dz²)
            )
        end
    end

    # Update Processor boundaries
    update_borders!(us,mesh,par_env)
    update_borders!(vs,mesh,par_env)
    update_borders!(ws,mesh,par_env)

    return nothing
end

function corrector!(uf,vf,wf,P,dt,param,mesh,mask)
    @unpack rho = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        # Derivatives 
        dp_dx = ( P[i,j,k] - P[i-1,j,k] )/(dx)
        dp_dy = ( P[i,j,k] - P[i,j-1,k] )/(dy)
        dp_dz = ( P[i,j,k] - P[i,j,k-1] )/(dz)
                    
        # Corrector step 
        uf[i,j,k] = uf[i,j,k] - dt/rho * dp_dx
        vf[i,j,k] = vf[i,j,k] - dt/rho * dp_dy
        wf[i,j,k] = wf[i,j,k] - dt/rho * dp_dz
    end

    return nothing
end

function interpolateFace!(u,v,w,uf,vf,wf,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    for i in imin_:imax_+1, j in jmin_:jmax_, k in kmin_:kmax_
        uf[i,j,k] = 0.5*(u[i-1,j,k] + u[i,j,k])
    end
    for i in imin_:imax_, j in jmin_:jmax_+1, k in kmin_:kmax_
        vf[i,j,k] = 0.5*(v[i,j-1,k] + v[i,j,k])
    end
    for i in imin_:imax_, j in jmin_:jmax_, k in kmin_:kmax_+1
        wf[i,j,k] = 0.5*(w[i,j,k-1] + w[i,j,k])
    end
    return nothing
end

function interpolateCenter!(u,v,w,uf,vf,wf,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    for i in imin_:imax_, j in jmin_:jmax_, k in kmin_:kmax_
        u[i,j,k] = 0.5*(uf[i,j,k] + uf[i+1,j,k])
    end
    for i in imin_:imax_, j in jmin_:jmax_, k in kmin_:kmax_
        v[i,j,k] = 0.5*(vf[i,j,k] + vf[i,j+1,k])
    end
    for i in imin_:imax_, j in jmin_:jmax_, k in kmin_:kmax_
        w[i,j,k] = 0.5*(wf[i,j,k] + wf[i,j,k+1])
    end
    return nothing
end

function divergence(uf,vf,wf,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    
    divg = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        # Deritives 
        du_dx = ( uf[i+1,j,k] - uf[i,j,k] )/(dx)
        dv_dy = ( vf[i,j+1,k] - vf[i,j,k] )/(dy)
        dw_dz = ( wf[i,j,k+1] - wf[i,j,k] )/(dz)
            
        # Divergence
        divg[i,j,k] = du_dx + dv_dy + dw_dz
    end

    return divg
end