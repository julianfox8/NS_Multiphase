function predictor!(us,vs,ws,P,u,v,w,Fx,Fy,Fz,dt,param,mesh,par_env,mask)
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
            dP_dx   = ( P[i+1,j,k] - P[i-1,j,k] )/(2dx)
            dP_dy   = ( P[i,j+1,k] - P[i,j-1,k] )/(2dy)
            dP_dz   = ( P[i,j,k+1] - P[i,j,k-1] )/(2dz)
            du_dx   = ( u[i+1,j,k] - u[i-1,j,k] )/(2dx)
            du_dy   = ( u[i,j+1,k] - u[i,j-1,k] )/(2dy)
            du_dz   = ( u[i,j,k+1] - u[i,j,k-1] )/(2dz)
            dv_dx   = ( v[i+1,j,k] - v[i-1,j,k] )/(2dx)
            dv_dy   = ( v[i,j+1,k] - v[i,j-1,k] )/(2dy)
            dv_dz   = ( v[i,j,k+1] - v[i,j,k-1] )/(2dz)
            dw_dx   = ( w[i+1,j,k] - w[i-1,j,k] )/(2dx)
            dw_dy   = ( w[i,j+1,k] - w[i,j-1,k] )/(2dy)
            dw_dz   = ( w[i,j,k+1] - w[i,j,k-1] )/(2dz)
            duu_dx  = ( (u[i+1,j,k]*u[i+1,j,k]) - (u[i-1,j,k]*u[i-1,j,k]) )/(2dx)
            duv_dy  = ( (u[i,j+1,k]*v[i,j+1,k]) - (u[i,j-1,k]*v[i,j-1,k]) )/(2dy)
            duw_dz  = ( (u[i,j,k+1]*w[i,j,k+1]) - (u[i,j,k-1]*w[i,j,k-1]) )/(2dz)
            dvu_dx  = ( (v[i+1,j,k]*u[i+1,j,k]) - (v[i-1,j,k]*u[i-1,j,k]) )/(2dx)
            dvv_dy  = ( (v[i,j+1,k]*v[i,j+1,k]) - (v[i,j-1,k]*v[i,j-1,k]) )/(2dy)
            dvw_dz  = ( (v[i,j,k+1]*w[i,j,k+1]) - (v[i,j,k-1]*w[i,j,k-1]) )/(2dz)
            dwu_dx  = ( (w[i+1,j,k]*u[i+1,j,k]) - (w[i-1,j,k]*u[i-1,j,k]) )/(2dx)
            dwv_dy  = ( (w[i,j+1,k]*v[i,j+1,k]) - (w[i,j-1,k]*v[i,j-1,k]) )/(2dy)
            dww_dz  = ( (w[i,j,k+1]*w[i,j,k+1]) - (w[i,j,k-1]*w[i,j,k-1]) )/(2dz)
            d²u_dx² = ( u[i-1,j,k] - 2u[i,j,k] + u[i+1,j,k] )/(dx^2)
            d²u_dy² = ( u[i,j-1,k] - 2u[i,j,k] + u[i,j+1,k] )/(dy^2)
            d²u_dz² = ( u[i,j,k-1] - 2u[i,j,k] + u[i,j,k+1] )/(dz^2)
            d²v_dx² = ( v[i-1,j,k] - 2v[i,j,k] + v[i+1,j,k] )/(dx^2)
            d²v_dy² = ( v[i,j-1,k] - 2v[i,j,k] + v[i,j+1,k] )/(dy^2)
            d²v_dz² = ( v[i,j,k-1] - 2v[i,j,k] + v[i,j,k+1] )/(dz^2)
            d²w_dx² = ( w[i-1,j,k] - 2w[i,j,k] + w[i+1,j,k] )/(dx^2)
            d²w_dy² = ( w[i,j-1,k] - 2w[i,j,k] + w[i,j+1,k] )/(dy^2)
            d²w_dz² = ( w[i,j,k-1] - 2w[i,j,k] + w[i,j,k+1] )/(dz^2)

            # Predictor step for u: skew-symmetric form
            us[i,j,k] = u[i,j,k] + dt * (
                - 0.5 * ( duu_dx + duv_dy + duw_dz )
                - 0.5 * ( u[i,j,k]*du_dx + v[i,j,k]*du_dy + w[i,j,k]*du_dz )
                - 1/rho * dP_dx
                + mu/rho * ( d²u_dx² + d²u_dy² + d²u_dz²)
            )

            # Predictor step for v: skew-symmetric form
            vs[i,j,k] = v[i,j,k] + dt * (
                - 0.5 * ( dvu_dx + dvv_dy + dvw_dz )
                - 0.5 * ( u[i,j,k]*dv_dx + v[i,j,k]*dv_dy + w[i,j,k]*dv_dz )
                - 1/rho * dP_dy
                + mu/rho * ( d²v_dx² + d²v_dy² + d²v_dz²)
            )

            # Predictor step for w: skew-symmetric form
            ws[i,j,k] = w[i,j,k] + dt * (
                - 0.5 * ( dwu_dx + dwv_dy + dww_dz )
                - 0.5 * ( u[i,j,k]*dw_dx + v[i,j,k]*dw_dy + w[i,j,k]*dw_dz )
                - 1/rho * dP_dz
                + mu/rho * ( d²w_dx² + d²w_dy² + d²w_dz²)
            )
        end
    end

    # Update Processor boundaries
    update_borders!(us,mesh,par_env)
    update_borders!(vs,mesh,par_env)
    update_borders!(ws,mesh,par_env)

end