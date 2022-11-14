function poisson_solve!(P,RHS,mesh,mesh_par,par_env)

    GaussSeidel!(P,RHS,1000,mesh,mesh_par,par_env)

    return nothing
end

# Serial GaussSeidel Poisson Solver
function GaussSeidel!(P,RHS,niter,mesh,mesh_par,par_env)
    @unpack dx,dy = mesh
    @unpack imin_,imax_,jmin_,jmax_ = mesh_par
    @unpack irank,iroot = par_env
    for n=1:niter
        max_update=0.0
        for j=jmin_:jmax_
            for i=imin_:imax_
                Pnew = ( (RHS[i,j]
                        - (P[i-1,j]+P[i+1,j])/dx^2
                        - (P[i,j-1]+P[i,j+1])/dy^2) / (-2/dx^2 - 2/dy^2) )
                max_update=max(max_update,abs(Pnew-P[i,j]))
                P[i,j] = Pnew
            end
        end
        update_borders!(P,mesh_par,par_env)
        BC_apply!(P,"Neuman",mesh_par,par_env)
        max_update = parallel_max(max_update,par_env,recvProcs="all")
        # irank == iroot && println("Max update = ",max_update)

        max_update < 1e-10 && return nothing # Converged
    end

    return nothing
end

# Serial Multigrid solver
function multigrid(P,RHS,level,nlevels,mesh,mesh_par)

    # Pre-smoothing
    P = GaussSeidel(P,RHS,10,mesh,mesh_par)

    if level < nlevels
        # Restrict
        Pn,RHSn,meshn,mesh_parn = multigrid_restrict(P,RHS,mesh,mesh_par)
        # Call multigrid on restricted mesh
        Pn,RHSn = multigrid(Pn,RHSn,level+1,nlevels,meshn,mesh_parn)
        # Prolongate
        P = multigrid_prolongate(Pn,RHSn,meshn,mesh_parn)

        # Post-smoothting
        P = GaussSeidel(P,RHS,10,mesh,mesh_par)
    end

    return P,RHS
end

function multigrid_restrict(P,RHS,mesh,mesh_par)

    @unpack x,y,dx,dy = mesh
    @unpack imin_,imax_,jmin_,jmax_,imino_,imaxo_,jmino_,jmaxo_ = mesh_par

    # Compute restricted mesh
    xn=x[imin_:2:imax_]

    # Compute the residual
    r = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_)
    for j=jmin_:jmax_
        for i=imin_:imax_
            r[i,j]=(RHS[i,j]
                    - (P[i-1,j]+P[i+1,j])/dx^2
                    - (P[i,j-1]+P[i,j+1])/dy^2
                    + P[i,j]*(-2/dx^2 - 2/dy^2) )
        end
    end




    return Pn,RHSn,meshn,mesh_parn
end

function multigrid_prolongate(P,RHS,mesh,mesh_par)

    return P
end
