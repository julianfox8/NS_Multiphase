function poisson_solve!(P,RHS,param,mesh,par_env)

    GaussSeidel!(P,RHS,param,mesh,par_env)

    return nothing
end

# Serial GaussSeidel Poisson Solver
function GaussSeidel!(P,RHS,param,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack isroot = par_env
    maxIter=1000
    iter = 0
    while true
        iter += 1
        max_update=0.0
        for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            Pnew = ( (RHS[i,j,k]
                    - (P[i-1,j,k]+P[i+1,j,k])/dx^2
                    - (P[i,j-1,k]+P[i,j+1,k])/dy^2 
                    - (P[i,j,k-1]+P[i,j,k+1])/dz^2) 
                    / (-2.0/dx^2 - 2.0/dy^2 - 2.0/dz^2) )
            max_update=max(max_update,abs(Pnew-P[i,j,k]))
            P[i,j,k] = Pnew
        end
        update_borders!(P,mesh,par_env)
        BC_apply!(P,"Neuman",param,mesh,par_env)
        # Check if converged
        max_update = parallel_max(max_update,par_env,recvProcs="all")
        max_update < 1e-10 && return iter # Converged
        # Check if hit max iteration
        if iter == maxIter 
            isroot && println("Failed to converged Poisson equation max_upate = $max_update")
            return nothing 
        end
    end
end

# # Serial Multigrid solver
# function multigrid(P,RHS,level,nlevels,mesh)

#     # Pre-smoothing
#     P = GaussSeidel(P,RHS,10,mesh)

#     if level < nlevels
#         # Restrict
#         Pn,RHSn,meshn = multigrid_restrict(P,RHS,mesh)
#         # Call multigrid on restricted mesh
#         Pn,RHSn = multigrid(Pn,RHSn,level+1,nlevels,meshn)
#         # Prolongate
#         P = multigrid_prolongate(Pn,RHSn,meshn)

#         # Post-smoothting
#         P = GaussSeidel(P,RHS,10,mesh)
#     end

#     return P,RHS
# end

# function multigrid_restrict(P,RHS,mesh)

#     @unpack x,y,dx,dy,imin_,imax_,jmin_,jmax_,imino_,imaxo_,jmino_,jmaxo_ = mesh

#     # Compute restricted mesh
#     xn=x[imin_:2:imax_]

#     # Compute the residual
#     r = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_)
#     for j=jmin_:jmax_
#         for i=imin_:imax_
#             r[i,j]=(RHS[i,j]
#                     - (P[i-1,j]+P[i+1,j])/dx^2
#                     - (P[i,j-1]+P[i,j+1])/dy^2
#                     + P[i,j]*(-2/dx^2 - 2/dy^2) )
#         end
#     end




#     return Pn,RHSn,meshn
# end

# function multigrid_prolongate(P,RHS,mesh)

#     return P
# end
