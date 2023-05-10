"""
Compute interface normal using ELVIRA 
Pilliod and Puckett, JCP 199, 2004  
- Overwrites nx,ny,nz
"""
function ELVIRA!(nx,ny,nz,VF,param,mesh,par_env)
    @unpack VFlo,VFhi = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # Zero normals
    fill!(nx,0.0)
    fill!(ny,0.0)
    fill!(nz,0.0)

    # Loop over cells
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
       if VF[i,j,k] >= VFlo && VF[i,j,k] <= VFhi
            ELVIRA_calc!(nx,ny,nz,VF,i,j,k,param,mesh)
       end 
    end  
    # Update processor boundaries
    update_borders!(nx,mesh,par_env)
    update_borders!(ny,mesh,par_env)
    update_borders!(nz,mesh,par_env)

    return nothing 
end

"""
Use ELVIRA to compute interface normal
in i,j,k computational cell 
- Overwrites nx,ny,nz
"""
function ELVIRA_calc!(nx,ny,nz,VF,i,j,k,param,mesh)
    @unpack Nx,Ny,Nz = mesh
    @unpack dx,dy,dz = mesh
    @unpack xm,ym,zm = mesh
  
    # Set initial error to huge
    err=[Inf,]

    # X-Y Stencils 
    if Nz > 1 
        for jj=[[j-1,j],[j-1,j+1],[j,j+1]], ii=[[i-1,i],[i-1,i+1],[i,i+1]]
            nx_ = -( sum(VF[ii[2],j,k-1:k+1]*dz) - sum(VF[ii[1],j,k-1:k+1]*dz) ) / (xm[ii[2]] - xm[ii[1]])
            ny_ = -( sum(VF[i,jj[2],k-1:k+1]*dz) - sum(VF[i,jj[1],k-1:k+1]*dz) ) / (ym[jj[2]] - ym[jj[1]])
            nz_ = VF[i,j,k+1] > VF[i,j,k-1] ? -1.0 : 1.0
            check_normal!(err,nx_,ny_,nz_,nx,ny,nz,VF,i,j,k,param,mesh)
        end
    end

    # X-Z Stencils 
    if Ny > 1 
        for kk=[[k-1,k],[k-1,k+1],[k,k+1]], ii=[[i-1,i],[i-1,i+1],[i,i+1]]
            nx_ = -( sum(VF[ii[2],j-1:j+1,k]*dy) - sum(VF[ii[1],j-1:j+1,k]*dy) ) / (xm[ii[2]] - xm[ii[1]])
            ny_ = VF[i,j+1,k] > VF[i,j-1,k] ? -1.0 : 1.0
            nz_ = -( sum(VF[i,j-1:j+1,kk[2]]*dy) - sum(VF[i,j-1:j+1,kk[1]]*dy) ) / (zm[kk[2]] - zm[kk[1]])
            check_normal!(err,nx_,ny_,nz_,nx,ny,nz,VF,i,j,k,param,mesh)
        end
    end

    # Y-Z Stencils 
    if Nx > 1 
        for kk=[[k-1,k],[k-1,k+1],[k,k+1]], jj=[[j-1,j],[j-1,j+1],[j,j+1]]
            nx_ = VF[i+1,j,k] > VF[i-1,j,k] ? -1.0 : 1.0
            ny_ = -( sum(VF[i-1:i+1,jj[2],k]*dx) - sum(VF[i-1:i+1,jj[1],k]*dx) ) / (ym[jj[2]] - ym[jj[1]])
            nz_ = -( sum(VF[i-1:i+1,j,kk[2]]*dx) - sum(VF[i-1:i+1,j,kk[1]]*dx) ) / (zm[kk[2]] - zm[kk[1]])
            check_normal!(err,nx_,ny_,nz_,nx,ny,nz,VF,i,j,k,param,mesh)
        end
    end
    return nothing 
end
    
""" 
Checks if [nx_,ny_,nz_] is a better normal.  
If it is then update [nx,ny,nz] and err
"""
function check_normal!(err,nx_,ny_,nz_,nx,ny,nz,VF,i,j,k,param,mesh)

    #Initialize error for this normal
    err_=0.0

    # Normalize normal
    norm = sqrt(nx_^2 + ny_^2 + nz_^2)
    nx_=nx_/norm
    ny_=ny_/norm
    nz_=nz_/norm

    # Form PLIC
    dist = computeDist(i,j,k,nx_,ny_,nz_,VF[i,j,k],param,mesh)

    # Loop over neighbors and compute error in VF
    for kk=k-1:k+1, jj=j-1:j+1, ii=i-1:i+1
        VF_ = computePLIC2VF(ii,jj,kk,nx_,ny_,nz_,dist,mesh)
           
        # Update L2 error
        err_ += (VF[ii,jj,kk] - VF_)^2
    end
        
    # Check if this is the best normal 
    if err_ < err[1]
        err[1] = err_
        nx[i,j,k] = nx_
        ny[i,j,k] = ny_
        nz[i,j,k] = nz_
    end
    
    return nothing
end