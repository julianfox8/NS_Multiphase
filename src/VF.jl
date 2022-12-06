# Structs to hold interface PLIC 
struct Point
    pt :: Vector{Float64}
end

struct Tri
    tri :: Vector{Int64}
end

function VF_transport!(VF,nx,ny,nz,D,band,u,v,w,uf,vf,wf,VFnew,t,dt,param,mesh,par_env)
    @unpack solveNS, VFVelocity = param
    @unpack dx,dy,dz = mesh 
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack x,y,z,xm,ym,zm = mesh
    
    # Set velocity if not using NS solver
    if !solveNS 
        # Define velocity functions
        if VFVelocity == "Deformation"
            u_fun(x,y,z,t) = -2(sin(π*x))^2*sin(π*y)*cos(π*y)*cos(π*t/8.0)
            v_fun(x,y,z,t) = +2(sin(π*y))^2*sin(π*x)*cos(π*x)*cos(π*t/8.0)
            w_fun(x,y,z,t) = 0.0
            # Set velocities (including ghost cells)
            for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
                u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
                v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
                w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
                uf[i,j,k] = u_fun( x[i],ym[j],zm[k],t)
                vf[i,j,k] = v_fun(xm[i], y[j],zm[k],t)
                wf[i,j,k] = w_fun(xm[i],ym[j], z[k],t)
            end
        else
            error("Unknown VFVelocity = $VFVelocity")
        end
    end

    # Create band around interface 
    computeBand!(band,VF,param,mesh,par_env)
    
    # Compute interface normal 
    computeNormal!(nx,ny,nz,VF,param,mesh,par_env)
    
    # Compute PLIC reconstruction 
    computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)

    # Transport VF with semi-Lagrangian cells
    fill!(VFnew,0.0)
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 

        if i==-30 && j==36 && k==1
            debug = true 
        else
            debug = false
        end

        # From projected cell and break into tets 
        tets,inds = cell2tets_withProject(i,j,k,u,v,w,dt,mesh)
        # Compute VF in semi-Lagrangian cell 
        vol  = 0.0
        vLiq = 0.0
        for tet=1:5
            tetVol, tetVLiq = cutTet(tets[:,:,tet],inds[:,:,tet],nx,ny,nz,D,mesh,debug)
            vol += tetVol
            vLiq += tetVLiq
        end
        VFnew[i,j,k] = vLiq/vol

        if debug 
            tets2VTK(tets,"debugCell")
        end

    end
    VF[:,:,:] .= VFnew[:,:,:]
    
    
    # # Compute fluxes
    # fill!(Fx,0.0)
    # for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_+1 # Loop over faces 
    #     VFface = 0.5*(VF[i-1,j,k] + VF[i,j,k])
    #     Fx[i,j,k] = dy*dz*( - uf[i,j,k]*VFface )
    # end
    # fill!(Fy,0.0)
    # for k = kmin_:kmax_, j = jmin_:jmax_+1, i = imin_:imax_ # Loop over faces 
    #     VFface = 0.5*(VF[i,j-1,k] + VF[i,j,k])
    #     Fy[i,j,k] = dx*dz*( - vf[i,j,k]*VFface )
    # end
    # fill!(Fz,0.0)
    # for k = kmin_:kmax_+1, j = jmin_:jmax_, i = imin_:imax_ # Loop over faces 
    #     VFface = 0.5*(VF[i,j,k-1] + VF[i,j,k])
    #     Fz[i,j,k] = dx*dy*( - wf[i,j,k]*VFface )
    # end
    # for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
    #     VF[i,j,k] += dt/(dx*dy*dz) * (
    #         Fx[i+1,j,k] - Fx[i,j,k] +
    #         Fy[i,j+1,k] - Fy[i,j,k] + 
    #         Fz[i,j,k+1] - Fz[i,j,k]
    #     )
    # end
    return nothing
end

""" 
Compute band around interface 
⋅ band== 0 = Has interface
⋅ band==+n = liquid phase (n-away from interface)
⋅ band==-n = gas phase (n-away from interface)
""" 
function computeBand!(band,VF,param,mesh,par_env)
    @unpack VFlo,VFhi = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh 

    # Number of bands to create 
    nband=2

    # Sweep to set gas/liquid/interface cells 
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 
        if VF[i,j,k] < VFlo
            band[i,j,k] = -nband-1
        elseif VF[i,j,k] > VFhi 
            band[i,j,k] = +nband+1
        else
            band[i,j,k] = 0
        end
        # TODO: Add check for implicit interfaces between cells (only will matter for test cases)
    end
    update_borders!(band,mesh,par_env)

    # Sweep to identify the bands 
    for n=1:nband
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 
            if abs(band[i,j,k]) > n
                # Check neighbors 
                for kk=k-1:k+1, jj=j-1:j+1, ii=i-1:i+1
                    if abs(band[ii,jj,kk]) == n-1
                        band[i,j,k] = n*sign(band[i,j,k])
                    end
                end
            end
        end
        update_borders!(band,mesh,par_env)
    end
    return nothing
end

"""
Compute interface normal vector
"""
function computeNormal!(nx,ny,nz,VF,param,mesh,par_env)
    @unpack normalMethod = param 
    
    if normalMethod == "ELVIRA"
        ELVIRA!(nx,ny,nz,VF,param,mesh,par_env)
    else
        error("Unknown method to copute interface normal: normalMethod = $normalMethod")
    end
    
    return nothing
end

"""
Compute interface reconstruction (PLIC)
"""
function computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh 
    
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        D[i,j,k]=computeDist(i,j,k,nx[i,j,k],ny[i,j,k],nz[i,j,k],VF[i,j,k],param,mesh)
    end
    update_borders!(D,mesh,par_env)
    
    return nothing
end

"""
Compute d in equation for plane 
nx*x+ny*y+nz*z=d       
"""
function computeDist(i,j,k,nx,ny,nz,VF,param,mesh)
    @unpack VFlo, VFhi = param
    @unpack dx,dy,dz = mesh
    @unpack Lx,Ly,Lz = mesh
    @unpack xm,ym,zm = mesh
    
    # Cells without interface 
    if VF < VFlo 
        return dist = -(Lx+Ly+Lz)
    elseif VF > VFhi
        return dist = +(Lx+Ly+Lz)
    end
    
    # Form plane equation variables, dealing with non-cubic cells
    mm = [nx*dx, ny*dy, nz*dz]
    # Normalize mm
    norm = sum(abs.(mm)) + eps()
    mm = mm/norm
    
    # Deal with negative mm
    factor=0.0
    if mm[1] < 0.0; factor += mm[1]; end
    if mm[2] < 0.0; factor += mm[2]; end
    if mm[3] < 0.0; factor += mm[3]; end
    
    # Deal with VF > 0.5
    VFo = VF > 0.5 ? 1.0 - VF : VF
    
    # Compute alpha
    alpha = computeAlpha(abs.(mm),VFo)
    
    # Finish dealing with VF > 0.5
    alpha = VF > 0.5 ? 1.0 - alpha : alpha
    
    # Adjust alpha due to negative mm
    alpha += factor
    
    # Write plane with barycenter as origin inside cube
    alpha -= 0.5*sum(mm)
    
    # Make distance consistant with original normal
    dist = alpha * norm
    
    # Write plane in a global reference frame
    dist += nx*xm[i] + ny*ym[j] + nz*zm[k]
    
    return dist
end

"""
Compute alpha for plane given volume of fluid
m1*x1+m2*x2+m3*x3=alpha and VF=V            
assumes cubic cell, positive normal, and VF<0.5
Scardovelli & Zaleski, JCP 164,228-247 (2000)
"""
function computeAlpha(m,VF)    
    
    # Sort m so that m[1]<m[2]<m[3]
    sort!(m)
    
    #Form constants
    m1 = m[1]
    m2 = m[2]
    m3 = m[3]
    m12 = m[1] + m[2]
    V1 = m[1]^2/max(6.0*m[2]*m[3],eps())
    V2 = V1 + 0.5*(m[2]-m[1])/̂m3 
    if m12 <= m[3] 
        V3 = 0.5m12 /̂ m3
    else
        V3 = (m3^2*(3.0m12-m3) + m1^2*(m1-3.0m3) + m2^2*(m2-3.0m3))/̂(6.0m1*m2*m3)
    end
    
    # Calculate alpha
    if 0.0 <= VF && VF < V1 
        alpha = cbrt(6.0*m1*m2*m3*VF)
    elseif V1 <= VF && VF < V2 
        alpha = 0.5*(m1+sqrt(m1^2+8.0m2*m3*(VF-V1)))
    elseif V2 <= VF && VF < V3 
        a0 = m1^3 + m2^3 - 6.0m1*m2*m3*VF
        a1 = -3.0*(m1^2+m2^2)
        a2 = 3.0m12
        a3 = -1.0
        a0 = a0/a3; a1=a1/a3; a2=a2/a3; a3=a3/a3;
        p0 = a1/3.0 - a2^2/9.0
        q0 = (a1*a2-3.0a0)/6.0 - a2^3/27.0
        theta = acos(min(+1.0,max(-1.0,q0/sqrt(-1.0*p0^3))))/3.0
        alpha = sqrt(-p0)*(sqrt(3.0)*sin(theta)-cos(theta))-a2/3.0
    else
        if (m12 <= m3) 
            alpha = m3*VF + 0.5m12
        else
            a0=m1^3+m2^3+m3^3-6.0*m1*m2*m3*VF
            a1=-3.0*(m1^2+m2^2+m3^2)
            a2=3.0
            a3=-2.0    
            a0=a0/a3; a1=a1/a3; a2=a2/a3; a3=a3/a3;          
            p0=a1/3.0-a2^2/9.0
            q0=(a1*a2-3*a0)/6.0-a2^3/27.0
            theta=acos(min(+1.0,max(-1.0,q0/sqrt(-p0^3))))/3.0
            alpha=sqrt(-p0)*(sqrt(3.0)*sin(theta)-cos(theta))-a2/3.0
        end
    end
    
    return alpha
end

""" 
Compute liquid volume in cell using PLIC
"""
function computePLIC2VF(i,j,k,nx,ny,nz,dist,mesh)    
    @unpack x,y,z = mesh
    @unpack dx,dy,dz = mesh
    
    # Allocate work arrays 
    vert = Array{Float64}(undef,3,8)
    d = Array{Float64}(undef,4)
    
    # Compute VF in this cell 
    VF=0.0
    tets=cell2tets(i,j,k,mesh)
    for tet=1:5
        # Copy verts
        for n=1:4
            vert[:,n] = tets[:,n,tet]
        end
        # Calculate distance to PLIC reconstruction
        for n=1:4
            d[n] = nx*vert[1,n]+ny*vert[2,n]+nz*vert[3,n]-dist
        end
        # Handle zero distances
        npos = length(d[d.>0.0])
        nneg = length(d[d.<0.0])
        d[d.==0] .= eps()*( npos > nneg ? 1.0 : -1.0 )
        # Determine case
        case=(
        1+Int(0.5+0.5*sign(d[1]))+
        2*Int(0.5+0.5*sign(d[2]))+
        4*Int(0.5+0.5*sign(d[3]))+
        8*Int(0.5+0.5*sign(d[4])))
        
        # Create interpolated vertices on cut plane
        for n=1:cut_nvert[case]
            v1 = cut_v1[n,case]; v2 = cut_v2[n,case]
            mu=min(1.0,max(0.0, -d[v1] /̂ (d[v2]-d[v1])))
            vert[:,4+n]=(1.0-mu)*vert[:,v1]+mu*vert[:,v2]
        end
        # Create new tets on liquid side
        for n=cut_ntets[case]:-1:cut_nntet[case]
            # Compute volume
            VF += tet_vol(vert[:,cut_vtet[:,n,case]])/(dx*dy*dz)
            # a=vert[:,cut_vtet[1,n,case]] - vert[:,cut_vtet[4,n,case]]
            # b=vert[:,cut_vtet[2,n,case]] - vert[:,cut_vtet[4,n,case]]
            # c=vert[:,cut_vtet[3,n,case]] - vert[:,cut_vtet[4,n,case]]
            # vol=abs(a[1]*(b[2]*c[3]-c[2]*b[3]) 
            # -   a[2]*(b[1]*c[3]-c[1]*b[3]) 
            # +   a[3]*(b[1]*c[2]-c[1]*b[2]))/6.0
            # Update VF in this cell
            #VF += vol/(dx*dy*dz)
        end
    end
    return VF
end

""" 
Semi-Lagrangian projection of point back in time 
""" 
function project(pt,i,j,k,u,v,w,dt,mesh)
    v1=get_velocity(pt         ,i,j,k,u,v,w,mesh)
    v2=get_velocity(pt+0.5dt*v1,i,j,k,u,v,w,mesh)
    v3=get_velocity(pt+0.5dt*v2,i,j,k,u,v,w,mesh)
    v4=get_velocity(pt+   dt*v3,i,j,k,u,v,w,mesh)
    pt+=dt/6.0*(v1+2.0v2+2.0v3+v4)
    return pt 
end

""" 
Compute unstructured mesh representing PLIC 
- points - vertices of each triangle 
- tris - [ntri,3] array of points that make up each triangle 
"""
function PLIC2Mesh(nx,ny,nz,D,VF,param,mesh)
    @unpack x,y,z = mesh
    @unpack VFlo, VFhi = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh 
    
    
    # Initialize output with nodata 
    pts  = Vector{Point}(undef,0)
    tris = Vector{Tri}(undef,0)
    
    # Allocate work arrays 
    vert  = Array{Float64}(undef,3,6)
    vert2 = Array{Float64}(undef,3,6)
    inter = Array{Float64}(undef,3,4)
    d = Array{Float64}(undef,4)
    
    # Loop over domain
    npts = 0
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 
        # Check for interface
        if VF[i,j,k] >= VFlo && VF[i,j,k] <= VFhi
            # Construct tets in cell 
            tets = cell2tets(i,j,k,mesh)
            for tet=1:5
                # Copy verts
                for n=1:4
                    vert[:,n] = tets[:,n,tet]
                end
                # Calculate distance to PLIC reconstruction
                for n=1:4
                    d[n] = nx[i,j,k]*vert[1,n]+ny[i,j,k]*vert[2,n]+nz[i,j,k]*vert[3,n]-D[i,j,k]
                end
                # Handle zero distances
                npos = length(d[d.>0.0])
                nneg = length(d[d.<0.0])
                d[d.==0] .= eps()*( npos > nneg ? 1.0 : -1.0 )
                # Determine case
                case=(
                1+Int(0.5+0.5*sign(d[1]))+
                2*Int(0.5+0.5*sign(d[2]))+
                4*Int(0.5+0.5*sign(d[3]))+
                8*Int(0.5+0.5*sign(d[4])))
                
                # Create interpolated vertices on cut plane
                for n=1:cut_nvert[case]
                    v1 = cut_v1[n,case]; v2 = cut_v2[n,case]
                    mu=min(1.0,max(0.0, -d[v1] /̂ (d[v2]-d[v1])))
                    inter[:,n]=(1.0-mu)*vert[:,v1]+mu*vert[:,v2]
                    # Store interpolated vertices to points output 
                    npts += 1
                    push!(pts,Point(inter[:,n]))
                end
                
                # Create node list of tris on interface 
                for n=1:cut_ntris[case]
                    tri = Tri(npts.-cut_nvert[case].+(cut_vtri[:,n,case].-4))
                    push!(tris,tri)
                end
            end
        end
        
        # Inherent inferface between cells
        for dim=1:3
            pos=[0,0,0]; pos[dim]=-1
            ii=i+pos[1]; jj=j+pos[2]; kk=k+pos[3]
            if (
                (VF[ i, j, k] > VFlo && VF[i ,j ,k ] < VFhi) ||  # Cell has interface
                (VF[ii,jj,kk] > VFlo && VF[ii,jj,kk] < VFhi) ||  # Neighbor has interface
                (VF[ i, j, k] < VFlo && VF[ii,jj,kk] > VFhi) ||  # Empty <-> Full cell 
                (VF[ii,jj,kk] < VFlo && VF[ i, j, k] > VFhi)     # Empty <-> Full cell 
                )
                # Break cell face into tris
                if dim == 1
                    p1=[x[i],y[j  ],z[k  ]]
                    p2=[x[i],y[j+1],z[k  ]]
                    p3=[x[i],y[j+1],z[k+1]]
                    p4=[x[i],y[j  ],z[k+1]]
                elseif dim == 2
                    p1=[x[i  ],y[j],z[k  ]]
                    p2=[x[i+1],y[j],z[k  ]]
                    p3=[x[i+1],y[j],z[k+1]]
                    p4=[x[i  ],y[j],z[k+1]]
                else
                    p1=[x[i  ],y[j  ],z[k]]
                    p2=[x[i+1],y[j  ],z[k]]
                    p3=[x[i+1],y[j+1],z[k]]
                    p4=[x[i  ],y[j+1],z[k]]
                end
                # Form tris
                t = Array{Float64}(undef,(3,3,2))
                t[:,1,1]=p1; t[:,2,1]=p2; t[:,3,1]=p3
                t[:,1,2]=p1; t[:,2,2]=p3; t[:,3,2]=p4
                for tri=1:2
                    # Transfer verts 
                    for n=1:3
                        vert[:,n]=t[:,n,tri]
                    end
                    # Calculate distance to PLIC reconstruction
                    if VF[i,j,k] < VFlo 
                        d[:] .= +1.0
                    elseif VF[i,j,k] > VFhi 
                        d[:] .= -1.0 
                    else
                        for n=1:3
                            d[n] = nx[i,j,k]*vert[1,n]+ny[i,j,k]*vert[2,n]+nz[i,j,k]*vert[3,n]-D[i,j,k]
                        end
                    end
                    # Handle zero distances
                    npos = length(d[d.>0.0])
                    nneg = length(d[d.<0.0])
                    d[d.==0] .= eps()*( npos > nneg ? 1.0 : -1.0 )
                    # Determine case of tri
                    case=(
                    1+Int(0.5+0.5*sign(d[1]))+
                    2*Int(0.5+0.5*sign(d[2]))+
                    4*Int(0.5+0.5*sign(d[3])))
                    # Add points on cut plane to verts 
                    for n=1:cutTri_nvert[case]
                        v1=cutTri_v1[n,case]; v2=cutTri_v2[n,case]
                        mu=min(1.0,max(0.0, -d[v1] /̂ (d[v2]-d[v1])))
                        vert[:,3+n] = (1.0-mu)*vert[:,v1]+mu*vert[:,v2]
                    end
                    
                    # Cut tris on liquid side by neighboring cell interface
                    for n=cutTri_np[case]+1:cutTri_np[case]+cutTri_nn[case]
                        # Transfer verts
                        for nn=1:3
                            vert2[:,nn]=vert[:,cutTri_v[nn,n,case]]
                        end
                        # Compute distances
                        if VF[ii,jj,kk] < VFlo
                            d[:] .= +1.0
                        elseif VF[ii,jj,kk] > VFhi
                            d[:] .= -1.0
                        else
                            for nn=1:3
                                d[nn]=nx[ii,jj,kk]*vert2[1,nn]+ny[ii,jj,kk]*vert2[2,nn]+nz[ii,jj,kk]*vert2[3,nn]-D[ii,jj,kk]
                            end
                        end
                        # Handle zero distances
                        npos = length(d[d.>0.0])
                        nneg = length(d[d.<0.0])
                        d[d.==0] .= eps()*( npos > nneg ? 1.0 : -1.0 )
                        # Find cut case
                        case2=(
                        1+Int(0.5+0.5*sign(d[1]))+
                        2*Int(0.5+0.5*sign(d[2]))+
                        4*Int(0.5+0.5*sign(d[3])))
                        # Add points on cut plane to nodes 
                        for nn=1:cutTri_nvert[case2]
                            v1=cutTri_v1[nn,case2]; v2=cutTri_v2[nn,case2]
                            mu=min(1.0,max(0.0, -d[v1] /̂ (d[v2]-d[v1])))
                            vert2[:,3+nn]=(1.0-mu)*vert2[:,v1]+mu*vert2[:,v2]
                        end
                        # Tris on liquid side of (i,j,k) and gas side of (ii,jj,kk) cut planes
                        for nn=1:cutTri_np[case2]
                            # Create tris and save in Visit output
                            for nnn=1:3
                                npts += 1
                                push!(pts,Point(vert2[:,cutTri_v[nnn,nn,case2]]))
                            end
                            tri = Tri([npts-2,npts-1,npts])
                            push!(tris,tri)
                        end
                    end
                    
                    # Cut tris on gas side by neighboring cell interface
                    for n=1:cutTri_np[case]
                        # Transfer verts
                        for nn=1:3
                            vert2[:,nn]=vert[:,cutTri_v[nn,n,case]]
                        end
                        # Compute distances
                        if VF[ii,jj,kk] < VFlo
                            d[:] .= +1.0
                        elseif VF[ii,jj,kk] > VFhi
                            d[:] .= -1.0
                        else
                            for nn=1:3
                                d[nn]=nx[ii,jj,kk]*vert2[1,nn]+ny[ii,jj,kk]*vert2[2,nn]+nz[ii,jj,kk]*vert2[3,nn]-D[ii,jj,kk]
                            end
                        end
                        # Handle zero distances
                        npos = length(d[d.>0.0])
                        nneg = length(d[d.<0.0])
                        d[d.==0] .= eps()*( npos > nneg ? 1.0 : -1.0 )
                        # Find cut case
                        case2=(
                        1+Int(0.5+0.5*sign(d[1]))+
                        2*Int(0.5+0.5*sign(d[2]))+
                        4*Int(0.5+0.5*sign(d[3])))
                        # Add points on cut plane to nodes 
                        for nn=1:cutTri_nvert[case2]
                            v1=cutTri_v1[nn,case2]; v2=cutTri_v2[nn,case2]
                            mu=min(1.0,max(0.0, -d[v1] /̂ (d[v2]-d[v1])))
                            vert2[:,3+nn]=(1.0-mu)*vert2[:,v1]+mu*vert2[:,v2]
                        end
                        # Tris on gas side of (i,j,k) and liquid side of (ii,jj,kk) cut planes
                        for nn=cutTri_np[case2]+1:cutTri_np[case2]+cutTri_nn[case2]
                            # Create tris and save in Visit output
                            for nnn=1:3
                                npts += 1
                                push!(pts,Point(vert2[:,cutTri_v[nnn,nn,case2]]))
                            end
                            tri = Tri([npts-2,npts-1,npts])
                            push!(tris,tri)
                        end
                    end      
                end
            end
        end
    end

    return pts, tris
end