# Structs to hold interface PLIC 
struct Point
    pt :: Vector{Float64}
end

struct Tri
    tri :: Vector{Int64}
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
    nband=1

    # Sweep to set gas/liquid/interface cells 
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # if VF[i,j,k] < VFlo
        #     band[i,j,k] = -nband-1
        # elseif VF[i,j,k] > VFhi 
        #     band[i,j,k] = +nband+1
        # else
        #     band[i,j,k] = 0
        # end
        if VF[i,j,k] < VFlo
            for kk=k-1:k+1,jj=j-1:j+1, ii=i-1:i+1
                if VF[ii,jj,kk] >= VFhi
                    band[i,j,k] = 0
                    break
                else 
                    band[i,j,k] = -nband-1
                end
            end
        elseif VF[i,j,k] > VFhi
            for kk=k-1:k+1,jj=j-1:j+1, ii=i-1:i+1
                if VF[ii,jj,kk] <= VFlo
                    band[i,j,k] = 0
                    break
                else 
                    band[i,j,k] = +nband+1
                end
            end 
        else
            band[i,j,k] = 0
        end
        
        # TODO: Add check for implicit interfaces between cells (only will matter for test cases)
    end
    Neumann!(band,mesh,par_env)
    update_borders!(band,mesh,par_env)

    # Sweep to identify the bands 
    for n=1:nband
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
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
    
    @loop param for k=kmino_:kmaxo_, j=jmino_:jmaxo_, i=imino_:imaxo_
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
function computePLIC2VF(i,j,k,nx,ny,nz,dist,param,mesh)    
    @unpack x,y,z = mesh
    @unpack dx,dy,dz = mesh
    
    # Allocate work arrays 
    tet = Array{Float64}(undef,3,4)
    tets = Array{Float64}(undef,3,4,5)
    vert = Array{Float64}(undef,3,8)
    d = Array{Float64}(undef,4)
    # ext = get_
    # Compute VF in this cell 
    VF=0.0
    tetsign = cell2tets!(vert,tets,i,j,k,param,mesh)
    for t=1:5
        # Copy verts
        for n=1:4
            for p=1:3
                vert[p,n] = tets[p,n,t]
            end
        end
        # Calculate distance to PLIC reconstruction
        for n=1:4
            d[n] = nx*vert[1,n]+ny*vert[2,n]+nz*vert[3,n]-dist
        end
        # Compute cut case 
        case = d2case(d)
        
        # Create interpolated vertices on cut plane
        for n=1:cut_nvert[case]
            v1 = cut_v1[n,case]; v2 = cut_v2[n,case]
            mu=min(1.0,max(0.0, -d[v1] /̂ (d[v2]-d[v1])))
            for nn = 1:3
                vert[nn,4+n]=(1.0-mu)*vert[nn,v1]+mu*vert[nn,v2]
            end
        end
        # Create new tets on liquid side
        for n=cut_ntets[case]:-1:cut_nntet[case]
            # Form tet
            for t = 1:4
                vt = cut_vtet[t,n,case]
                for p = 1:3
                    tet[p,t] = vert[p,vt]
                end
            end
            # Compute volume
            VF += tetsign*tet_vol(tet)/(dx*dy*dz)
        end
    end
    return VF
end


""" 
Compute unstructured mesh representing PLIC 
- points - vertices of each triangle 
- tris - [ntri,3] array of points that make up each triangle 
"""
function PLIC2Mesh(nx,ny,nz,D,VF,verts,tets,param,mesh)
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
            tetsign = cell2tets!(verts,tets,i,j,k,param,mesh)
            for tet=1:5
                # Copy verts
                for n=1:4
                    vert[:,n] = tets[:,n,tet]
                end
                # Calculate distance to PLIC reconstruction
                for n=1:4
                    d[n] = nx[i,j,k]*vert[1,n]+ny[i,j,k]*vert[2,n]+nz[i,j,k]*vert[3,n]-D[i,j,k]
                end
                # Compute cut case 
                case = d2case(d)
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
                    # Compute cut case 
                    case = d2case(d[1:3])
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
                        # Compute cut case 
                        case2 = d2case(d[1:3])
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
                        # Compute cut case 
                        case2 = d2case(d[1:3])
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
    # Add zero area tri if no tri on this proc
    if npts==0
        for nnn=1:3
            npts += 1
            push!(pts,Point([x[imin_],y[jmin_],z[kmin_]]))
        end
        tri = Tri([npts-2,npts-1,npts])
        push!(tris,tri)
    end

    return pts, tris
end