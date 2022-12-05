function VF_transport!(VF,nx,ny,nz,u,v,w,uf,vf,wf,Fx,Fy,Fz,t,dt,param,mesh,par_env)
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

    # Compute interface normal 
    computeNormal!(nx,ny,nz,VF,param,mesh,par_env)

    # # Compute fluxes
    # fill!(Fx,0.0)
    # for i = imin_:imax_+1, j = jmin_:jmax_, k = kmin_:kmax_ # Loop over faces 
    #     VFface = 0.5*(VF[i-1,j,k] + VF[i,j,k])
    #     Fx[i,j,k] = dy*dz*( - uf[i,j,k]*VFface )
    # end
    # fill!(Fy,0.0)
    # for i = imin_:imax_, j = jmin_:jmax_+1, k = kmin_:kmax_ # Loop over faces 
    #     VFface = 0.5*(VF[i,j-1,k] + VF[i,j,k])
    #     Fy[i,j,k] = dx*dz*( - vf[i,j,k]*VFface )
    # end
    # fill!(Fz,0.0)
    # for i = imin_:imax_, j = jmin_:jmax_, k = kmin_:kmax_+1 # Loop over faces 
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
        return dist =  (Lx+Ly+Lz)
    elseif VF > VFhi
        return dist = -(Lx+Ly+Lz)
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
    tets = Array{Float64}(undef,3,4,5)
    vert = Array{Float64}(undef,3,8)
    d = Array{Float64}(undef,4)

    # Compute VF in this cell 
    VF=0.0
    # Cell vertices 
    p1=[x[i  ],y[j  ],z[k  ]]
    p2=[x[i+1],y[j  ],z[k  ]]
    p3=[x[i  ],y[j+1],z[k  ]]
    p4=[x[i+1],y[j+1],z[k  ]]
    p5=[x[i  ],y[j  ],z[k+1]]
    p6=[x[i+1],y[j  ],z[k+1]]
    p7=[x[i  ],y[j+1],z[k+1]]
    p8=[x[i+1],y[j+1],z[k+1]]
    # Make five tets 
    tets[:,1,1]=p1; tets[:,2,1]=p2; tets[:,3,1]=p4; tets[:,4,1]=p6
    tets[:,1,2]=p1; tets[:,2,2]=p4; tets[:,3,2]=p3; tets[:,4,2]=p7
    tets[:,1,3]=p1; tets[:,2,3]=p5; tets[:,3,3]=p6; tets[:,4,3]=p7
    tets[:,1,4]=p4; tets[:,2,4]=p7; tets[:,3,4]=p6; tets[:,4,4]=p8
    tets[:,1,5]=p1; tets[:,2,5]=p4; tets[:,3,5]=p7; tets[:,4,5]=p6
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
            a=vert[:,cut_vtet[1,n,case]] - vert[:,cut_vtet[4,n,case]]
            b=vert[:,cut_vtet[2,n,case]] - vert[:,cut_vtet[4,n,case]]
            c=vert[:,cut_vtet[3,n,case]] - vert[:,cut_vtet[4,n,case]]
            vol=abs(a[1]*(b[2]*c[3]-c[2]*b[3]) 
                -   a[2]*(b[1]*c[3]-c[1]*b[3]) 
                +   a[3]*(b[1]*c[2]-c[1]*b[2]))/6.0
            # Update VF in this cell
            VF += vol/(dx*dy*dz)
        end
    end
    return VF
end