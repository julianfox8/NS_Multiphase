function initArrays(mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Allocate memory
    u  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(u ,0.0)
    v  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(v ,0.0)
    w  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(w ,0.0)
    VF = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(VF,0.0)
    nx = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(nx,0.0)
    ny = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(ny,0.0)
    nz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(nz,0.0)
    D  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(D ,0.0)
    band = OffsetArray{Int16}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(band,0)
    us = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(us,0.0)
    vs = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(vs,0.0)
    ws = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(ws,0.0)
    uf = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(uf,0.0)
    vf = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(vf,0.0)
    wf = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(wf,0.0)
    P  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(P ,0.0)
    tmp1 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(tmp1,0.0)
    tmp2 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(tmp2,0.0)
    tmp3 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(tmp3,0.0)

    return P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3
end

"""
Compute timestep 
"""
function compute_dt(u,v,w,param,mesh,par_env)
    @unpack mu,CFL = param
    @unpack dx,dy,dz = mesh

    # Convective Δt
    local_min_dx_vel = minimum([dx/maximum(u),dy/maximum(v),dz/maximum(w)])
    min_dx_vel= parallel_min_all(local_min_dx_vel,par_env)
    convec_dt = min_dx_vel

    # Viscous Δt 
    viscous_dt = minimum([dx,dy,dz])/mu
    
    # Timestep
    dt=CFL*minimum([convec_dt,viscous_dt])

    return dt
end

"""
Prints a parallel array
"""
function printArray(text,A,par_env)
    @unpack irankx,nprocx,nprocy,nprocz,irank,isroot,comm = par_env

    (nprocy > 1 || nprocz>1) && error("printArray only works with 1 proc in y and 1 proc in z")

    MPI.Barrier(comm)
    for k in axes(A,3)
        isroot && print("$text[:,:,$k]\n")
        for j in reverse(axes(A,2))
            for rankx = 0:nprocx-1
                if rankx == irankx 
                    @printf("|")
                    for i in axes(A,1)
                        @printf("%10.3g ",A[i,j,k])
                    end
                end
                flush(stdout)
                MPI.Barrier(comm)
            end
            flush(stdout)
            MPI.Barrier(comm)
            isroot && sleep(0.01)
            isroot && @printf("\n")
        end
    end

    return nothing
end

""" 
Safe Divide (avoids division by zero)
"""
function /̂(a,b)
    return abs(b)<=eps() ? typemax(Float64) : a/b
end


""" 
Determine which cell (index) a point 
lies within 
"""
function pt2index(pt,i,j,k,mesh)
    @unpack x,y,z = mesh
    I=[i,j,k]
    while pt[1] > x[I[1]+1]+eps(); I[1]=I[1]+1; end
    while pt[1] < x[I[1]  ]-eps(); I[1]=I[1]-1; end
    while pt[2] > y[I[2]+1]+eps(); I[2]=I[2]+1; end
    while pt[2] < y[I[2]  ]-eps(); I[2]=I[2]-1; end
    while pt[3] > z[I[3]+1]+eps(); I[3]=I[3]+1; end
    while pt[3] < z[I[3]  ]-eps(); I[3]=I[3]-1; end
    return I
end

""" 
Interpolate cell centered velocity to location of pt 
"""
function get_velocity(pt,i,j,k,u,v,w,mesh)
    @unpack xm,ym,zm = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    # Find right i index
    while pt[1]-xm[i  ] <  0.0 && i   > imino_
       i=i-1
    end
    while pt[1]-xm[i+1] >= 0.0 && i+1 < imaxo_
       i=i+1
    end
    # Find right j index
    while pt[2]-ym[j  ] <  0.0 && j   > jmino_
       j=j-1
    end
    while pt[2]-ym[j+1] >= 0.0 && j+1 < jmaxo_
       j=j+1
    end
    # Find right k index
    while pt[3]-zm[k  ] <  0.0 && k   > kmino_
       k=k-1
    end
    while pt[3]-zm[k+1] >= 0.0 && k+1 < kmaxo_
       k=k+1
    end
    # Prepare tri-linear interpolation coefficients
    wx1=(pt[1]-xm[i])/(xm[i+1]-xm[i]); wx2=1.0-wx1
    wy1=(pt[2]-ym[j])/(ym[j+1]-ym[j]); wy2=1.0-wy1
    wz1=(pt[3]-zm[k])/(zm[k+1]-zm[k]); wz2=1.0-wz1
    # Tri-linear interpolation
    u_pt=( wz1*(wy1*(wx1*u[i+1,j+1,k+1]  +
                     wx2*u[i  ,j+1,k+1]) +
                wy2*(wx1*u[i+1,j  ,k+1]  +
                     wx2*u[i  ,j  ,k+1]))+
           wz2*(wy1*(wx1*u[i+1,j+1,k  ]  +
                     wx2*u[i  ,j+1,k  ]) +
                wy2*(wx1*u[i+1,j  ,k  ]  +
                     wx2*u[i  ,j  ,k  ])))
    v_pt=( wz1*(wy1*(wx1*v[i+1,j+1,k+1]  +
                     wx2*v[i  ,j+1,k+1]) +
                wy2*(wx1*v[i+1,j  ,k+1]  +
                     wx2*v[i  ,j  ,k+1]))+
           wz2*(wy1*(wx1*v[i+1,j+1,k  ]  +
                     wx2*v[i  ,j+1,k  ]) +
                wy2*(wx1*v[i+1,j  ,k  ]  +
                     wx2*v[i  ,j  ,k  ])))
    w_pt=( wz1*(wy1*(wx1*w[i+1,j+1,k+1]  +
                     wx2*w[i  ,j+1,k+1]) +
                wy2*(wx1*w[i+1,j  ,k+1]  +
                     wx2*w[i  ,j  ,k+1]))+
           wz2*(wy1*(wx1*w[i+1,j+1,k  ]  +
                     wx2*w[i  ,j+1,k  ]) +
                wy2*(wx1*w[i+1,j  ,k  ]  +
                     wx2*w[i  ,j  ,k  ])))
    return [u_pt,v_pt,w_pt]

end


"""
Exact VF values for 2D circle
"""
function VFcircle(xmin,xmax,ymin,ymax,rad,xo,yo)
    
    dx = max(xmax-xmin,ymax-ymin)
    xm = 0.5( xmin + xmax)
    ym = 0.5( ymin + ymax)
    G = rad-sqrt((xm-xo)^2 + (ym-yo)^2)
    if G > 2dx
        return VF = 1.0 # Liquid phase
    elseif G < -2dx 
        return VF = 0.0 # Gas phase
    end
    VF=0.0
    xdone = false
    xcycl = 1
    while !xdone
        # Split cell into parts located within the 1st quadrant
        if (xmax-xo)*(xmin-xo) < 0.0
            # Cell needs to be split into two
            if xcycl == 1
                x_min = 0.0
                x_max = abs(xmin-xo)
                xcycl = 2
            else
                x_min = 0.0
                x_max = abs(xmax-xo)
                xdone = true
            end
        else
            x_min = min(abs(xmin-xo),abs(xmax-xo))
            x_max = max(abs(xmin-xo),abs(xmax-xo))
            xdone = true
        end

        ydone = false
        ycycl = 1
        while !ydone
            if ((ymin-yo)*(ymax-yo) < 0.0) 
                if ycycl == 1
                    y_min = 0.0
                    y_max = abs(ymin-yo)
                    ycycl = 2
                else
                    y_min = 0.0
                    y_max = abs(ymax-yo)
                    ydone = true
                end
            else
                y_min = min(abs(ymin-yo),abs(ymax-yo))
                y_max = max(abs(ymin-yo),abs(ymax-yo))
                ydone = true
            end

            # Find Integration bounds
            if rad^2-x_min^2 > 0.0
                yint=sqrt(rad^2-x_min^2)
                if yint < y_min
                    #  Cell in gas phase
                    VF += 0.0
                    continue
                elseif (yint < y_max) 
                    #  Intersection on left face
                    x_1=x_min
                else
                    #  Intersection on top face
                    x_1=sqrt(rad^2-y_max^2)
                end
            else
                #  Cell in gas phase
                VF += 0.0
                continue
            end

            if (rad^2-x_max^2 > 0.0) 
                yint=sqrt(rad^2-x_max^2)
                if (yint > y_max)  
                    #  Cell in liquid phase
                    VF += ((x_max-x_min)*(y_max-y_min))/((xmax-xmin)*(ymax-ymin))
                    continue
                elseif (yint > y_min) 
                    #  Intersection on right face
                    x_2=x_max
                else
                    #  Intersection on bottom face
                    x_2=sqrt(rad^2-y_min^2)
                end
            else
                #  Intersection on bottom face
                x_2=sqrt(rad^2-y_min^2)
            end

            #  Integrate 
            VF += (
                ( (x_1-x_min)*(y_max-y_min) + 
                (rad^2*asin(x_2/rad))/2.0 - (rad^2*asin(x_1/rad))/2.0 
                - (x_1*(rad^2 - x_1^2)^(0.5))/2.0 
                + (x_2*(rad^2 - x_2^2)^(0.5))/2.0 
                - (y_min*(x_2-x_1)) 
                ) 
                /((xmax-xmin)*(ymax-ymin))
            )
            #  2nd order VOF using trapizoidal rule
            #  VF = VF + ( &
            #       + (x_1-x_min)*(y_max-y_min) &
            #       + (x_2-x_1)*(sqrt(rad^2-x_2^2)-y_min) &
            #       + 0.5*(x_2-x_1)*(sqrt(rad^2-x_1^2)-sqrt(rad^2-x_2^2)) &
            #       ) / ((xs2(s,i)-xs1(s,i))*(ys2(s,j)-ys1(s,j)))

        end
    end
    return VF
end

"""
VF values for sphree
"""
function VFsphere(xmin,xmax,ymin,ymax,zmin,zmax,rad,xo,yo,zo)
    nF = 20
    VF=0.0
    VFsubcell = 1.0/nF^3
    # Loop over finer grid to evaluate VF 
    for k=1:nF, j=1:nF, i=1:nF
        xh = xmin + i/(nF+1)*(xmax-xmin)
        yh = ymin + j/(nF+1)*(ymax-ymin)
        zh = zmin + k/(nF+1)*(zmax-zmin)
        G = rad^2 - ((xh-xo)^2 + (yh-yo)^2 + (zh-zo)^2)
        if G > 0.0
            VF += VFsubcell
        end
    end
    return VF
end
