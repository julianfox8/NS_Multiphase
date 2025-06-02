""" 
Apply BC's on pressure
"""
function Neumann!(A,mesh,par_env)
    # @unpack xper,yper,zper = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack nprocx,nprocy,nprocz,irankx,iranky,irankz = par_env

    irankx == 0        ? A[imin_-1,:,:]=A[imin_,:,:] : nothing # Left 
    irankx == nprocx-1 ? A[imax_+1,:,:]=A[imax_,:,:] : nothing # Right

    iranky == 0        ? A[:,jmin_-1,:]=A[:,jmin_,:] : nothing # Bottom
    iranky == nprocy-1 ? A[:,jmax_+1,:]=A[:,jmax_,:] : nothing # Top

    irankz == 0        ? A[:,:,kmin_-1]=A[:,:,kmin_] : nothing # Back
    irankz == nprocz-1 ? A[:,:,kmax_+1]=A[:,:,kmax_] : nothing # Front
    return nothing
end

"""
Compute the magnitude of an array
""" 
function mag(A,par_env)
    # Compute ∑(A_i^2)
    mag = 0
    for i in eachindex(A)
        mag += A[i]^2
    end
    # Parallel sum 
    parallel_sum_all(mag,par_env)
    # Compute √(∑A_i^2)
    return sqrt(mag)
end

""" 
Macro to easily change looping behavior throughout code 
- Careful: threads and floop only work on loops where each grid point can be updated independently
"""
macro loop(args...)
    length(args) == 2 || error("Expecting param and for ...")
    p = args[1]
    ex = args[2]
        
    # Extract iterator and body of loop
    iter = ex.args[1]
    lbody = ex.args[2]
    # println(iter)

    # Check iterator has three arguments
    if length(iter.args)==3 # || error("Missing iterator")

        # Pull out iterator ids (names) and rages
        id1 = iter.args[1].args[1]; range1=iter.args[1].args[2]
        id2 = iter.args[2].args[1]; range2=iter.args[2].args[2]
        id3 = iter.args[3].args[1]; range3=iter.args[3].args[2]

        # Check id ordering
        if  id1 == :i && 
            id2 == :j && 
            id3 == :k
            idx = id1; rangex = range1
            idy = id2; rangey = range2
            idz = id3; rangez = range3
        elseif  id1 == :k && 
                id2 == :j && 
                id3 == :i
            idx = id3; rangex = range3
            idy = id2; rangey = range2
            idz = id1; rangez = range1
        elseif   id1 == :ii && 
                 id2 == :jj && 
                 id3 == :kk
            idx = id1; rangex = range1
            idy = id2; rangey = range2
            idz = id3; rangez = range3
        elseif  id1 == :kk && 
                id2 == :jj && 
                id3 == :ii
            idx = id3; rangex = range3
            idy = id2; rangey = range2
            idz = id1; rangez = range1 
        else
            error("Must provide i,j,k or k,j,i iterators")
        end


        quote
            if eval($(esc(p))).iter_type == "standard"
                # Standard for loops k,j,i
                for $(esc(idz)) = $(esc(rangez)),$(esc(idy)) = $(esc(rangey)),$(esc(idx)) = $(esc(rangex))
                    $(esc(lbody))
                end

            elseif $(esc(p)).iter_type == "threads"
                # Threads
                @threads for ind in CartesianIndices(($(esc(rangex)),$(esc(rangey)),$(esc(rangez))))
                    $(esc(idx)),$(esc(idy)),$(esc(idz)) = ind[1],ind[2],ind[3]
                    $(esc(lbody))
                end

            elseif $(esc(p)).iter_type == "floop"
                # FLoops
                @floop for ind in CartesianIndices(($(esc(rangex)),$(esc(rangey)),$(esc(rangez))))
                    $(esc(idx)),$(esc(idy)),$(esc(idz)) = ind[1],ind[2],ind[3]
                    $(esc(lbody))
                end
            else
                error("Unknown iterator type specificed")
            end 

            nothing
        end
    # elseif iter.head == :(=) && iter.args[1] == :I && iter.args[2].head == :call && iter.args[2].args[1] == :CartesianIndices
    elseif iter.args[2].head == :call       
        # Handle I = CartesianIndices(P.f[ii]) iterator
        indices_arg = iter.args[2].args[2]
        println("here")
        # println(indices_arg[1])
        # println(iter.args[2].args[1])
        # i = iter.args[1]
        quote
            indices_iter = CartesianIndices(inside($(esc(indices_arg))))
            $I = @index(Global,Cartesian)
            for $I in indices_iter
                # $(esc(i)) = I
                $(esc(lbody))
            end
        end
    else 
        error("Missing iterator")
    end
end



function initArrays(mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Allocate memory
    u  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(u ,0.0)
    v  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(v ,0.0)
    w  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(w ,0.0)

    VF = OffsetArray{Float64}(undef, imino_-3:imaxo_+3,jmino_-3:jmaxo_+3,kmino_-3:kmaxo_+3); fill!(VF,0.0)

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
    tmp4 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(tmp4,0.0)
    tmp5 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(tmp5,0.0)
    tmp6 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(tmp6,0.0)
    tmp7 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(tmp7,0.0)
    tmp8 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(tmp8,0.0)
    tmp9 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(tmp9,0.0)
    tmplrg = OffsetArray{Float64}(undef, imino_-3:imaxo_+3,jmino_-3:jmaxo_+3,kmino_-3:kmaxo_+3); fill!(tmp4,0.0)
    Curve = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(Curve,0.0)
    sfx = OffsetArray{Float64}(undef, imino_:imaxo_+1,jmino_:jmaxo_,kmino_:kmaxo_); fill!(sfx,0.0)
    sfy = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_+1,kmino_:kmaxo_); fill!(sfy,0.0)
    sfz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_+1); fill!(sfz,0.0)
    denx = OffsetArray{Float64}(undef, imino_:imaxo_+1,jmino_:jmaxo_,kmino_:kmaxo_); fill!(denx,0.0)
    deny = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_+1,kmino_:kmaxo_); fill!(deny,0.0)
    denz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_+1); fill!(denz,0.0)
    viscx = OffsetArray{Float64}(undef, imino_:imaxo_+1,jmino_:jmaxo_,kmino_:kmaxo_); fill!(viscx,0.0)
    viscy = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_+1,kmino_:kmaxo_); fill!(viscy,0.0)
    viscz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_+1); fill!(viscz,0.0)
    gradx = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(gradx,0.0)
    grady = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(grady,0.0)
    gradz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(gradz,0.0)
    divg  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(divg,0.0)
    mask = OffsetArray{Array{Bool,1}}([falses(3) for _ in imino_:imaxo_, _ in jmino_:jmaxo_, _ in kmino_:kmaxo_],imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)

    tets  = Array{Float64}(undef, 3, 4, 24); fill!(tets,0.0)
    inds  = Array{Int32}(undef, 3, 4, 24); fill!(inds,0.0)
    verts = Array{Float64}(undef, 3, 8)
    vInds = Array{Int32}(undef, 3, 8)

    return P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,divg,mask,tets,verts,inds,vInds

end

"""
Compute timestep 
"""
function compute_dt(u,v,w,param,mesh,par_env)
    @unpack CFL,max_dt,mu_liq,mu_gas = param
    @unpack dx,dy,dz = mesh


    # Convective Δt
    local_min_dx_vel = minimum([dx/maximum(abs.(u)),dy/maximum(abs.(v)),dz/maximum(abs.(w))])
    min_dx_vel= parallel_min_all(local_min_dx_vel,par_env)
    convec_dt = min_dx_vel

    # Viscous Δt 
    viscous_dt = minimum([dx,dy,dz])/max(mu_liq,mu_gas)
    
    # Timestep
    dt=min(max_dt,CFL*minimum([convec_dt,viscous_dt]))

    return dt::Float64
end

"""
    printArray(text,A,par_env)
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

function *̂(a, b)
    if abs(a) <= eps() || abs(b) <= eps()
        return 0.0  # If either a or b is close to zero, the result will be zero
    else
        return a * b
    end
end

""" 
Semi-Lagrangian projection of point back in time
- assumes face velocities
- pt is updated by function
""" 
function project!(pt,i,j,k,uf,vf,wf,dt,param,mesh)
    @unpack projection_method = param
    if projection_method == "RK4"
        v1=get_velocity_face(pt         ,i,j,k,uf,vf,wf,mesh)
        v2=get_velocity_face(pt+0.5dt*v1,i,j,k,uf,vf,wf,mesh)
        v3=get_velocity_face(pt+0.5dt*v2,i,j,k,uf,vf,wf,mesh)
        v4=get_velocity_face(pt+   dt*v3,i,j,k,uf,vf,wf,mesh)
        pt[:]+=(-dt)/6.0*(v1+2.0v2+2.0v3+v4)
        
    elseif projection_method == "Euler"
        v1=get_velocity_face(pt         ,i,j,k,uf,vf,wf,mesh)
        pt[:]+=(-dt)*v1
    elseif projection_method == "Midpoint"
        v1=get_velocity_face(pt         ,i,j,k,uf,vf,wf,mesh)
        v2=get_velocity_face(pt+0.5dt*v1,i,j,k,uf,vf,wf,mesh)
        pt[:]+=(-dt)*(v2)
    else
        error("Unknown projection_method in project!")
    end
    return nothing 
end

""" 
Determine which cell (index) a point lies within 
- I is a 3 vector and is updated by the function
"""
function pt2index!(I,pt,i,j,k,mesh)
    @unpack x,y,z,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    I.=[i,j,k]
    while pt[1] > x[I[1]+1]+eps(); I[1] += +1; end
    while pt[1] < x[I[1]  ]-eps(); I[1] += -1; end
    while pt[2] > y[I[2]+1]+eps(); I[2] += +1; end
    while pt[2] < y[I[2]  ]-eps(); I[2] += -1; end
    while pt[3] > z[I[3]+1]+eps(); I[3] += +1; end
    while pt[3] < z[I[3]  ]-eps(); I[3] += -1; end
    return nothing
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
Interpolate velocities defined on faces to location of pt 
"""
function get_velocity_face(pt,i,j,k,uf,vf,wf,mesh)
    u_pt = get_velocity_uface(pt,i,j,k,uf,mesh)
    v_pt = get_velocity_vface(pt,i,j,k,vf,mesh)
    w_pt = get_velocity_wface(pt,i,j,k,wf,mesh)
    return [u_pt,v_pt,w_pt]
end

""" 
Interpolate x face velocity to location of pt 
"""
function get_velocity_uface(pt,i,j,k,uf,mesh)
    @unpack x ,ym,zm = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Find right i index
    while pt[1]-x[i  ] <  0.0 && i   > imino_
       i=i-1
    end
    while pt[1]-x[i+1] >= 0.0 && i+1 < imaxo_
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
    wx1=(pt[1]- x[i])/̂( x[i+1]- x[i]); wx2=1.0-wx1
    wy1=(pt[2]-ym[j])/̂(ym[j+1]-ym[j]); wy2=1.0-wy1
    wz1=(pt[3]-zm[k])/̂(zm[k+1]-zm[k]); wz2=1.0-wz1

    # Tri-linear interpolation
    u_pt=( wz1*(wy1*(wx1*uf[i+1,j+1,k+1]  +
                     wx2*uf[i  ,j+1,k+1]) +
                wy2*(wx1*uf[i+1,j  ,k+1]  +
                     wx2*uf[i  ,j  ,k+1]))+
           wz2*(wy1*(wx1*uf[i+1,j+1,k  ]  +
                     wx2*uf[i  ,j+1,k  ]) +
                wy2*(wx1*uf[i+1,j  ,k  ]  +
                     wx2*uf[i  ,j  ,k  ])))
    return u_pt
end
""" 
Interpolate v face velocity to location of pt 
"""
function get_velocity_vface(pt,i,j,k,vf,mesh)
    @unpack xm,y ,zm = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    # Find right i index
    while pt[1]-xm[i  ] <  0.0 && i   > imino_
       i=i-1
    end
    while pt[1]-xm[i+1] >= 0.0 && i+1 < imaxo_
       i=i+1
    end
    # Find right j index
    while pt[2]-y[j  ] <  0.0 && j   > jmino_
       j=j-1
    end
    while pt[2]-y[j+1] >= 0.0 && j+1 < jmaxo_
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
    wx1=(pt[1]-xm[i])/̂(xm[i+1]-xm[i]); wx2=1.0-wx1
    wy1=(pt[2]- y[j])/̂( y[j+1]- y[j]); wy2=1.0-wy1
    wz1=(pt[3]-zm[k])/̂(zm[k+1]-zm[k]); wz2=1.0-wz1
    # Tri-linear interpolation
    v_pt=( wz1*(wy1*(wx1*vf[i+1,j+1,k+1]  +
                     wx2*vf[i  ,j+1,k+1]) +
                wy2*(wx1*vf[i+1,j  ,k+1]  +
                     wx2*vf[i  ,j  ,k+1]))+
           wz2*(wy1*(wx1*vf[i+1,j+1,k  ]  +
                     wx2*vf[i  ,j+1,k  ]) +
                wy2*(wx1*vf[i+1,j  ,k  ]  +
                     wx2*vf[i  ,j  ,k  ])))
    return v_pt
end
""" 
Interpolate w face velocity to location of pt 
"""
function get_velocity_wface(pt,i,j,k,wf,mesh)
    @unpack xm,ym,z  = mesh
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
    while pt[3]-z[k  ] <  0.0 && k   > kmino_
       k=k-1
    end
    while pt[3]-z[k+1] >= 0.0 && k+1 < kmaxo_
       k=k+1
    end
    # Prepare tri-linear interpolation coefficients
    wx1=(pt[1]-xm[i])/̂(xm[i+1]-xm[i]); wx2=1.0-wx1
    wy1=(pt[2]-ym[j])/̂(ym[j+1]-ym[j]); wy2=1.0-wy1
    wz1=(pt[3]- z[k])/̂( z[k+1]- z[k]); wz2=1.0-wz1
    # Tri-linear interpolation
    w_pt=( wz1*(wy1*(wx1*wf[i+1,j+1,k+1]  +
                     wx2*wf[i  ,j+1,k+1]) +
                wy2*(wx1*wf[i+1,j  ,k+1]  +
                     wx2*wf[i  ,j  ,k+1]))+
           wz2*(wy1*(wx1*wf[i+1,j+1,k  ]  +
                     wx2*wf[i  ,j+1,k  ]) +
                wy2*(wx1*wf[i+1,j  ,k  ]  +
                     wx2*wf[i  ,j  ,k  ])))
    return w_pt
end



"""
Define velocity field (usually for VF testing)
"""
function defineVelocity!(t,u,v,w,uf,vf,wf,param,mesh)
    @unpack VFVelocity = param
    @unpack x, y, z  = mesh
    @unpack xm,ym,zm = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Set velocity if not using NS solver
    if VFVelocity == "Deformation"
        u_fun = (x,y,z,t) -> -2(sin(π*x))^2*sin(π*y)*cos(π*y)*cos(π*t/8.0)
        v_fun = (x,y,z,t) -> +2(sin(π*y))^2*sin(π*x)*cos(π*x)*cos(π*t/8.0)
        w_fun = (x,y,z,t) -> 0.0
    elseif VFVelocity == "Deformation3D"
        u_fun = (x,y,z,t) -> 2(sin(π*x))^2*sin(2π*y)*sin(2π*z)*cos(π*t/3.0)
        v_fun = (x,y,z,t) -> -(sin(π*y))^2*sin(2π*x)*sin(2π*z)*cos(π*t/3.0)
        w_fun = (x,y,z,t) -> -(sin(π*z))^2*sin(2π*x)*sin(2π*y)*cos(π*t/3.0)
    elseif VFVelocity == "divFlow"
        u_fun = (x,y,z,t) -> 2*x
        v_fun = (x,y,z,t) -> -1*(2.5-y)
        w_fun = (x,y,z,t) -> 0.0
    elseif VFVelocity == "divFlow3D"
        u_fun = (x,y,z,t) -> 200*x
        v_fun = (x,y,z,t) -> -100(5-y)
        w_fun = (x,y,z,t) -> -50*(5-z)
    else
        error("Unknown VFVelocity = $VFVelocity")
    end
       
    # Set velocities (including ghost cells)
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
        v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
        w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
        uf[i,j,k] = u_fun( x[i],ym[j],zm[k],t)
        vf[i,j,k] = v_fun(xm[i], y[j],zm[k],t)
        wf[i,j,k] = w_fun(xm[i],ym[j], z[k],t)
    end
    return nothing
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
        #? the goal of these if statements is to orient xmax and xmin
        #? occurs when x0 is between xmax and xmin
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

"""
Exact VF values for 2D bubble
"""
function VFbubble(xmin,xmax,ymin,ymax,rad,xo,yo)
    
    dx = max(xmax-xmin,ymax-ymin)
    xm = 0.5( xmin + xmax)
    ym = 0.5( ymin + ymax)
    G = rad-sqrt((xm-xo)^2 + (ym-yo)^2)
    if G > 2dx
        return VF = 0.0 # Liquid phase
    elseif G < -2dx 
        return VF = 1.0 # Gas phase
    end
    VF=1.0
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
                    VF += 1.0
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
                VF += 1.0
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
VF values for 2D Bubble
"""
function VFbubble2d(xmin,xmax,ymin,ymax,rad,xo,yo)
    nF = 20
    VF=1.0
    VFsubcell = 1.0/nF^2
    # Loop over finer grid to evaluate VF 
    for j=1:nF, i=1:nF
        xh = xmin + i/(nF+1)*(xmax-xmin)
        yh = ymin + j/(nF+1)*(ymax-ymin)
        G = rad^2 - ((xh-xo)^2 + (yh-yo)^2 )
        if G > 0.0
            VF -= VFsubcell
        end
    end
    return VF
end

"""
VF values for 3D bubble
"""
function VFbubble3d(xmin,xmax,ymin,ymax,zmin,zmax,rad,xo,yo,zo)
    nF = 20
    VF=1.0
    VFsubcell = 1.0/nF^3
    # Loop over finer grid to evaluate VF 
    for k=1:nF, j=1:nF, i=1:nF
        xh = xmin + i/(nF+1)*(xmax-xmin)
        yh = ymin + j/(nF+1)*(ymax-ymin)
        zh = zmin + k/(nF+1)*(zmax-zmin)
        G = rad^2 - ((xh-xo)^2 + (yh-yo)^2 + (zh-zo)^2)
        if G > 0.0
            VF -= VFsubcell
        end
    end
    return VF
end

"""
VF values for 2D Bubble
"""
function VFdroplet2d(xmin,xmax,ymin,ymax,rad,xo,yo)
    nF = 20
    VF=0.0
    VFsubcell = 1.0/nF^2
    # Loop over finer grid to evaluate VF 
    for j=1:nF, i=1:nF
        xh = xmin + i/(nF+1)*(xmax-xmin)
        yh = ymin + j/(nF+1)*(ymax-ymin)
        G = rad^2 - ((xh-xo)^2 + (yh-yo)^2 )
        if G > 0.0
            VF += VFsubcell
        end
    end
    return VF
end

"""
VF values for 3D bubble
"""
function VFellipbub3d(xmin,xmax,ymin,ymax,zmin,zmax,xrad,yrad,zrad,xo,yo,zo)
    nF = 20
    VF=1.0
    VFsubcell = 1.0/nF^3
    # Loop over finer grid to evaluate VF 
    for k=1:nF, j=1:nF, i=1:nF
        xh = xmin + i/(nF+1)*(xmax-xmin)
        yh = ymin + j/(nF+1)*(ymax-ymin)
        zh = zmin + k/(nF+1)*(zmax-zmin)
        G = 1 - ((xh-xo)^2/xrad^2 + (yh-yo)^2/yrad^2 + (zh-zo)^2/zrad^2)
        if G > 0.0
            VF -= VFsubcell
        end
    end
    return VF
end

"""
VF values for 3D bubble
"""
function VFellipbub2d(xmin,xmax,ymin,ymax,xrad,yrad,xo,yo)
    nF = 20
    VF=1.0
    VFsubcell = 1.0/nF^2
    # Loop over finer grid to evaluate VF 
    for j=1:nF, i=1:nF
        xh = xmin + i/(nF+1)*(xmax-xmin)
        yh = ymin + j/(nF+1)*(ymax-ymin)
        G = 1 - ((xh-xo)^2/xrad^2 + (yh-yo)^2/yrad^2)
        if G > 0.0
            VF -= VFsubcell
        end
    end
    return VF
end

"""
VF values for 3D bubble
"""
function VFdroplet3d(xmin,xmax,ymin,ymax,zmin,zmax,rad,xo,yo,zo)
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

"""
Density/Viscosity calculation
"""
function compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack rho_liq,mu_liq,rho_gas,mu_gas = param

    @loop param for  k = kmin_-1:kmax_+1, j = jmin_-1:jmax_+1, i = imin_-1:imax_+2
        vfx = min(1.0,max(0.0,(VF[i,j,k]+VF[i-1,j,k])/2))
        denx[i,j,k] = rho_liq*(vfx) + rho_gas*(1-vfx)
        viscx[i,j,k] = vfx*mu_liq+(1-vfx)*mu_gas
    end
    @loop param for  k = kmin_-1:kmax_+1, j = jmin_-1:jmax_+2, i = imin_-1:imax_+1
        vfy = min(1.0,max(0.0,(VF[i,j,k]+VF[i,j-1,k])/2))
        deny[i,j,k] = rho_liq*(vfy) +rho_gas*(1-vfy)
        viscy[i,j,k] = vfy*mu_liq+(1-vfy)*mu_gas
    end
    @loop param for  k = kmin_-1:kmax_+2, j = jmin_-1:jmax_+1, i = imin_-1:imax_+1
        vfz = min(1.0,max(0.0,(VF[i,j,k]+VF[i,j,k-1])/2))
        denz[i,j,k] = rho_liq*(vfz) +rho_gas*(1-vfz)
        viscz[i,j,k] = vfz*mu_liq+(1-vfz)*mu_gas

    end
    return nothing
end

"""
Correct outflow such that sum(divg)=0 
"""
function outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,p,tets_arr,param,mesh,par_env)
    @unpack dx,dy,dz = mesh
    @unpack tol = param
    @unpack isroot = par_env
    iter=0; maxIter=1000    
    
    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,p,tets_arr,param,mesh,par_env)
    d = parallel_sum_all(AP*dx*dy*dz,par_env)
    while abs(d) > 1e-1*tol # || iter < 2
        iter += 1
        # Correct outflow 
        correction = -0.5d/outflow.area(mesh,par_env)
        outflow.correction(correction,uf,vf,wf,mesh,par_env)
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,p,tets_arr,param,mesh,par_env)
        dnew = parallel_sum_all(AP*dx*dy*dz,par_env)
        d=dnew
        if iter == maxIter
            error("outflowCorrection did not converge!")
            return
        end
    end
    return nothing
end

"""
Prepares matrix containing index mapping of i,j,k matrix location to corresponding vector index
"""
function prepare_indices(p_index,par_env,mesh)
    @unpack kmin_,kmax_,jmin_,jmax_,imin_,imax_= mesh
    @unpack comm,nproc,irank,iroot,isroot = par_env
    npcells = 0
    local_npcells = 0
    for k = kmin_:kmax_, j = jmin_:jmax_,i = imin_:imax_
            local_npcells += 1
    end

    MPI.Allreduce!([local_npcells], [npcells], MPI.SUM, comm)

    npcells_proc = zeros(Int, nproc)

    MPI.Allgather!([local_npcells], npcells_proc, comm)

    npcells_proc = cumsum(npcells_proc)
    local_count = npcells_proc[irank+1] - local_npcells
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            local_count += 1
            p_index[i, j, k] = local_count
    end

    MPI.Barrier(comm)

    update_borders!(p_index,mesh,par_env)
    
    p_max = -1
    p_min = maximum(p_index)
    for k = kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        if p_index[i,j,k] != -1
            p_min = min(p_min,p_index[i,j,k])
            p_max = max(p_max,p_index[i,j,k])
        end
    end
    return p_min,p_max

end


"""
Prepares matrix containing index mapping of i,j,k matrix location to corresponding vector index over ghost cells
"""
function prepare_indicesGhost(p_index,par_env,mesh)
    @unpack kmin_,kmax_,jmin_,jmax_,imin_,imax_,kmax,jmax,imax= mesh
    @unpack comm,nproc,irank,iroot,isroot = par_env
    npcells = 0
    local_npcells = 0

    for k = kmin_-1:kmax_+1, j = jmin_-1:jmax_+1,i = imin_-1:imax_+1
        local_npcells += 1
    end

    MPI.Allreduce!([local_npcells], [npcells], MPI.SUM, comm)

    npcells_proc = zeros(Int, nproc)

    MPI.Allgather!([local_npcells], npcells_proc, comm)

    npcells_proc = cumsum(npcells_proc)
    local_count = npcells_proc[irank+1] - local_npcells
    for k = kmin_-1:kmax_+1, j = jmin_-1:jmax_+1, i = imin_-1:imax_+2
            local_count += 1
            p_index[i, j, k] = local_count
    end
    # println(p_index[imin_-1:imax_+2,jmin_-1:jmax_+2,kmin_-1:kmax_+2])
    MPI.Barrier(comm)

    # update_borders!(p_index,mesh,par_env)
    
    p_maxo = -1
    p_mino = maximum(p_index)
    for k = kmin_-1:kmax_+1, j in jmin_-1:jmax_+1, i in imin_-1:imax_+2
        if p_index[i,j,k] != -1
            p_mino = min(p_mino,p_index[i,j,k])
            p_maxo = max(p_maxo,p_index[i,j,k])
        end
    end
    return p_mino,p_maxo

end

"""
Adds pressure pertubation required for Jacobian calculation
"""
function add_perturb!(P,delta,ii,jj,kk,mesh,par_env)
    @unpack imin, imax, jmin,jmax,kmin,kmax = mesh

    P[ii,jj,kk] += delta

    if ii == imin
        P[ii-1,jj,kk] = P[ii,jj,kk]
    end

    if ii == imax
        P[ii+1,jj,kk] = P[ii,jj,kk]
    end

    if jj == jmin
        P[ii,jj-1,kk] = P[ii,jj,kk]
    end

    if jj == jmax 
        P[ii,jj+1,kk] = P[ii,jj,kk]
    end

    if kk == kmin
        P[ii,jj,kk-1] =P[ii,jj,kk]
    end

    if kk == kmax
        P[ii,jj,kk+1] = P[ii,jj,kk]
    end
    return nothing
end


"""
Removes pressure pertubation required for Jacobian calculation
"""
function remove_perturb!(P,delta,ii,jj,kk,mesh,par_env)
    @unpack imin, imax, jmin,jmax,kmin,kmax = mesh

    P[ii,jj,kk] -= delta

    if ii == imin
        P[ii-1,jj,kk] = P[ii,jj,kk]
    end

    if ii == imax
        P[ii+1,jj,kk] = P[ii,jj,kk]
    end

    if jj == jmin
        P[ii,jj-1,kk] = P[ii,jj,kk]
    end

    if jj == jmax 
        P[ii,jj+1,kk] = P[ii,jj,kk]
    end

    if kk == kmin
        P[ii,jj,kk-1] =P[ii,jj,kk]
    end

    if kk == kmax
        P[ii,jj,kk+1] = P[ii,jj,kk]
    end
    return nothing
end

"""
code to grab the terminal velocity along centerline
"""

function term_vel(grav_cl,xo,yo,VF,D,param,mesh,par_env)
    @unpack x,y,dx,dy,xm,ym,zm,imin_,imax_,imin,imax,kmin_,kmax_,kmin,kmax,jmax_,jmin_,Lx,Ly,Lz = mesh
    @unpack VFhi,grav_x,grav_y,grav_z = param
    @unpack nproc,irank = par_env

    term_vel_height = zeros(nproc)


    # calculate grav angle in radians
    if grav_x > 0 && grav_y > 0
        m = grav_y/grav_x
        rads = atan(m)
    else
        #! odd number of cells in x and z-direction assumed
        mid_x = div(length(xm[imin:imax]),2)+1
        mid_z = div(length(zm[kmin:kmax]),2)+1

        if mid_x >= imin_ && mid_x <= imax_ && mid_z >= kmin_ && mid_z <= kmax_
            for j = jmax_:-1:jmin_
                if VF[mid_x,j,mid_z] < VFhi
                    term_vel_height[irank+1] = dy*(1-VF[mid_x,j,mid_z])+y[j]
                    return term_vel_height
                end
            end
        end
    end

    # m = tan(rads)
    b = yo-(m*xo)
    k_ind = div(kmax,2)+1

    for i in grav_cl 
        if VF[i[1],i[2],k_ind] < VFhi && D[i[1],i[2],k_ind] < Lx*Ly*Lz
            dh_y = 0.0
            dh_x = 0.0
            # determine if slope intercepts with x or y axis
            if y[i[2]] < (y_star=x[i[1]]*m+b) < y[i[2]+1]
                # then y-intercept
                
                d_star = y_star - y[i[2]]
                dx_p = dx*(1-VF[i[1],i[2],k_ind]) 
                dy_p = (dy-d_star)*(1-VF[i[1],i[2],k_ind])
                d = sqrt((x[i[1]]-xo)^2+(y_star-yo)^2)
                dh_y = dy_p/cos(rads) + d
                dh_x =dx_p/sin(rads) + d
                # println("y intercept value of $y_star with term vel height of  $((dh_y+dh_x)/2)")
            elseif x[i[1]] < (x_star=(y[i[2]]+b)/m) < x[i[1]+1]
                # then x-intercept
                d_star = x_star - x[i[1]]
                dx_p = (dx-d_star)*(1-VF[i[1],i[2],k_ind]) 
                dy_p = dy*(1-VF[i[1],i[2],k_ind])
                d = sqrt((x_star-xo)^2+(y[i[2]]-yo)^2)
                dh_y = dy_p/sin(rads) + d
                dh_x =dx_p/cos(rads) + d
                # println("x intercept value of $x_star with term vel height of  $((dh_y+dh_x)/2)")
            elseif x[i[1]]*m+b == y[i[2]]
                # then cell vertex is intercept
                # println("single gravity term")
                d_star = 0
                dy_p = dx*(1-VF[i[1],i[2],k_ind]) 
                dx_p = dy*(1-VF[i[1],i[2],k_ind])
                d = sqrt((x[i[1]]-xo)^2+(y[i[2]]-yo)^2)
                dh_y = dy_p/cos(rads) + d
                dh_x =dx_p/sin(rads) + d
            end

            if (new_height = (dh_x+dh_y)/2) > term_vel_height[irank+1]
                term_vel_height[irank+1] = new_height
            end
        end     
    end
    return term_vel_height
end

function bub_height(grav_cl,VF,xo,yo,zo,nx,ny,nz,D,mesh,param,par_env)
    @unpack x,y,dx,dy,xm,ym,zm,imin_,imax_,imin,imax,kmin_,kmax_,kmin,kmax,jmax_,jmin_,Lx,Ly,Lz = mesh
    @unpack VFhi,grav_x,grav_y,grav_z = param
    @unpack nproc,irank = par_env

    bubble_height = zeros(nproc)

    # calculate grav angle in radians
    if grav_x > 0 && grav_y > 0
        m = grav_y/grav_x
        a = atan(m)
    else
        #! odd number of cells in x and z-direction assumed
        mid_x = div(length(xm[imin:imax]),2)+1
        mid_z = div(length(zm[kmin:kmax]),2)+1

        if mid_x >= imin_ && mid_x <= imax_ && mid_z >= kmin_ && mid_z <= kmax_
            for j = jmax_:-1:jmin_
                if VF[mid_x,j,mid_z] < VFhi
                    bubble_height[irank+1] = dy*(1-VF[mid_x,j,mid_z])+y[j]
                    return bubble_height
                end
            end
        end
    end

    lo = [xo,yo]
    l = [1,a]
    k_ind = div(kmax,2)+1
    # println(grav_cl)
    po = zeros(3)
    for (i,j) in grav_cl
        # determine points on plane given D and 2 coordinates of that point
        if abs(D[i,j,k_ind]) < Lx+Ly+Lz
            # println("potential at $i,$j with VF of $(VF[i,j,k_ind])")
            if nx[i,j,k_ind] > ny[i,j,k_ind] && nx[i,j,k_ind] > nz[i,j,k_ind]
                y_mid = ym[j]
                z_mid = zm[k_ind]
                x_p = (D[i,j,k_ind] - ny[i,j,k_ind]*y_mid - nz[i,j,k_ind]*z_mid)/nx[i,j,k_ind]
                po = (x_p,y_mid,z_mid)
            elseif ny[i,j,k_ind] > nx[i,j,k_ind] && ny[i,j,k_ind] > nz[i,j,k_ind]
                x_mid = xm[i]
                z_mid = zm[k_ind]
                y_p = (D[i,j,k_ind] - nx[i,j,k_ind]*x_mid - nz[i,j,k_ind]*z_mid)/ny[i,j,k_ind]
                po = (x_mid,y_p,z_mid)
            elseif nz[i,j,k_ind] > ny[i,j,k_ind] && nz[i,j,k_ind] > ny[i,j,k_ind]
                y_mid = ym[j]
                x_mid = xm[i]
                z_p = (D[i,j,k_ind] - ny[i,j,k_ind]*y_mid - nx[i,j,k_ind]*x_mid)/nz[i,j,k_ind]
                po = (x_mid,y_mid,z_p)
            end
            # ensure line and plane are not parallel
            if (ldotn = l[1]*nx[i,j,k_ind] + a*ny[i,j,k_ind]) ≠ 0
                # solve for d 
                d = ((po[1]-xo)*nx[i,j,k_ind] + (po[2]-yo)*ny[i,j,k_ind] + (po[3]-zo)*nz[i,j,k_ind])/ldotn
                # calculate new height
                new_height = sqrt((l[1]*d)^2+(l[2]*d)^2)
                # new_height = sqrt((x_int-xo)^2+(y_int-yo)^2)
                
                # println("old height = $(bubble_height[irank+1])")
                # println("new height = $new_height occurs at $i,$j")
                # determine intersection point
                p = [xo+l[1]*d,yo+l[2]*d,zo]
                δ = 1.5*dx
                if x[i]-δ < p[1] < x[i+1]+δ && y[j]-δ < p[2] < y[j+1]+δ && bubble_height[irank+1] < new_height
                    bubble_height[irank+1] = new_height
                    return bubble_height
                # else
                #     println("new bubb intersection not within cell at $i, $j")
                #     println("intersection happens at $(p[1]) and $(p[2])")
                #     println("with x cell locs at $(x[i]) and $(x[i+1])")
                #     println("with y cell locs at $(y[j]) and $(y[j+1])")
                end
            end
        end
    end
    # error("stop")
    return bubble_height
end

function mask_maker!(mask,curve,mesh,param,par_env)
    @unpack imin_,imax_,imin,imax,kmin_,kmax_,kmin,kmax,jmax_,jmin_ = mesh

    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        if curve[i,j,k] > 0 || curve[i-1,j,k] > 0
            mask[i,j,k][1] = true
        end

        if curve[i,j,k] > 0 || curve[i,j-1,k] > 0
            mask[i,j,k][2] = true
        end

        if curve[i,j,k] > 0 || curve[i,j,k-1] > 0
            mask[i,j,k][3] = true
        end
    end
end

function hypreMat2JSON(jacob,cell1,cell2)
    #! used to grab jacobian rows at (12,13,11) & (14,13,11)
    #! output JSON dictionary is the input for the jacobian_check.jl function
    jacobians = Dict(
        cell1 => Vector{Tuple{String, Float64}}(),
        cell2 => Vector{Tuple{String, Float64}}()
        )
    cell1_ind = parse.(Int,split(cell1,","))
    cell2_ind = parse.(Int,split(cell2,","))
    count = 0
    for k in kmin_:kmax_,j in jmin_:jmax_, i in imin_:imax_
        count+=1
        int_x1 = zeros(1)
        HYPRE_IJMatrixGetValues(jacob,1,pointer(Int32.([1])),pointer(Int32.([p_index[cell1_ind[]]])),pointer(Int32.([p_index[i,j,k]])),int_x1)
        if int_x1[1] != 0.0
            push!(jacobians[cell1], ("$i,$j,$k",int_x1[1]))
        end
        int_x2 = zeros(1)
        HYPRE_IJMatrixGetValues(jacob,1,pointer(Int32.([1])),pointer(Int32.([p_index[cell2_ind[]]])),pointer(Int32.([p_index[i,j,k]])),int_x2)
        if int_x2[1] != 0.0
            push!(jacobians[cell2], ("$i,$j,$k",int_x2[1]))
        end
    end

    open("jacob_comp_dict_24tets.json","w") do file
        JSON.print(file,jacobians)
    end
end

function compute_kinEnergy(u,v,w,denx,deny,denz,mesh,param,par_env)
    @unpack dx,dy,dx,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    KE = 0.0

    for j in jmin_:jmax, i in imin_:imax_
        KE += 0.5*(denx[i,j,1]*u[i,j,1]^2+deny[i,j,1]*v[i,j,1]^2)*dx*dy
    end
    
    return KE
end

function curve_error(curve,ro,xo,yo,zo,param,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,x,y,z,xm,ym,zm = mesh

    curve_exact = 1/ro


end

# Identify cells that contain the centerline of gravity
function grav_centerline(xo,yo,mesh,param,par_env)
    @unpack dx,imin_,imax_,jmin_,jmax_,kmin_,kmax_,x,y,z,xm,ym,zm = mesh
    @unpack grav_x,grav_y,grav_z = param

    grav_cl = Tuple{Int64,Int64}[]

    δ = 0.5*dx
    # println(dx)
    #calculate angle of gravity, slope, and slope intercept
    if grav_x > 0 && grav_y > 0
        m = grav_y/grav_x
    else
        m = tan(π/2)
    end
    
    b = yo-(m*xo)
    # println(b)
    # println(m)
    for i = imin_:imax_
        # println("gets into mesh loop")
        if x[i] > xo  
            y_pos = m*(x[i]-δ) + b
            y_neg = m*(x[i+1]+δ) + b

            for j = jmin_:jmax_
                # if i == 44 && j == 44
                #     prinlnt()
                #identify cells at 
                if y_pos > (y[j]-δ) && y_pos < (y[j+1]+δ) 
                    if (i,j) ∉ grav_cl
                        push!(grav_cl,(i,j))
                    end
                elseif y_neg > (y[j]-δ) && y_neg < (y[j+1]+δ) 
                    if (i,j) ∉ grav_cl
                        push!(grav_cl,(i,j))
                    end
                end
            end
        end
    end

    for j = jmin_:jmax_
        if y[j] > yo
            x_pos = ((y[j]-δ)-b)/m
            x_neg = ((y[j+1]+δ)-b)/m

            for i=imin_:imax_
                if x_pos > (x[i]-δ) && x_pos < (x[i+1]+δ)
                    if (i,j) ∉ grav_cl
                        push!(grav_cl,(i,j))
                    end
                elseif x_neg > (x[i]-δ) && x_neg < (x[i+1]+δ)
                    if (i,j) ∉ grav_cl
                        push!(grav_cl,(i,j))
                    end
                end
            end
        end
    end

    sort!(grav_cl, by = x -> (x[2],x[1]), rev = true)
    # println(grav_cl)
    # error("stop")
    return grav_cl

end

