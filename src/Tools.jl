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

# struct vec_floats
#     value::Vector{Float64}
# end

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

    tets  = Array{Float64}(undef, 3, 4, 17)
    inds  = Array{Int32}(undef, 3, 4, 17)
    verts = Array{Float64}(undef, 3, 8)
    vInds = Array{Int32}(undef, 3, 8)

    return P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,divg,tets,verts,inds,vInds

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

# """
# Density/Viscosity calculation
# """
function compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack rho_liq,mu_liq,rho_gas,mu_gas,gravity = param

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

function term_vel(uf,vf,wf,VF,param,mesh,par_env)
    @unpack y,dy,xm,ym,zm,imin_,imax_,jmin_,jmax_,kmin_,kmax_ =mesh
    @unpack VFhi,Nx,Ny,Nz = param
    term_vel_height = 0
    terminal_vel = 0
    #! odd number of cells in x and z-direction assumed
    mid_x = div(length(xm[imin_:imax_]),2)+1
    mid_z = div(length(zm[kmin_:kmax_]),2)+1
    
    for j = jmax_:-1:jmin_
        if VF[mid_x,j,mid_z] < VFhi
            term_vel_height = dy*(1-VF[mid_x,j,mid_z])+y[j]
            terminal_vel = vf[mid_x,j,mid_z] + (vf[mid_x,j+1,mid_z]-vf[mid_x,j,mid_z])/(y[j+1]-y[j])*(term_vel_height-y[j])
            return term_vel_height,terminal_vel
        end
    end
    return term_vel_height,terminal_vel
end

