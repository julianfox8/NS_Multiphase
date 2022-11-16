function initArrays(mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Allocate memory
    u  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(u ,0.0)
    v  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(v ,0.0)
    w  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(w ,0.0)
    us = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(us,0.0)
    vs = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(vs,0.0)
    ws = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(ws,0.0)
    uf = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(uf,0.0)
    vf = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(vf,0.0)
    wf = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(wf,0.0)
    P  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(P ,0.0)
    
    return P,u,v,w,us,vs,ws,uf,vf,wf
end

"""
Compute timestep 
"""
function compute_dt(u,v,w,param,mesh,par_env)
    @unpack mu,CFL = param
    @unpack dx,dy,dz = mesh

    # Convective Δt
    local_min_dx_vel = minimum([dx/maximum(u),dy/maximum(v),dz/maximum(w)])
    min_dx_vel= parallel_min(local_min_dx_vel,par_env,recvProcs="all")
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

    for k in axes(A,3)
        isroot && print("$text[:,:,$k]\n")
        for j in reverse(axes(A,2))
            for rankx in 0:nprocx-1
                if rankx == irankx 
                    for i in axes(A,1)
                        @printf("%10.3g ",A[i,j,k])
                    end
                end
                MPI.Barrier(comm)
            end
            isroot && print("\n")
        end
    end

    return nothing
end
