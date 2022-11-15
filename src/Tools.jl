function initArrays(mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Allocate memory
    P  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    u  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    v  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    w  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    us = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    vs = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    ws = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    Fx = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    Fy = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    Fz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    return P,u,v,w,us,vs,ws,Fx,Fy,Fz
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
