
function initArrays(mesh_par)
    @unpack imino_,imaxo_,jmino_,jmaxo_ = mesh_par

    # Allocate memory
    P = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_)
    u = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_)
    v = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_)

    # Fill
    fill!(P,0.0)
    fill!(u,0.0)
    fill!(v,0.0)

    return P,u,v
end

function plotArray(text,A,mesh,mesh_par,par_env)
    @unpack x,y = mesh
    @unpack imin_,imax_,jmin_,jmax_ = mesh_par
    @unpack irank = par_env

    # Remove OffestArray
    xl = parent(x[imin_:imax_])
    yl = parent(y[jmin_:jmax_])
    Al = parent(A[imin_:imax_,jmin_:jmax_])

    printArray("Al",Al,mesh_par,par_env)

    # Make contour plot of this processor's portion
    plt=contourf(xl,yl,Al')
    plt=title!(text)
    savefig("plot_$irank.png")
    return nothing
end

"""
Prints a parallel array
"""
function printArray(text,A,mesh_par,par_env)
    @unpack irankx,nprocx,nprocy,irank,iroot,comm = par_env

    nprocy > 1 && error("printArray only works with 1 proc in y")

    irank == iroot && print("$text\n")
    for j in reverse(axes(A,2))
        for rankx in 0:nprocx-1
            if rankx == irankx 
                print(A[:,j])
            end
            MPI.Barrier(comm)
        end
        irank == iroot && print("\n")
    end

    return nothing
end
