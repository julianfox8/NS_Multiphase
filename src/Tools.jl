using Printf

function initArrays(mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Allocate memory
    P = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    u = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    v = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    w = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)

    return P,u,v,w
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
