using HYPRE.LibHYPRE
using MPI
using OffsetArrays

# MPI.Init()
# if ~MPI.Initialized()
    MPI.Init()
# end

comm = MPI.COMM_WORLD

nproc = MPI.Comm_size(comm)

# Processors 
nproc_x = 2
nproc_y = 1
nproc_z = 1
# Periodicity
xper = false
yper = true
zper = true


# # Check number of procs
# if nprocx*nprocy*nprocz != nproc
#     # MPI.Finalize()

#     error("Wrong number of processors, # procs is $nproc, example requires $nprocx x $nprocy x $nprocz")
# end


# Cartesian grid of processor
dims=Int32[nproc_x,nproc_y,nproc_z]
peri=Int32[xper,yper,zper]

comm_3D = MPI.Cart_create(comm,dims,peri,true)
irank = MPI.Comm_rank(comm_3D)
coords = MPI.Cart_coords(comm_3D)
irankx=coords[1]
iranky=coords[2]
irankz=coords[3]

function n(i,j,Nx) 
    val = i + (j-1)*Nx
    # @show i,j,k,Ny,Nz,val
    return val
end

jmin = 1
jmax = 4
imin = 1
imax = 4
F = 1
if irankx == 0
    imin_= 1
    imino_ = 0
    imax_ = 2
    imaxo_ = 3
    jmin_ = jmin
    jmino_ = jmin-1
    jmax_ = jmax
    jmaxo_ = jmax+1
end
if irankx == 1
    imin_ = 3
    imino_ = 2
    imax_ = 4
    imaxo_ = 5
    jmin_ = jmin
    jmino_ = jmin-1
    jmax_ = jmax
    jmaxo_ =jmax+1
end
Nx = 4

offset = [-Nx, -1, 0, 1, Nx] # Define the offset

function update_borders_x!(A,comm,imin_,imax_,imino_,imaxo_,rank)
    
    # Send to left neighbor
    sendbuf = OffsetArrays.no_offset_view(A[imin_,:])
    recvbuf = OffsetArrays.no_offset_view(A[imaxo_,:])
    isource,idest = MPI.Cart_shift(comm,0,-1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[imaxo_,:] = recvbuf

    # Send to right neighbor     
    sendbuf = OffsetArrays.no_offset_view(A[imax_,:])
    recvbuf = OffsetArrays.no_offset_view(A[imino_,:])
    isource,idest = MPI.Cart_shift(comm,0,+1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[imino_,:] = recvbuf


    return nothing
end


function prepare_indices(temp_index,comm,Nx,jmino_,imino_,jmaxo_,imaxo_,jmin_,imin_,jmax_,imax_,rank)
    ntemps = 0
    local_ntemps = 0
    for j = jmin_:jmax_,i = imin_:imax_
        # if mask[i,j] == 0 #not needed? (present in NGA hypre_amg.f90 )
            local_ntemps += 1
        # end
    end

    MPI.Allreduce!([local_ntemps], [ntemps], MPI.SUM, comm)

    ntemps_proc = zeros(Int, nproc)

    # MPI.Barrier(comm)

    MPI.Allgather!([local_ntemps], ntemps_proc, comm)

    ntemps_proc = cumsum(ntemps_proc)
    
    local_count = ntemps_proc[irankx+1] - local_ntemps
    for j = jmin_:jmax_, i = imin_:imax_
        # if mask[i, j] == 0
            local_count += 1
            temp_index[i, j] = local_count
        # end
    end
    # MPI.Barrier(comm)

    update_borders_x!(temp_index,comm,imin_,imax_,imino_,imaxo_,rank)
    
    temp_max = -1
    temp_min = maximum(temp_index)
    for j in jmin_:jmax_, i in imin_:imax_
        if temp_index[i,j] != -1
            temp_min = min(temp_min,temp_index[i,j])
            temp_max = max(temp_max,temp_index[i,j])
        end
    end
    return temp_min,temp_max

end

# mask = zeros(Int, imax_-imin_+1, jmax_-jmin_+1)
mask = OffsetArray{Int32}(undef,imino_:imaxo_,jmino_:jmaxo_); fill!(mask,0.0)

# coeff_index = zeros(Int, imax_-imin_+1, jmax_-jmin_+1)
coeff_index = OffsetArray{Int32}(undef,imino_:imaxo_,jmino_:jmaxo_); fill!(coeff_index,0.0)

temp_min,temp_max = prepare_indices(coeff_index,comm_3D,Nx,jmino_,imino_,jmaxo_,imaxo_,jmin_,imin_,jmax_,imax_,irankx)

# if irankx == 0
#     println("Proc 1 coeffecient index")
#     println(coeff_index)
# end
# MPI.Barrier(comm)
# if irankx == 1
#     println("Proc 2 coeffecient index")
#     println(coeff_index)
# end
# error("stop")

A_ref = Ref{HYPRE_IJMatrix}(C_NULL)
# HYPRE_IJMatrixCreate(comm,temp_min,temp_max,temp_min,temp_max,A_ref)
HYPRE_IJMatrixCreate(comm_3D,temp_min,temp_max,n(imin,jmin,Nx),n(imax,jmax,Nx),A_ref)
# HYPRE_IJMatrixCreate(comm,temp_min,temp_max,temp_min,temp_max,A_ref)
A = A_ref[]

HYPRE_IJMatrixSetObjectType(A,HYPRE_PARCSR)

HYPRE_IJMatrixInitialize(A)

function fill_matrix_coefficients!(matrix, coeff_index,offset, imin_, imax_, jmin_, jmax_, cols_, values_,irankx,jmino_,imino_,jmaxo_,imaxo_)
    nrows = 1
    nx = imax_-imin_+1
    ny = jmax_-jmin_+1
    for j = jmin_:jmax_,i = imin_:imax_
        if i == imin || i == imax || j == jmin || j == jmax
            fill!(cols_,0)
            fill!(values_,0.0)
            nst = 1
            cols_[nst] = coeff_index[i,j]
            values_[nst] = 1.0
            ncols = nst
            rows_ = coeff_index[i,j]
            HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
        elseif coeff_index[i,j] != -1
            fill!(cols_,0)
            fill!(values_,0.0)
            nst = 0
            
            # Diagonal
            nst += 1
            cols_[nst] = coeff_index[i,j]
            values_[nst] = 5.0

            for st in offset
                if st == 0
                    continue
                end
                # Left-Right
                if nx != 1
                    ni = i + st # Get neighbor index in the x direction
                    nj = j # No change in the y direction
                    if ni in imino_:imaxo_ && nj in jmino_:jmaxo_ && coeff_index[ni,nj] != -1
                        nst += 1
                        cols_[nst] = coeff_index[ni,nj]
                        values_[nst] = -1.0
                    end
                end
                
                # Top-Bottom
                if ny != 1
                    ni = i # No change in the x direction
                    nj = j + st # Get neighbor index in the y direction
                    if ni in imino_:imaxo_ && nj in jmino_:jmaxo_ && coeff_index[ni,nj] != -1
                        nst += 1
                        cols_[nst] = coeff_index[ni,nj]
                        values_[nst] = -1.0
                    end
                end
            end

            # Sort the points
            for st in 1:nst
                ind = st + argmin(cols_[st:nst], dims=1)[1] - 1
                tmpr = values_[st]
                values_[st] = values_[ind]
                values_[ind] = tmpr
                tmpi = cols_[st]
                cols_[st] = cols_[ind]
                cols_[ind] = tmpi
            end
            
            ncols = nst
            rows_ = coeff_index[i,j]

            # Call function to set matrix values
            HYPRE_IJMatrixSetValues(matrix, nrows, pointer(Int32.([ncols])), pointer(Int32.([rows_])), pointer(Int32.((cols_))), pointer(Float64.(values_)))
        end
    end
    # error("stop")
end

# cols_= zeros(Int,size(offset,1))
cols_ = OffsetArray{Int32}(undef,1:size(offset,1)); fill!(cols_,0)
# values_ = zeros(Float64,size(offset,1))
values_ = OffsetArray{Float64}(undef,1:size(offset,1)); fill!(values_,0.0)

fill_matrix_coefficients!(A,coeff_index,offset,imin_, imax_, jmin_, jmax_,cols_,values_,irankx,jmino_,imino_,jmaxo_,imaxo_)


HYPRE_IJMatrixAssemble(A)

HYPRE_IJMatrixPrint(A,"A")
MPI.Barrier(comm)

error("stop")


b_ref = Ref{HYPRE_IJVector}(C_NULL)
HYPRE_IJVectorCreate(comm,n(jmin_,imin_,Nx),n(jmax_,imax_,Nx),b_ref)
b = b_ref[]
HYPRE_IJVectorSetObjectType(b,HYPRE_PARCSR)
HYPRE_IJVectorInitialize(b)


for j in jmin_:jmax_,i in imin_:imax_
    HYPRE_IJVectorSetValues(b, 1, pointer([n(j,i,Nx)]), pointer([Float64(i)]))
end

HYPRE_IJVectorAssemble(b)

HYPRE_IJVectorPrint(b, "b")

# new_b = zeros(4)

# # HYPRE_IJVectorGetValues(b,1,[3],[new_b[3]])
# ib = zeros(1)
# for i in 1:4
    
#     HYPRE_IJVectorGetValues(b,1,[i],ib)
#     new_b[i] = ib[1]
# end
# println(new_b)

# MPI.Finalize()


