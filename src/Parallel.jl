using OffsetArrays

struct par_env_struct
    nprocx :: Int; nprocy :: Int; nprocz :: Int;
    comm  :: MPI.Comm; nproc :: Int;
    irank :: Int; iroot :: Int; isroot :: Bool;
    irankx :: Int; iranky :: Int; irankz :: Int;
end

function parallel_init(param)

    @unpack nprocx,nprocy,nprocz,xper,yper,zper = param

    # Start parallel environment
    if ~MPI.Initialized()
        MPI.Init()
    end

    

    comm = MPI.COMM_WORLD
    nproc = MPI.Comm_size(comm)

    # Check number of procs
    if nprocx*nprocy*nprocz != nproc
        MPI.Finalize()

        error("Wrong number of processors, # procs is $nproc, example requires $nprocx x $nprocy x $nprocz")
    end


    # Cartesian grid of processor
    dims=Int32[nprocx,nprocy,nprocz]
    peri=Int32[xper,yper,zper]
    comm_3D = MPI.Cart_create(comm,dims,peri,true)
    irank = MPI.Comm_rank(comm_3D)
    coords = MPI.Cart_coords(comm_3D)
    irankx=coords[1]
    iranky=coords[2]
    irankz=coords[3]

    # Root processor
    iroot = 0 
    isroot = irank == iroot

    par_env = par_env_struct(
        nprocx,nprocy,nprocz,
        comm_3D,nproc,
        irank,iroot,isroot,
        irankx,iranky,irankz
        )
    
    return par_env
end

function parallel_finalize()
    MPI.Finalize()
end

""" 
Update ghost cells of A on parallel boundaries
"""
function update_borders!(A,mesh_par,par_env)
    
    update_borders_x!(A,mesh_par,par_env)
    update_borders_y!(A,mesh_par,par_env)
    update_borders_z!(A,mesh_par,par_env)

end

function update_VF_borders!(A,mesh_par,par_env)
    VFghost = 3
    update_borders_x!(A,mesh_par,par_env,VFghost)
    update_borders_y!(A,mesh_par,par_env,VFghost)
    update_borders_z!(A,mesh_par,par_env,VFghost)

end
function update_borders_x!(A,mesh_par,par_env,VFghost=nothing)
    @unpack imin_,imax_,imino_,imaxo_,nghost = mesh_par
    @unpack comm = par_env
    if VFghost !== nothing
        nghost == VFghost
    end
    # Send to left neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[imin_:imin_+nghost-1,:,:])
    recvbuf = OffsetArrays.no_offset_view(A[imax_+1:imaxo_,:,:])
    isource,idest = MPI.Cart_shift(comm,0,-1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[imax_+1:imaxo_,:,:] = recvbuf

    # Send to right neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[imax_-nghost+1:imax_,:,:])
    recvbuf = OffsetArrays.no_offset_view(A[imino_:imin_-1,:,:])
    isource,idest = MPI.Cart_shift(comm,0,+1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[imino_:imin_-1,:,:] = recvbuf

    return nothing
end
function update_borders_y!(A,mesh_par,par_env,VFghost=nothing)
    @unpack jmin_,jmax_,jmino_,jmaxo_,nghost = mesh_par
    @unpack comm = par_env
    if VFghost !== nothing
        nghost == VFghost
    end
    # Send to below neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[:,jmin_:jmin_+nghost-1,:])
    recvbuf = OffsetArrays.no_offset_view(A[:,jmax_+1:jmaxo_,:])
    isource,idest = MPI.Cart_shift(comm,1,-1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,jmax_+1:jmaxo_,:] = recvbuf

    # Send to above neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[:,jmax_-nghost+1:jmax_,:])
    recvbuf = OffsetArrays.no_offset_view(A[:,jmino_:jmin_-1,:])
    isource,idest = MPI.Cart_shift(comm,1,+1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,jmino_:jmin_-1,:] = recvbuf

    return nothing
end
function update_borders_z!(A,mesh_par,par_env,VFghost=nothing)
    @unpack kmin_,kmax_,kmino_,kmaxo_,nghost = mesh_par
    @unpack comm = par_env
    if VFghost !== nothing
        nghost == VFghost
    end
    # Send to below neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[:,:,kmin_:kmin_+nghost-1])
    recvbuf = OffsetArrays.no_offset_view(A[:,:,kmax_+1:kmaxo_])
    isource,idest = MPI.Cart_shift(comm,2,-1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,:,kmax_+1:kmaxo_] = recvbuf

    # Send to above neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[:,:,kmax_-nghost+1:kmax_])
    recvbuf = OffsetArrays.no_offset_view(A[:,:,kmino_:kmin_-1])
    isource,idest = MPI.Cart_shift(comm,2,+1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,:,kmino_:kmin_-1] = recvbuf

    return nothing
end

"""
Update ghost cells of A on parallel boundaries for face centered values
"""
function update_xface_borders!(A,mesh_par,par_env)
    
    update_face_borders_x!(A,mesh_par,par_env)
    update_borders_y!(A,mesh_par,par_env)
    update_borders_z!(A,mesh_par,par_env)

end

function update_yface_borders!(A,mesh_par,par_env)
    
    update_borders_x!(A,mesh_par,par_env)
    update_face_borders_y!(A,mesh_par,par_env)
    update_borders_z!(A,mesh_par,par_env)

end

function update_zface_borders!(A,mesh_par,par_env)
    
    update_borders_x!(A,mesh_par,par_env)
    update_borders_y!(A,mesh_par,par_env)
    update_face_borders_z!(A,mesh_par,par_env)

end

function update_face_borders_x!(A,mesh_par,par_env)
    @unpack imin_,imax_,imino_,imaxo_,nghost = mesh_par
    @unpack comm = par_env

    # Send to left neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[imin_+1:imin_+nghost-1,:,:])
    recvbuf = OffsetArrays.no_offset_view(A[imax_+2:imaxo_,:,:])
    isource,idest = MPI.Cart_shift(comm,0,-1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[imax_+2:imaxo_,:,:] = recvbuf

    # Send to right neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[imax_-nghost+1:imax_,:,:])
    recvbuf = OffsetArrays.no_offset_view(A[imino_:imin_-1,:,:])
    isource,idest = MPI.Cart_shift(comm,0,+1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[imino_:imin_-1,:,:] = recvbuf

    return nothing
end
function update_face_borders_y!(A,mesh_par,par_env)
    @unpack jmin_,jmax_,jmino_,jmaxo_,nghost = mesh_par
    @unpack comm = par_env

    # Send to below neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[:,jmin_+1:jmin_+nghost-1,:])
    recvbuf = OffsetArrays.no_offset_view(A[:,jmax_+2:jmaxo_,:])
    isource,idest = MPI.Cart_shift(comm,1,-1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,jmax_+2:jmaxo_,:] = recvbuf

    # Send to above neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[:,jmax_-nghost+1:jmax_,:])
    recvbuf = OffsetArrays.no_offset_view(A[:,jmino_:jmin_-1,:])
    isource,idest = MPI.Cart_shift(comm,1,+1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,jmino_:jmin_-1,:] = recvbuf

    return nothing
end
function update_face_borders_z!(A,mesh_par,par_env)
    @unpack kmin_,kmax_,kmino_,kmaxo_,nghost = mesh_par
    @unpack comm = par_env

    # Send to below neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[:,:,kmin_+1:kmin_+nghost-1])
    recvbuf = OffsetArrays.no_offset_view(A[:,:,kmax_+2:kmaxo_])
    isource,idest = MPI.Cart_shift(comm,2,-1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,:,kmax_+2:kmaxo_] = recvbuf

    # Send to above neighbor 
    sendbuf = OffsetArrays.no_offset_view(A[:,:,kmax_-nghost+1:kmax_])
    recvbuf = OffsetArrays.no_offset_view(A[:,:,kmino_:kmin_-1])
    isource,idest = MPI.Cart_shift(comm,2,+1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,:,kmino_:kmin_-1] = recvbuf

    return nothing
end


"""
Wrapper for MPI.Reduce and MPI.Allreduce
"""
function callReduce(A,par_env,type)
    @unpack comm, iroot = par_env
    return MPI.Reduce(A,type,comm,root=iroot)
end
function callReduce_all(A,par_env,type)
    @unpack comm, iroot = par_env
    return MPI.Allreduce(A,type,comm)
end

""" 
Parallel Sum of A (by default output goes only to iroot)
"""
function parallel_sum(A,par_env)
    return callReduce(sum(A),par_env,MPI.SUM)
end
function parallel_sum_all(A,par_env)
    return callReduce_all(sum(A),par_env,MPI.SUM)
end

""" 
Parallel Max of A (by default output goes only to iroot)
"""
function parallel_max(A,par_env)
    return callReduce(maximum(A),par_env,MPI.MAX)
end
function parallel_max_all(A,par_env)
    return callReduce_all(maximum(A),par_env,MPI.MAX)
end

""" 
Parallel Min of A (by default output goes only to iroot)
"""
function parallel_min(A,par_env)
    return callReduce(minimum(A),par_env,MPI.MIN)
end
function parallel_min_all(A,par_env)
    return callReduce_all(minimum(A),par_env,MPI.MIN)
end


""" 
Parallel Mean of A (by default output goes only to iroot)
"""
function parallel_mean(A,par_env)
    return parallel_sum(A,par_env)/parallel_sum(length(A),par_env)
end
function parallel_mean_all(A,par_env)
    return parallel_sum_all(A,par_env)/parallel_sum_all(length(A),par_env)
end
