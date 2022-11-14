
struct par_env_struct
    nprocx :: Int;
    nprocy :: Int;
    comm  :: MPI.Comm;
    irank :: Int;
    iroot :: Int;
    nproc :: Int;
    irankx :: Int;
    iranky :: Int;
end

function parallel_init(param)

    @unpack nprocx,nprocy = param

    # Start parallel environment
    if ~MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    irank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    # Check number of procs
    if nprocx*nprocy != nproc
        MPI.Finalize()

        error("Wrong number of processors, # procs is $nproc, example requires $nprocx x $nprocy")
    end


    # Cartesian grid of processor
    dims=Int32[nprocx,nprocy]
    peri=Int32[0,0]
    comm_2D = MPI.Cart_create(comm,2,dims,peri,false)
    coords = MPI.Cart_coords(comm_2D)
    irankx=coords[1]
    iranky=coords[2]

    # Root processor
    iroot=0

    par_env = par_env_struct(nprocx,nprocy,comm_2D,irank,iroot,nproc,irankx,iranky)
    
    return par_env
end

function create_mesh_par(mesh,par_env)

    @unpack imin,jmin,Nx,Ny = mesh 
    @unpack nproc,nprocx,nprocy,irankx,iranky,comm = par_env
    
    # Distribute mesh amongst process
    Nx_=floor(Nx/nprocx)
    extra=rem(Nx,nprocx)
    if (irankx < extra)
        Nx_=Nx_+1
    end
    imin_ = imin + irankx*floor(Nx/nprocx) + min(irankx,extra)
    imax_ = imin_ + Nx_ - 1

    # Distribute mesh amongst process
    Ny_=floor(Ny/nprocy)
    extra=rem(Ny,nprocy)
    if (iranky < extra)
        Ny_=Ny_+1
    end
    jmin_ = jmin + iranky*floor(Ny/nprocy) + min(iranky,extra)
    jmax_ = jmin_ + Ny_ - 1

    # Add ghost cells
    nghost=1
    imino_=imin_-nghost
    imaxo_=imax_+nghost
    jmino_=jmin_-nghost
    jmaxo_=jmax_+nghost

    # Create parallel mesh structure
    mesh_par=mesh_struct_par(imin_,imax_,jmin_,jmax_,imino_,imaxo_,jmino_,jmaxo_,Nx_,Ny_,nghost)

    # Create global extents for VTK output 
    Gimin_ = MPI.Allgather(imin_,comm)
    Gimax_ = MPI.Allgather(imax_,comm)
    Gjmin_ = MPI.Allgather(jmin_,comm)
    Gjmax_ = MPI.Allgather(jmax_,comm)
    Gmesh_par = global_mesh_struct_par(Gimin_,Gimax_,Gjmin_,Gjmax_)

    return mesh_par, Gmesh_par
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

end
function update_borders_x!(A,mesh_par,par_env)
    @unpack imin_,imax_,imino_,imaxo_,nghost = mesh_par
    @unpack comm = par_env
    
    # Send to left neighbor 
    sendbuf = parent(A[imin_:imin_+nghost-1,:])
    recvbuf = similar(sendbuf)
    isource,idest = MPI.Cart_shift(comm,0,-1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[imax_+1:imaxo_,:] = recvbuf

    # Send to right neighbor 
    sendbuf = parent(A[imax_-nghost+1:imax_,:])
    recvbuf = similar(sendbuf)
    isource,idest = MPI.Cart_shift(comm,0,+1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[imino_:imin_-1,:] = recvbuf

    return nothing
end
function update_borders_y!(A,mesh_par,par_env)
    @unpack jmin_,jmax_,jmino_,jmaxo_,nghost = mesh_par
    @unpack comm = par_env
    
    # Send to below neighbor 
    sendbuf = parent(A[:,jmin_:jmin_+nghost-1])
    recvbuf = similar(sendbuf)
    isource,idest = MPI.Cart_shift(comm,1,-1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,jmax_+1:jmaxo_,:] = recvbuf

    # Send to above neighbor 
    sendbuf = parent(A[:,jmax_-nghost+1:jmax_])
    recvbuf = similar(sendbuf)
    isource,idest = MPI.Cart_shift(comm,1,+1)
    data, status = MPI.Sendrecv!(sendbuf,recvbuf,comm,dest=idest,source=isource)
    A[:,jmino_:jmin_-1,:] = recvbuf

    return nothing
end


""" 
Parallel Sum of A (output goes to iroot)
"""
function parallel_sum(A,par_env)
    @unpack comm, iroot = par_env
    return MPI.Reduce(A,MPI.SUM,comm,root=iroot)
end

""" 
Parallel Max of A (by default output goes only to iroot)
"""
function parallel_max(A,par_env; recvProcs="iroot")
    @unpack comm, iroot = par_env
    if recvProcs == "iroot"
        return MPI.Reduce(A,MPI.MAX,comm,root=iroot)
    elseif recvProcs == "all"
        return MPI.Allreduce(A,MPI.MAX,comm)
    else
        error("Unknown recvProcs of $recvProcs")
    end
end
