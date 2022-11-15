struct mesh_struct
    x::Array{Float64,1}; y::Array{Float64,1}; z::Array{Float64,1};
    xm::Array{Float64,1}; ym::Array{Float64,1}; zm::Array{Float64,1};
    dx::Float64; dy::Float64; dz::Float64;
    imin::Int; imax::Int; jmin::Int; jmax::Int; kmin::Int; kmax::Int;
    Nx::Int; Ny::Int; Nz::Int;
    Lx::Float64; Ly::Float64; Lz::Float64;
    # Parallel
    imin_::Int; imax_::Int; 
    jmin_::Int; jmax_::Int;
    kmin_::Int; kmax_::Int;
    imino_::Int; imaxo_::Int; 
    jmino_::Int; jmaxo_::Int;
    kmino_::Int; kmaxo_::Int;
    Nx_::Int; Ny_::Int; Nz_::Int;
    nghost::Int
    # VTK
    Gimin_::Vector{Int}; 
    Gimax_::Vector{Int};
    Gjmin_::Vector{Int}; 
    Gjmax_::Vector{Int};
    Gkmin_::Vector{Int}; 
    Gkmax_::Vector{Int};
end

function  create_mesh(param,par_env)
    @unpack Nx,Ny,Nz,Lx,Ly,Lz = param
    # Index extents
    imin=2
    imax=Nx+1
    jmin=2
    jmax=Ny+1
    kmin=2
    kmax=Nz+1

    # Cell face arrays (1 more face than cells)
    x=zeros(Nx+3)
    y=zeros(Ny+3)
    z=zeros(Nz+3)
    x[imin:imax+1]=range(0,stop=Lx,length=Nx+1);
    y[jmin:jmax+1]=range(0,stop=Ly,length=Ny+1);
    z[kmin:kmax+1]=range(0,stop=Lz,length=Nz+1);

    # Cell size
    dx=x[imin+1]-x[imin];
    dy=y[jmin+1]-y[jmin];
    dz=z[kmin+1]-z[kmin];

    # Fill in ghost x and y values
    x[imin-1]=x[imin  ]-dx;
    x[imax+2]=x[imax+1]+dx;
    y[jmin-1]=y[jmin  ]-dy;
    y[jmax+2]=y[jmax+1]+dy;
    z[kmin-1]=z[kmin  ]-dz;
    z[kmax+2]=z[kmax+1]+dz;

    # Cell centers - Average of cell faces (including ghost cells)
    xm=zeros(Nx+2)
    ym=zeros(Ny+2)
    zm=zeros(Ny+2)
    xm[1:imax+1]=0.5*(x[1:imax+1]+x[2:imax+2]);
    ym[1:jmax+1]=0.5*(y[1:jmax+1]+y[2:jmax+2]);
    zm[1:kmax+1]=0.5*(z[1:kmax+1]+z[2:kmax+2]);

    # -------------
    # Parallel mesh
    # -------------
    @unpack nproc,nprocx,nprocy,nprocz,irankx,iranky,irankz,comm = par_env
    
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

    # Distribute mesh amongst process
    Nz_=floor(Nz/nprocz)
    extra=rem(Nz,nprocz)
    if (irankz < extra)
        Nz_=Nz_+1
    end
    kmin_ = kmin + irankz*floor(Nz/nprocz) + min(irankz,extra)
    kmax_ = kmin_ + Nz_ - 1

    # Add ghost cells
    nghost=1
    imino_=imin_-nghost
    imaxo_=imax_+nghost
    jmino_=jmin_-nghost
    jmaxo_=jmax_+nghost
    kmino_=kmin_-nghost
    kmaxo_=kmax_+nghost

    # Create global extents for VTK output 
    Gimin_ = MPI.Allgather(imin_,comm)
    Gimax_ = MPI.Allgather(imax_,comm)
    Gjmin_ = MPI.Allgather(jmin_,comm)
    Gjmax_ = MPI.Allgather(jmax_,comm)
    Gkmin_ = MPI.Allgather(kmin_,comm)
    Gkmax_ = MPI.Allgather(kmax_,comm)
    
    # Put in struct
    mesh = mesh_struct(
        x,y,z,
        xm,ym,zm,
        dx,dy,dz,
        imin,imax,jmin,jmax,kmin,kmax,
        Nx,Ny,Nz,
        Lx,Ly,Lz,
        imin_,imax_,
        jmin_,jmax_,
        kmin_,kmax_,
        imino_,imaxo_,
        jmino_,jmaxo_,
        kmino_,kmaxo_,
        Nx_,Ny_,Nz_,
        nghost,
        Gimin_,
        Gimax_,
        Gjmin_,
        Gjmax_,
        Gkmin_,
        Gkmax_,
        )

    return mesh::mesh_struct

end
