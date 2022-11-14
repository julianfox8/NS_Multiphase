struct mesh_struct
    x::Array{Float64,1}; y::Array{Float64,1};
    xm::Array{Float64,1}; ym::Array{Float64,1};
    dx::Float64; dy::Float64;
    imin::Int; imax::Int; jmin::Int; jmax::Int;
    Nx::Int; Ny::Int;
    Lx::Float64; Ly::Float64;
    # Parallel
    imin_::Int; imax_::Int; 
    jmin_::Int; jmax_::Int;
    imino_::Int; imaxo_::Int; 
    jmino_::Int; jmaxo_::Int;
    Nx_::Int; Ny_::Int;
    nghost::Int
    # VTK
    Gimin_::Vector{Int}; 
    Gimax_::Vector{Int};
    Gjmin_::Vector{Int}; 
    Gjmax_::Vector{Int};
end

function  create_mesh(Lx,Ly,Nx,Ny,par_env)
    # Index extents
    imin=2
    imax=Nx+1
    jmin=2
    jmax=Ny+1

    # Cell face arrays (1 more face than cells)
    x=zeros(Nx+3)
    y=zeros(Ny+3)
    x[imin:imax+1]=range(0,stop=Lx,length=Nx+1);
    y[jmin:jmax+1]=range(0,stop=Ly,length=Ny+1);

    # Cell size
    dx=x[imin+1]-x[imin];
    dy=y[jmin+1]-y[jmin];

    # Fill in ghost x and y values
    x[imin-1]=x[imin  ]-dx;
    x[imax+2]=x[imax+1]+dx;
    y[jmin-1]=y[jmin  ]-dy;
    y[jmax+2]=y[jmax+1]+dy;

    # Cell centers - Average of cell faces (including ghost cells)
    xm=zeros(Nx+2)
    ym=zeros(Ny+2)
    xm[1:imax+1]=0.5*(x[1:imax+1]+x[2:imax+2]);
    ym[1:jmax+1]=0.5*(y[1:jmax+1]+y[2:jmax+2]);


    # -------------
    # Parallel mesh
    # -------------
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

    # Create global extents for VTK output 
    Gimin_ = MPI.Allgather(imin_,comm)
    Gimax_ = MPI.Allgather(imax_,comm)
    Gjmin_ = MPI.Allgather(jmin_,comm)
    Gjmax_ = MPI.Allgather(jmax_,comm)
    
    # Put in struct
    mesh = mesh_struct(
        x,y,
        xm,ym,
        dx,dy,
        imin,imax,jmin,jmax,
        Nx,Ny,
        Lx,Ly,
        imin_,imax_,
        jmin_,jmax_,
        imino_,imaxo_,
        jmino_,jmaxo_,
        Nx_,Ny_,
        nghost,
        Gimin_,
        Gimax_,
        Gjmin_,
        Gjmax_,
        )

    return mesh::mesh_struct

end
