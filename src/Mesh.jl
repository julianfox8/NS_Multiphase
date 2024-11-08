struct mesh_struct
    x  :: OffsetArrays.OffsetVector{Float64, Vector{Float64}};
    y  :: OffsetArrays.OffsetVector{Float64, Vector{Float64}};
    z  :: OffsetArrays.OffsetVector{Float64, Vector{Float64}};
    xm :: OffsetArrays.OffsetVector{Float64, Vector{Float64}};
    ym :: OffsetArrays.OffsetVector{Float64, Vector{Float64}};
    zm :: OffsetArrays.OffsetVector{Float64, Vector{Float64}};
    dx :: Float64;
    dy :: Float64;
    dz :: Float64;
    imin :: Int64; 
    imax :: Int64; 
    jmin :: Int64; 
    jmax :: Int64; 
    kmin :: Int64; 
    kmax :: Int64;
    Nx :: Int64; 
    Ny :: Int64; 
    Nz :: Int64;
    Lx :: Float64; 
    Ly :: Float64; 
    Lz :: Float64;
    # Parallel
    imin_ :: Int64; 
    imax_ :: Int64; 
    jmin_ :: Int64; 
    jmax_ :: Int64;
    kmin_ :: Int64; 
    kmax_ :: Int64;
    imino_ :: Int64; 
    imaxo_ :: Int64; 
    jmino_ :: Int64; 
    jmaxo_ :: Int64;
    kmino_ :: Int64; 
    kmaxo_ :: Int64;
    Nx_ :: Int64; 
    Ny_ :: Int64; 
    Nz_ :: Int64;
    nghost :: Int64;
    # VTK
    Gimin_ :: Vector{Int64};
    Gimax_ :: Vector{Int64};
    Gjmin_ :: Vector{Int64};
    Gjmax_ :: Vector{Int64};
    Gkmin_ :: Vector{Int64};
    Gkmax_ :: Vector{Int64};
    Gimino_ :: Vector{Int64};
    Gimaxo_ :: Vector{Int64};
    Gjmino_ :: Vector{Int64};
    Gjmaxo_ :: Vector{Int64};
    Gkmino_ :: Vector{Int64};
    Gkmaxo_ :: Vector{Int64};
end

function  create_mesh(param,par_env)
    @unpack Nx,Ny,Nz,Lx,Ly,Lz = param
    # Index extents 
    imin=1
    imax=Nx
    jmin=1
    jmax=Ny
    kmin=1
    kmax=Nz

    # Define number of ghost cells
    nghost = 3

    # Index extents with ghost cells
    imino=imin-nghost
    imaxo=imax+nghost
    jmino=jmin-nghost
    jmaxo=jmax+nghost
    kmino=kmin-nghost
    kmaxo=kmax+nghost 

    # Cell face arrays (1 more face than cells)
    #x=zeros(Nx+3)
    #y=zeros(Ny+3)
    #z=zeros(Nz+3)
    x=OffsetArray{Float64}(undef,imino:imaxo+1)
    y=OffsetArray{Float64}(undef,jmino:jmaxo+1)
    z=OffsetArray{Float64}(undef,kmino:kmaxo+1)
    x[imin:imax+1]=range(0,stop=Lx,length=Nx+1);
    y[jmin:jmax+1]=range(0,stop=Ly,length=Ny+1);
    z[kmin:kmax+1]=range(0,stop=Lz,length=Nz+1);

    # Cell size (assuming uniform!)
    dx=x[imin+1]-x[imin];
    dy=y[jmin+1]-y[jmin];
    dz=z[kmin+1]-z[kmin];

    # Fill in ghost x and y values
    for n = 1:nghost
        x[imin-n]  =x[imin]  -n*dx;
        x[imax+1+n]=x[imax+1]+n*dx;
        y[jmin-n]  =y[jmin]  -n*dy;
        y[jmax+1+n]=y[jmax+1]+n*dy;
        z[kmin-n]  =z[kmin]  -n*dz;
        z[kmax+1+n]=z[kmax+1]+n*dz;
    end

    # Cell centers - Average of cell faces (including ghost cells)
    xm=OffsetArray{Float64}(undef,imino:imaxo)
    ym=OffsetArray{Float64}(undef,jmino:jmaxo)
    zm=OffsetArray{Float64}(undef,kmino:kmaxo)
    xm[imino:imaxo]=0.5*(x[imino:imaxo]+x[imino+1:imaxo+1]);
    ym[jmino:jmaxo]=0.5*(y[jmino:jmaxo]+y[jmino+1:jmaxo+1]);
    zm[kmino:kmaxo]=0.5*(z[kmino:kmaxo]+z[kmino+1:kmaxo+1]);

    # -------------
    # Parallel mesh
    # -------------
    @unpack nproc,nprocx,nprocy,nprocz,irankx,iranky,irankz,comm = par_env
    
    # Distribute mesh amongst process
    Nx_=Int(floor(Nx/nprocx))
    extra=rem(Nx,nprocx)
    if (irankx < extra)
        Nx_=Nx_+1
    end
    imin_ = imin + irankx*Int(floor(Nx/nprocx)) + min(irankx,extra)
    imax_ = imin_ + Nx_ - 1

    # Distribute mesh amongst process
    Ny_=Int(floor(Ny/nprocy))
    extra=rem(Ny,nprocy)
    if (iranky < extra)
        Ny_=Ny_+1
    end
    jmin_ = jmin + iranky*Int(floor(Ny/nprocy)) + min(iranky,extra)
    jmax_ = jmin_ + Ny_ - 1

    # Distribute mesh amongst process
    Nz_=Int(floor(Nz/nprocz))
    extra=rem(Nz,nprocz)
    if (irankz < extra)
        Nz_=Nz_+1
    end
    kmin_ = kmin + irankz*Int(floor(Nz/nprocz)) + min(irankz,extra)
    kmax_ = kmin_ + Nz_ - 1

    # Add ghost cells
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
    Gimino_ = MPI.Allgather(imino_,comm)
    Gimaxo_ = MPI.Allgather(imaxo_,comm)
    Gjmino_ = MPI.Allgather(jmino_,comm)
    Gjmaxo_ = MPI.Allgather(jmaxo_,comm)
    Gkmino_ = MPI.Allgather(kmino_,comm)
    Gkmaxo_ = MPI.Allgather(kmaxo_,comm)
    
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
        Gimino_,
        Gimaxo_,
        Gjmino_,
        Gjmaxo_,
        Gkmino_,
        Gkmaxo_,
        )

    return mesh::mesh_struct

end
