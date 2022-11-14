struct mesh_struct
    x::Array{Float64,1}; y::Array{Float64,1};
    xm::Array{Float64,1}; ym::Array{Float64,1};
    dx::Float64; dy::Float64;
    imin::Int; imax::Int; jmin::Int; jmax::Int;
    Nx::Int; Ny::Int;
    Lx::Float64; Ly::Float64;
end

function  create_mesh(Lx::Float64,Ly::Float64,Nx::Int64,Ny::Int64)
    #CREATE_MESH creates a staggerred grid
    #   Inputs
    #     Lx x Ly : Domain size
    #     Nx x Ny : Number of grid cells
    #   Outputs
    #     mesh : contains all variables in mesh_structy type
    #  Mesh contains "ghost" cells (1 layer of cells on outside that are used
    #  to apply boundary conditions

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

    mesh = mesh_struct(x,y,xm,ym,dx,dy,imin,imax,jmin,jmax,Nx,Ny,Lx,Ly)

    return mesh::mesh_struct

end
