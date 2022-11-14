module NavierStokes_Parallel

export run_solver

using MPI
using UnPack
using OffsetArrays
using Plots

include("Geometry.jl")
include("Parallel.jl")
include("Mask.jl")
include("Tools.jl")
include("BoundaryConditions.jl")
include("Pressure.jl")
include("Poisson.jl")

function run_solver(nprocx,nprocy)

    # Constants
    mu=0.1       # Dynamic viscosity
    rho=1.0           # Density
    Lx=3.0            # Domain size
    Ly=1.0
    tFinal=1.0      # Simulation time
    u_lef=100.0
    u_bot=100.0       # Boundary velocities
    u_top=100.0
    v_rig=0.0
    v_lef=0.0

    # Discretization inputs
    Nx=5;           # Number of grid cells
    Ny=3;
    stepMax=1000;   # Maximum number of timesteps
    CFL=0.5;         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    out_freq=200;    # Number of steps between when plots are updated

    # Create the mesh
    mesh = create_mesh(Lx,Ly,Nx,Ny);

    # Create parallel environment
    par_env = parallel_init(nprocx,nprocy)

    # Create parallel mesh (partition)
    mesh_par = create_mesh_par(mesh,par_env)

    # Create mask of object
    obj = mask_object(0.2,0.5,0.4,0.6)
    mask=mask_create(obj,mesh,mesh_par);

    # Create work arrays
    P,u,v = initArrays(mesh_par)

    # Call pressure Solver 
    pressure_solver!(P,mesh,mesh_par,par_env)

    # Plot pressure field
    plotArray(P,mesh,mesh_par,par_env)

    printArray("Pressure",P,mesh_par,par_env)

#     # Precommpute Laplacian operator
#     L=Subfuns.lap_opp(mesh,mesh_par,mask);
#     Li=inv(L);
#
#     # Initial condition: t=0, u=0, v=0
#     t=0;
#     u=zeros(Nx+2,Ny+2); # 1 layer of ghost cells
#     v=zeros(Nx+2,Ny+2);
#
#     # Apply boundary conditions
#     Subfuns.applyBCs(u,v,u_bot,u_top,v_lef,v_rig,u_lef,mesh);
#
#     # Preallocate
#     us=zeros(Nx+2,Ny+2)
#     vs=zeros(Nx+2,Ny+2)
#     P =zeros(Nx+2,Ny+2)
#
#     # Loop over time
#     for nstep=1:stepMax # nstep<stepMax && t<tFinal
#
#         # Update step counter
#         nstep=nstep+1;
#
#         # Compute timestep and update time
#         dt=CFL*min(mesh.dx/maximum(maximum((u.^2+v.^2))),mesh.dx^2/mu);
#         t = t+dt;
#
#         # Predictor step
#         Subfuns.predictor(us,vs,u,v,mu,rho,dt,mesh,mask);
#
#         # Apply BCs on u* and v*
#         Subfuns.applyBCs(us,vs,u_bot,u_top,v_lef,v_rig,u_lef,mesh);
#
#
#         # Pressure Poisson equation
#         Subfuns.poisson(P,Li,us,vs,rho,dt,mesh,mask);
#
#         # Corrector step
#         Subfuns.corrector(u,v,us,vs,P,rho,dt,mesh,mask);
#
#         # Outputs
#         Subfuns.outputs(u,v,P,t,nstep,out_freq,mesh);
#
#         # Interpolate velocity to cell centers
#         uI=zeros(mesh.imax,mesh.jmax)
#         vI=zeros(mesh.imax,mesh.jmax)
#         uM=zeros(mesh.imax,mesh.jmax)
#         for j=mesh.jmin:mesh.jmax
#             for i=mesh.imin:mesh.imax
#                 uI[i,j]=0.5*(u[i,j]+u[i+1,j]);
#                 vI[i,j]=0.5*(v[i,j]+v[i,j+1]);
#                 uM[i,j]=sqrt(uI[i,j]^2+vI[i,j]^2)
#             end
#         end
#
#     end
#
#     Parallel.finalize()
#
# end

end # run_solver

end
