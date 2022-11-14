module NavierStokes_Parallel

export run_solver, parameters

using MPI
using UnPack
using OffsetArrays
using Plots

include("Parameters.jl")
include("Mesh.jl")
include("Parallel.jl")
include("Mask.jl")
include("Tools.jl")
include("BoundaryConditions.jl")
include("Pressure.jl")
include("Poisson.jl")
include("WriteData.jl")

function run_solver(param)

    # Create parallel environment
    par_env = parallel_init(param)

    # Create mesh
    mesh = create_mesh(param,par_env)

    # Create mask of object
    obj = mask_object(0.2,0.5,0.4,0.6)
    mask=mask_create(obj,mesh);

    # Create work arrays
    P,u,v = initArrays(mesh)

    # Call pressure Solver 
    pressure_solver!(P,mesh,par_env)

    # Plot pressure field
    plotArray("Pressure",P,mesh,par_env)
    printArray("Pressure",P,par_env)
    pvd = VTK_init()
    VTK(0,0.0,P,mesh,par_env,pvd)
    
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

# Finish VTK
VTK_finalize(pvd)

end # run_solver

end
