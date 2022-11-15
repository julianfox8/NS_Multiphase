module NavierStokes_Parallel

export run_solver, parameters, mask_object, @unpack

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

function run_solver(param, IC!, BC!; mask_obj=nothing)

    # Create parallel environment
    par_env = parallel_init(param)

    # Create mesh
    mesh = create_mesh(param,par_env)

    # Create mask of object
    mask=mask_create(mask_obj,mesh);

    # Create work arrays
    P,u,v,w = initArrays(mesh)

    # Create initial condition
    IC!(P,u,v,w,mesh)

    # Apply boundary conditions
    BC!(u,v,w,mesh,par_env)

    # Update Processor boundaries
    update_borders!(u,mesh,par_env)
    update_borders!(v,mesh,par_env)
    update_borders!(w,mesh,par_env)


#    BC_apply!(u,"u",param,mesh,par_env)
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

# Call pressure Solver 
pressure_solver!(P,param,mesh,par_env)

# Plot pressure field
@unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
printArray("Pressure",P[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
printArray("Velocity-x",u,par_env)
# printArray("Velocity-y",v,par_env)
# printArray("Velocity-z",w,par_env)
pvd = VTK_init()
VTK(0,0.0,P,u,v,w,mesh,par_env,pvd)

# Finish VTK
VTK_finalize(pvd)

end # run_solver

end
