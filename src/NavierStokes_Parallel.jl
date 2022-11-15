"""
Soler based on
https://www.sciencedirect.com/science/article/pii/S0021999110001208?ref=cra_js_challenge&fr=RR-1
"""
module NavierStokes_Parallel

export run_solver, parameters, mask_object, @unpack

using MPI
using UnPack
using OffsetArrays
using Plots
using Printf

include("Parameters.jl")
include("Mesh.jl")
include("Parallel.jl")
include("Mask.jl")
include("Tools.jl")
include("BoundaryConditions.jl")
include("Velocity.jl")
include("Pressure.jl")
include("Poisson.jl")
include("WriteData.jl")

function run_solver(param, IC!, BC!; mask_obj=nothing)
    @unpack stepMax,tFinal = param

    # Create parallel environment
    par_env = parallel_init(param)

    # Create mesh
    mesh = create_mesh(param,par_env)

    # Create mask of object
    mask=mask_create(mask_obj,mesh);

    # Create work arrays
    P,u,v,w,us,vs,ws,Fx,Fy,Fz = initArrays(mesh)

    # Create initial condition
    t = 0.0
    IC!(P,u,v,w,mesh)

    # Apply boundary conditions
    BC!(u,v,w,mesh,par_env)

    # Update Processor boundaries
    update_borders!(u,mesh,par_env)
    update_borders!(v,mesh,par_env)
    update_borders!(w,mesh,par_env)

    # Loop over time
    nstep = 0
    while nstep<stepMax && t<tFinal

        # Update step counter
        nstep += 1

        # Compute timestep and update time
        dt = compute_dt(u,v,w,param,mesh,par_env)
        t += dt;

        # Predictor step
        predictor!(us,vs,ws,P,u,v,w,Fx,Fy,Fz,dt,param,mesh,par_env,mask)

        # Apply boundary conditions
        BC!(us,vs,ws,mesh,par_env)

        # Call pressure Solver 
        pressure_solver!(P,us,vs,ws,param,mesh,par_env)
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
end

# Plot pressure field
@unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
printArray("Pressure",P[imin_:imax_,jmin_:jmax_,kmin_:kmax_],par_env)
printArray("Velocity-x",us,par_env)
printArray("Velocity-y",vs,par_env)
# printArray("Velocity-z",w,par_env)
pvd = VTK_init()
VTK(0,0.0,P,u,v,w,mesh,par_env,pvd)

# Finish VTK
VTK_finalize(pvd)

end # run_solver

end
