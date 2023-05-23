"""
Example using inflow/outflow
"""

using NavierStokes_Parallel
using Random

#Domain set up similar to "Numerical simulation of a single rising bubble by VOF with surface compression"
# Define parameters 
param = parameters(
    # Constants
    mu=10.0,       # Dynamic viscosity
    rho_liq=1.0,           # Density
    rho_gas =0.1, 
    sigma = 0.1, #surface tension coefficient
    Lx=1.0,            # Domain size
    Ly=1.0,
    Lz=1/50,
    tFinal=1.0,      # Simulation time
    
    # Discretization inputs
    Nx=10,           # Number of grid cells
    Ny=10,
    Nz=10,
    stepMax=2,   # Maximum number of timesteps
    CFL=0.1,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    std_out_period = 0.0,
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-3,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = false,

    pressureSolver = "NLsolve",
    iter_type = "standard",
    VTK_dir= "VTK_example_static_bubble1"

)

"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,VF,mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_,
                imin_,imax_,jmin_,jmax_,kmin_,kmax_,
                x,y,z,xm,ym,zm,Lx,Ly,Lz = mesh
    # Pressure
    # rand!(P,1:10)
    fill!(P,0.0)

    # Velocity
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 
        u[i,j,k] = 0.0 #-(ym[j] - Ly/2.0)
        v[i,j,k] = 0.0 # (xm[i] - Lx/2.0)
        w[i,j,k] = 0.0
    end

    # Volume Fraction
    rad=0.15
    xo=0.5
    yo=0.5
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        VF[i,j,k]=VFcircle(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
    end

    return nothing    
end

"""
Boundary conditions for velocity
"""
function BC!(u,v,w,mesh,par_env)
    @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
    @unpack imin,imax,jmin,jmax,kmin,kmax = mesh
    @unpack jmin_,jmax_ = mesh
    @unpack xm,ym = mesh
    
    vsides = 0.0
     # Left 
     if irankx == 0 
        i = imin-1
        u[i,:,:] = -u[imin,:,:] # No flux
        v[i,:,:] = -v[imin,:,:] .+ vsides # slip
        w[i,:,:] = -w[imin,:,:] # No slip
    end
    # Right
    if irankx == nprocx-1
        i = imax+1
        u[i,:,:] = -u[imax,:,:] # No flux
        v[i,:,:] = -v[imax,:,:] .+ vsides # slip
        w[i,:,:] = -w[imax,:,:] # No slip
    end
    # Bottom 
    if iranky == 0 
        j = jmin-1
        u[:,j,:] = -u[:,jmin,:] # No slip
        v[:,j,:] = -v[:,jmin,:] # No flux
        w[:,j,:] = -w[:,jmin,:] # No slip
    end
    # Top
    utop=1.0
    if iranky == nprocy-1
        j = jmax+1
        u[:,j,:] = -u[:,jmax,:]  # No slip
        v[:,j,:] = -v[:,jmax,:]  # No flux
        w[:,j,:] = -w[:,jmax,:] # No slip
    end
    # Back 
    if irankz == 0 
        k = kmin-1
        u[:,:,k] = -u[:,:,kmin] # No slip
        v[:,:,k] = -v[:,:,kmin] .+ vsides # slip
        w[:,:,k] = -w[:,:,kmin] # No flux
    end
    # Front
    if irankz == nprocz-1
        k = kmax+1
        u[:,:,k] = -u[:,:,kmax] # No slip
        v[:,:,k] = -v[:,:,kmax] .+ vsides # slip
        w[:,:,k] = -w[:,:,kmax] # No flux
    end

    return nothing
end

# Simply run solver on 1 processor
run_solver(param, IC!, BC!)