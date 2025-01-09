"""
Example using 1 processors 

Run from terminal using >> mpiexecjl --project=. -np 1 julia examples/example1.jl
or REPL using julia> include("examples/example1.jl")
"""

using NavierStokes_Parallel

# Define parameters
param = parameters(
    # Constants
    mu_liq=1.265,       # Dynamic viscosity of liquid (N/m)
    mu_gas = 1.79e-5, # Dynamic viscosity of gas (N/m)
    rho_liq= 1346.5,           # Density of liquid (kg/m^3)
    rho_gas =1.225,  # Density of gas (kg/m^3)
    sigma = 0.0,#0.0769, # surface tension coefficient (N/m^2)
    gravity = 0.0, #9.8, # Gravity (m/s^2)
    Lx=1.0,            # Domain size
    Ly=1.0,
    Lz=1.0/50,
    tFinal=100.0,      # Simulation time

    # Discretization inputs
    Nx=250,           # Number of grid cells
    Ny=250,
    Nz=1,
    stepMax=200,   # Maximum number of timesteps
    CFL=0.2,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
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


    pressure_scheme = "semi-lagrangian",
    pressureSolver = "hypreSecant",
    # pressure_scheme = "finite-difference",
    # pressureSolver = "FC_hypre",
    # Iteration method used in @loop macro
    iter_type = "standard",
    # iter_type = "floop",
    test_case = "lid_driven_cavity_Re_1000"
) 

"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,VF,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,
                xm,ym,zm,Lx,Ly,Lz = mesh
    # Pressure
    
    fill!(P,0.0)

    # Velocity
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 
        u[i,j,k] = 0.0 #-(ym[j] - Ly/2.0)
        v[i,j,k] = 0.0 # (xm[i] - Lx/2.0)
        w[i,j,k] = 0.0
    end

    # Volume Fraction
    fill!(VF,1.0)

    return nothing    
end

"""
Boundary conditions for velocity
"""
function BC!(u,v,w,t,mesh,par_env)
    @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
    @unpack imin,imax,jmin,jmax,kmin,kmax = mesh
    
     # Left 
     if irankx == 0 
        i = imin-1
        u[i,:,:] = -u[imin,:,:] # No flux
        v[i,:,:] = -v[imin,:,:] # No slip
        w[i,:,:] = -w[imin,:,:] # No slip
    end
    # Right
    vright=0.0
    if irankx == nprocx-1
        i = imax+1
        u[i,:,:] = -u[imax,:,:] # No flux
        v[i,:,:] = -v[imax,:,:] .+ 2vright # No slip
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
        u[:,j,:] = -u[:,jmax,:] .+ 2utop # No slip
        v[:,j,:] = -v[:,jmax,:] # No flux
        w[:,j,:] = -w[:,jmax,:] # No slip
    end
    # Back 
    if irankz == 0 
        k = kmin-1
        u[:,:,k] = -u[:,:,kmin] # No slip
        v[:,:,k] = -v[:,:,kmin] # No slip
        w[:,:,k] = -w[:,:,kmin] # No flux
    end
    # Front
    if irankz == nprocz-1
        k = kmax+1
        u[:,:,k] = -u[:,:,kmax] # No slip
        v[:,:,k] = -v[:,:,kmax] # No slip
        w[:,:,k] = -w[:,:,kmax] # No flux
    end

    return nothing
end

# Simply run solver on 1 processor
run_solver(param, IC!, BC!)
