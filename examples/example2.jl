"""
Example using 2 processors 

Run from terminal using >> mpiexecjl --project=. -np 2 julia examples/example2.jl
"""

using NavierStokes_Parallel

# Define parameters 
param = parameters(
    # Constants
    # mu_liq=0.01,       # Dynamic viscosity 
    # mu_gas = 0.0001,``
    # rho_liq= 1000,           # Density
    # rho_gas =0.1, 
    mu_liq=1.0,       # Dynamic viscosity 
    mu_gas = 1.0,
    rho_liq= 11.0,           # Density
    rho_gas =1.0, 
    sigma = 1.0,#0.0072, #surface tension coefficient
    gravity = 1e-3,
    Lx=5.0,            # Domain size 
    Ly=5.0,
    Lz=1/50,
    tFinal=100.0,  
    
    # Discretization inputs
    Nx=10,           # Number of grid cells
    Ny=10,
    Nz=1,
    stepMax=2,   # Maximum number of timesteps
    max_dt = 5e-3,
    CFL=0.1,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    std_out_period = 0.0,
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-6,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = true,
    # VTK_dir= "VTK_example_2"

    # pressureSolver = "GaussSeidel",
    pressureSolver = "ConjugateGradient",
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
    fill!(VF,0.0)

    return nothing    
end

"""
Boundary conditions for velocity
"""
function BC!(u,v,w,mesh,par_env)
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
    vright=1.0
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
    utop=0.0
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