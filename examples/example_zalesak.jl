"""
Example using inflow/outflow
"""

using NavierStokes_Parallel
using Random

#Domain set up similar to "Numerical simulation of a single rising bubble by VOF with surface compression"
# Define parameters 
param = parameters(
    # Constants
    mu_liq=0.0,       # Dynamic viscosity of liquid (N/m)
    mu_gas = 0.0, # Dynamic viscosity of gas (N/m)
    rho_liq= 1.0,           # Density of liquid (kg/m^3)
    rho_gas = 1.0,  # Density of gas (kg/m^3)
    sigma = 0.0, # surface tension coefficient (N/m)
    grav_x = 0.0,
    grav_y = 0.0,
    grav_z = 0.0, # Gravity (m/s^2)
    Lx=1.0,            # Domain size of 8Dx30Dx8D where D is bubble diameter(m)
    Ly=1.0,             
    Lz=1/50,
    tFinal=1.0,      # Simulation time

    
    # Discretization inputs
    Nx=50,           # Number of grid cells
    Ny=50,
    Nz=1,
    stepMax=1000,   # Maximum number of timesteps
    max_dt = 1e-2,
    CFL=5.0,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    std_out_period = 0.0,
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-7,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = true,

    # Restart  
    # restart = true,
    # restart_itr = 190,

    solveNS = false,
    VFVelocity = "rotation",

    pressure_scheme = "semi-lagrangian",
    # pressureSolver = "hypreSecant",
    # pressureSolver = "hypreSecantLS",
    pressureSolver = "res_iteration",

    # pressureSolver = "Ostrowski",
    # pressureSolver = "SOR",
    # pressureSolver = "SecantSOR",

    # pressureSolver = "FC_hypre",
    # pressure_scheme = "finite-difference",
    # pressureSolver = "gauss-seidel",
    mg_lvl = 1,

    projection_method = "RK4",
    tesselation = "5_tets",

    
    iter_type = "standard",
    test_case = "Zalesak_FD", 

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

    # fill!(VF,1.0)
    # Volume Fraction
    rad=0.15
    xo=0.5
    yo=0.75
    slot_width = 0.05
    slot_length = 0.24
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        VF[i,j,k]=VFzalesak2d(x[i],x[i+1],y[j],y[j+1],rad,xo,yo,slot_width,slot_length)
        
    end

    return xo,yo
end

"""
Boundary conditions for velocity
"""
function BC!(u,v,w,mesh,par_env)
    @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
    @unpack imin,imax,jmin,jmax,kmin,kmax = mesh
    @unpack jmin_,jmax_ = mesh
    @unpack xm,ym = mesh
    
     # Left 
     if irankx == 0 
        i = imin-1
        u[i,:,:] = -u[imin,:,:] # No flux
        v[i,:,:] = -v[imin,:,:] # slip
        w[i,:,:] = -w[imin,:,:] # No slip
    end
    # Right
    if irankx == nprocx-1
        i = imax+1
        u[i,:,:] = -u[imax,:,:] # No flux
        v[i,:,:] = -v[imax,:,:] # slip
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
        v[:,:,k] = -v[:,:,kmin] # slip
        w[:,:,k] = -w[:,:,kmin] # No flux
    end
    # Front
    if irankz == nprocz-1
        k = kmax+1
        u[:,:,k] = -u[:,:,kmax] # No slip
        v[:,:,k] = -v[:,:,kmax] # slip
        w[:,:,k] = -w[:,:,kmax] # No flux
    end

    return nothing
end

# Simply run solver on 1 processor
@time run_solver(param, IC!, BC!)
