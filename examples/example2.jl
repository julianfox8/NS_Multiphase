using NavierStokes_Parallel

# Define parameters 
param = parameters(
    # Constants
    mu=0.1,       # Dynamic viscosity
    rho=1.0,           # Density
    Lx=3.0,            # Domain size
    Ly=1.0,
    Lz=1.0,
    tFinal=1.0,      # Simulation time
    
    # Discretization inputs
    Nx=5,           # Number of grid cells
    Ny=3,
    Nz=3,
    stepMax=1000,   # Maximum number of timesteps
    CFL=0.5,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    out_freq=200,    # Number of steps between when plots are updated

    # Processors 
    nprocx = 2,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = false,
)

obj = mask_object(0.2,0.5,0.4,0.6,0.4,0.6)

# Simply run solver on 1 processor
run_solver(param,obj)