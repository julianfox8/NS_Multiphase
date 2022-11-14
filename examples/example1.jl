using NavierStokes_Parallel

# Define parameters 
param = parameters(
    # Constants
    mu=0.1,       # Dynamic viscosity
    rho=1.0,           # Density
    Lx=3.0,            # Domain size
    Ly=1.0,
    tFinal=1.0,      # Simulation time
    u_lef=100.0,
    u_bot=100.0,       # Boundary velocities
    u_top=100.0,
    u_rig=0.0,
    v_lef=0.0,

    # Discretization inputs
    Nx=5,           # Number of grid cells
    Ny=3,
    stepMax=1000,   # Maximum number of timesteps
    CFL=0.5,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    out_freq=200,    # Number of steps between when plots are updated

    # Processors 
    nprocx = 1,
    nprocy = 1,
)

# Simply run solver on 1 processor
run_solver(param)