"""
Example using 1 processors 

Run from terminal using >> mpiexecjl --project=. -np 1 julia examples/example1.jl
or REPL using julia> include("examples/example1.jl")
"""

using NavierStokes_Parallel 
const NS = NavierStokes_Parallel 

# Define parameters 
param = parameters(
    # Constants
    mu=1 ,       # Dynamic viscosity
    rho=1.0,           # Density
    Lx=1e3,            # Domain size
    Ly=1e3,
    Lz=1.0,
    tFinal=100.0,      # Simulation time

    # Discretization inputs
    Nx=10,           # Number of grid cells
    Ny=10,
    Nz=1,
    stepMax=200,   # Maximum number of timesteps
    CFL=0.5,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    out_freq=10,     # Number of steps between when plots are updated
    tol = 1e-10,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = false,
)



par_env = NS.parallel_init(param)
mesh = NS.create_mesh(param,par_env)

RHS=[0.0 0.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 10.0 10.0 10.0 10.0 10.0 10.0 0.0 0.0]

P,u,v,w,us,vs,ws,uf,vf,wf = NS.initArrays(mesh)

NS.conjgrad!(P,RHS,param,mesh,par_env)
