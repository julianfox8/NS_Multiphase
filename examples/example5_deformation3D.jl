"""
Example using VF 
"""

using NavierStokes_Parallel

# Define parameters 
param = parameters(
    # Constants
    mu_liq=1.0,            # Dynamic viscosity
    mu_gas = 0.01,
    rho_liq=1.0,           # Density
    rho_gas = 0.01,
    sigma = 0.0,
    gravity = 0.0,
    Lx=1.0,            # Domain size
    Ly=1.0,
    Lz=1.0,
    tFinal=3.0,      # Simulation time
    
    # Discretization inputs
    Nx=64,           # Number of grid cells
    Ny=64,
    Nz=64,
    max_dt = 1.0/64/2.0*0.9,
    stepMax=10000,   # Maximum number of timesteps
    CFL=0.4 ,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-3,

    # Processors 
    nprocx = 2,
    nprocy = 2,
    nprocz = 2,

    # Periodicity
    xper = false,
    yper = false,
    zper = false,

    # Turn off NS solver
    solveNS = false,
    VFVelocity = "Deformation3D",
    VTK_dir= "VTK_example_static_bubble32_nosurf",


    pressureSolver = "FC_hypre",
    pressure_scheme = "finite-difference",

    iter_type = "standard"
)

"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,VF,mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack x,y,z = mesh

    # Volume Fraction
    rad=0.15
    xo=0.35
    yo=0.35
    zo=0.35
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        VF[i,j,k]=VFsphere(x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],rad,xo,yo,zo)
    end

    return nothing    
end

"""
Boundary conditions for velocity
"""
function BC!(u,v,w,t,mesh,par_env)
    # Not needed when solveNS=false
    return nothing
end

# Simply run solver on 1 processor
run_solver(param, IC!, BC!)