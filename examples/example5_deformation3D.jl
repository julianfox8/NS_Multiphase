"""
Example using VF 
"""

using NavierStokes_Parallel

# Define parameters 
param = parameters(
    # Constants
    mu=10.0,       # Dynamic viscosity
    rho=1.0,           # Density
    Lx=1.0,            # Domain size
    Ly=1.0,
    Lz=1.0,
    tFinal=8.0,      # Simulation time
    
    # Discretization inputs
    Nx=10,           # Number of grid cells
    Ny=10,
    Nz=10,
    stepMax=1, #0000,   # Maximum number of timesteps
    CFL=1,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-3,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = true,

    # Turn off NS solver
    solveNS = false,
    VFVelocity = "Deformation",
)

"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,VF,mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack x,y,z = mesh

    # Volume Fraction
    rad=0.25
    xo=0.5
    yo=0.5
    zo=0.5
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        VF[i,j,k]=VFsphere(x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],rad,xo,yo,zo)
    end

    return nothing    
end

"""
Boundary conditions for velocity
"""
function BC!(u,v,w,mesh,par_env)
    # Not needed when solveNS=false
    return nothing
end

# Simply run solver on 1 processor
run_solver(param, IC!, BC!)