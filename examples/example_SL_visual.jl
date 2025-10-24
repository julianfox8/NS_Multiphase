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
    sigma = 1,
    gravity = 0.0,
    Lx=5.0,            # Domain size
    Ly=5.0,
    Lz=5.0,
    tFinal=8.0,      # Simulation time
    
    # Discretization inputs
    Nx=3,           # Number of grid cells
    Ny=3,
    Nz=3,
    stepMax=100,   # Maximum number of timesteps
    max_dt = 1e-3,
    CFL=0.4,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
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

    # Turn off NS solver
    solveNS = false,
    VFVelocity = "divFlow3D",

    # Iteration method used in @loop macro
    projection_method = "RK4",
    tesselation = "5_tets",

    
    iter_type = "standard",
    test_case = "SL_visualization_test"
)

"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,VF,mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack x,y,z = mesh
    @unpack xm,ym,zm = mesh

    # Velocity
    t=0.0
    u_fun(x,y,z,t) = 2*(x)
    v_fun(x,y,z,t) = -1*(5-y)
    w_fun(x,y,z,t) = -0.5*(5-z)
    # Set velocities (including ghost cells)
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
        v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
        w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
    end

    # Volume Fraction
    fill!(VF, 0.0)

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