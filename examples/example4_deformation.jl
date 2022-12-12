"""
Example using VF 
"""

using NavierStokes_Parallel

# Define parameters 
param = parameters(
    # Constants
    mu=0.0,            # Dynamic viscosity
    rho=1.0,           # Density
    Lx=1.0,            # Domain size
    Ly=1.0,
    Lz=1/50,
    tFinal=8.0,      # Simulation time
    
    # Discretization inputs
    Nx=125,           # Number of grid cells
    Ny=125,
    Nz=1,
    stepMax=10000,   # Maximum number of timesteps
    max_dt = 0.008,
    CFL=10.9,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    out_period=10,     # Number of steps between when plots are updated
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

    # Iteration method used in @loop macro
    #iter_type = "standard",
    iter_type = "threads",
    #iter_type = "floop",
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
    u_fun(x,y,z,t) = -2(sin(π*x))^2*sin(π*y)*cos(π*y)*cos(π*t/8.0)
    v_fun(x,y,z,t) = +2(sin(π*y))^2*sin(π*x)*cos(π*x)*cos(π*t/8.0)
    w_fun(x,y,z,t) = 0.0
    # Set velocities (including ghost cells)
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
        v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
        w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
    end

    # Volume Fraction
    rad=0.15
    xo=0.5
    yo=0.75
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        VF[i,j,k]=VFcircle(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
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