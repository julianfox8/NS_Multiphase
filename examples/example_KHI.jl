"""
Example using 1 processors 

Run from terminal using >> mpiexecjl --project=. -np 1 julia examples/example_KHI.jl
or REPL using julia> include("examples/example_KHI.jl")
"""

using NavierStokes_Parallel
using Random 

# Define parameters
param = parameters(
    # Constants
    mu_liq= 0.001,       # Dynamic viscosity of liquid (N⋅s/m^2)
    mu_gas = 0.0015, # Dynamic viscosity of gas (N⋅s/m^2)
    rho_liq= 1000,  # Density of liquid (kg/m^3)
    rho_gas = 783,  # Density of gas (kg/m^3)
    sigma = 0.0295, # surface tension coefficient (N/m)
    grav_x = 0.706, # Gravity (m/s^2)
    grav_y = 9.785,#9.8, # Gravity (m/s^2)
    grav_z = 0.0, # Gravity (m/s^2)
    Lx=0.3,            # Domain size
    Ly=0.03,
    Lz=1/100,
    tFinal=5.0,      # Simulation time

    # Discretization inputs
    Nx=601,          # Number of grid cells
    Ny=61,
    Nz=1,
    stepMax=100000,   # Maximum number of timesteps
    CFL=0.2,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    max_dt = 1e-3,
    std_out_period = 0.0,
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-8,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = true,
    yper = false,
    zper = false,

    # solveNS = false,

    # pressure_scheme = "semi-lagrangian",
    # pressureSolver = "res_iteration",
    # pressureSolver = "hypreSecant",
    # pressurePrecond = "nl_jacobi",

    pressure_scheme = "finite-difference",
    # pressureSolver = "congugateGradient",
    pressureSolver = "FC_hypre",
    # pressureSolver = "gauss-seidel",

    hypreSolver = "LGMRES",
    mg_lvl = 1,

    instability = "kelvin-helmholtz", # Type of instability to simulate

    # Iteration method used in @loop macro
    iter_type = "standard",
    # iter_type = "floop",
    test_case = "KHI_gforced_test",
) 


"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,VF,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,
                xm,ym,y,Lx,Ly,Lz,dy = mesh
    # Pressure
    fill!(P,0.0)

    # Velocity & Volume Fraction
    # A = 0.0001
    # n = 4
    # k1 = 2π*n/Lx
    # ϕ = 0.0
    # VFkhi(VF, mesh, A, k1, ϕ)
    # u0 = 0.5
    
    Random.seed!(1234) # for reproducibility
    
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 
        
        #height
        height = Ly/2 + (2 * rand()-1)*Ly*1e-3  # add small perturbation on order of 1e-2*Ly to initial interface

        if y[j] < height && y[j+1] > height 
            VF[i,j,k] = (height - y[j])/(y[j+1]-y[j])
        elseif y[j] < height
            VF[i,j,k] = 1.0
        else
            VF[i,j,k] = 0.0
        end

        # u[i,j,k] = u0*tanh((ym[j]-Ly/2-A*sin(k1*xm[i]))/(4*dy))  # Smooth transition between the two layers
        u[i,j,k] = 0.0
        # v[i,j,k] = A*2π/λ*sin(2π*xm[i]/λ)*u[i,j,k]
        v[i, j, k] = 0.0  
        w[i,j,k] = 0.0
        
    end

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
        u[i,:,:] = u[imax,:,:] # periodic
        v[i,:,:] = v[imax,:,:] # No slip
        w[i,:,:] = -w[imin,:,:] # No slip
    end
    # Right
    if irankx == nprocx-1
        i = imax+1
        u[i,:,:] = u[imin,:,:] #periodic
        v[i,:,:] = v[imin,:,:] # No slip
        w[i,:,:] = -w[imax,:,:] # No slip
    end
    # Bottom 
    if iranky == 0 
        j = jmin-1
        u[:,j,:] .= -u[:,jmin,:] # No slip
        v[:,j,:] .= -v[:,jmin,:] # No slip
        w[:,j,:] .= -w[:,jmin,:] # No slip
    end
    # Top
    if iranky == nprocy-1
        j = jmax+1
        u[:,j,:] .= -u[:,jmax,:] # No slip
        v[:,j,:] .= -v[:,jmax,:] # No slip
        w[:,j,:] .= -w[:,jmax,:] # No slip
    end
    # Back 
    if irankz == 0 
        k = kmin-1
        u[:,:,k] = -u[:,:,kmin] # No slip
        v[:,:,k] = -v[:,:,kmin] # No slip
        w[:,:,k] = -w[:,:,kmin] # No slip
    end
    # Front
    if irankz == nprocz-1
        k = kmax+1
        u[:,:,k] = -u[:,:,kmax] # No slip
        v[:,:,k] = -v[:,:,kmax] # No slip
        w[:,:,k] = -w[:,:,kmax] # No slip
    end

    return nothing
end

# Simply run solver on 1 processor
run_solver(param, IC!, BC!)
