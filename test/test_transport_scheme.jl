using NavierStokes_Parallel
using Random

# Define parameters for the Zalesak's Disk Test Case
param = parameters(
    # Constants
    mu_liq=1.0,       # Dynamic viscosity of liquid (N/m)
    mu_gas = 0.0,     # Dynamic viscosity of gas (ignored in this case)
    rho_liq= 1.0,     # Density of liquid (kg/m^3)
    rho_gas = 0.0,    # Density of gas (ignored in this case)
    sigma = 0.0,      # Surface tension coefficient (Ns/m^2, not used here)
    gravity = 0.0,    # Gravity (m/s^2, not used here)
    Lx=1.0,           # Domain size (length in x-direction)
    Ly=1.0,           # Domain size (length in y-direction)
    Lz=0.01,          # Thin 2D slice in z-direction
    tFinal=1.0,       # Simulation time (1 full rotation)

    # Discretization inputs
    Nx=100,           # Number of grid cells in x
    Ny=100,           # Number of grid cells in y
    Nz=1,             # Single slice in z for 2D test
    stepMax=1000,     # Maximum number of timesteps
    max_dt = 0.01,
    CFL=0.4,          # Courant-Friedrichs-Lewy (CFL) condition
    std_out_period = 0.0,
    out_period=10,    # Steps between outputs
    tol = 1e-5,

    # Processors
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = true,
    yper = true,
    zper = false,

    solveNS = false,
    
    iter_type = "standard",
    VTK_dir = "VTK_zalesak_test"
)

"""
Initial conditions for Zalesak's Disk
"""
function IC!(P, u, v, w, VF, mesh)
    @unpack imino_, imaxo_, jmino_, jmaxo_, kmino_, kmaxo_,
                x, y, z, Lx, Ly = mesh
    
    # Initialize pressure
    fill!(P, 0.0)

    # Initialize velocity (rotational flow)
    xc = 0.5 * Lx  # Center of rotation
    yc = 0.5 * Ly
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_
        u[i, j, k] = -(y[j] - yc)  # Rotational velocity field
        v[i, j, k] = x[i] - xc
        w[i, j, k] = 0.0
    end

    # Initialize volume fraction (Zalesak's Disk)
    radius = 0.15       # Radius of the disk
    slot_width = 0.05   # Width of the slot
    slot_length = 0.15  # Length of the slot

    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_
        x_center = x[i] - xc
        y_center = y[j] - yc
        r = sqrt(x_center^2 + y_center^2)

        # Check if inside the circular region
        if r <= radius
            # Check if outside the slot
            if !(x[i] >= xc - slot_width / 2 && x[i] <= xc + slot_width / 2 && y[j] < yc && y[j] >= yc - slot_length)
                VF[i, j, k] = 1.0  # Inside the disk but outside the slot
            else
                VF[i, j, k] = 0.0  # Inside the slot
            end
        else
            VF[i, j, k] = 0.0  # Outside the disk
        end
    end
end

"""
Boundary conditions for velocity
"""
function BC!(u, v, w, t, mesh, par_env)
    # Not needed when solveNS = false
    return nothing
end

# Run the solver
@time run_solver(param, IC!, BC!, outflow)
