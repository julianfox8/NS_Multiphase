using Parameters

@with_kw struct parameters 
    # Material properties

    mu_liq  :: Float64
    mu_gas :: Float64
    rho_liq :: Float64
    rho_gas :: Float64
    sigma :: Float64
    gravity :: Float64

    # Domain size
    Lx  :: Float64
    Ly  :: Float64
    Lz  :: Float64

    # Simulation time
    tFinal :: Float64

    # Grid points
    Nx  :: Int64
    Ny  :: Int64
    Nz  :: Int64

    # Maximum iterations
    stepMax :: Int64

    # Time step
    max_dt :: Float64 = Inf 
    CFL :: Float64

    # Number of iterations between writing output files
    std_out_period :: Float64 = 1.0
    out_period :: Int64
    VTK_dir :: String = "VTK" 
   
    # Procs
    nprocx :: Int64
    nprocy :: Int64
    nprocz :: Int64

    # Periodicity
    xper :: Bool
    yper :: Bool
    zper :: Bool

    # Restart simulation
    restart :: Bool = false
    restart_itr :: Int64 = 0

    
    # Navier Stokes solver
    solveNS :: Bool = true 
    # pressureSolver :: String = "ConjugateGradient"
    pressureSolver :: String = "Secant" 
    tol :: Float64
    pressure_scheme :: String = "semi-lagrangian"

    # Interface solver
    VFlo :: Float64 = 1e-10 
    VFhi :: Float64 = 1.0 - VFlo 
    normalMethod :: String = "ELVIRA" 
    # Velocity used for Vf transport when solveNS = false
    VFVelocity :: String  = "Nothing" 
    projection_method :: String = "RK4"

    # Iterator type 
    # - standard : standard julia for loop  
    # - threads : parallelizes loop with .Threads library - not working 
    # - floop : parallelizes loop with FLoop
    iter_type :: String = "floop"


end