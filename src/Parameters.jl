using Parameters

@with_kw struct parameters 
    # Material properties
    mu  :: Float64
    rho :: Float64

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
    
    # Navier Stokes solver
    solveNS :: Bool = true 
    pressureSolver :: String = "ConjugateGradient" 
    tol :: Float64

    # Interface solver
    VFlo :: Float64 = 1e-10 
    VFhi :: Float64 = 1.0 - VFlo 
    normalMethod :: String = "ELVIRA" 
    # Velocity used for Vf transport when solveNS = false
    VFVelocity :: String  = "Nothing" 

    # Iterator type 
    # - standard : standard julia for loop  
    # - threads : parallelizes loop with .Threads library - not working 
    # - floop : parallelizes loop with FLoop
    iter_type :: String = "floop"


end