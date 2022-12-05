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

    # CFL Number (for time step)
    CFL :: Float64

    # Number of iterations between writing output files
    out_period :: Int64
   
    # Procs
    nprocx :: Int64
    nprocy :: Int64
    nprocz :: Int64

    # Periodicity
    xper :: Bool
    yper :: Bool
    zper :: Bool
    
    # Navier Stokes solver
    solveNS = true :: Bool
    pressureSolver = "ConjugateGradient" :: String
    tol :: Float64

    # Interface solver
    VFlo = 1e-10 :: Float64
    VFhi = 1.0 - VFlo :: Float64
    normalMethod = "ELVIRA" :: String
    # Velocity used for Vf transport when solveNS = false
    VFVelocity = "Nothing" :: String 

end