using Parameters

@with_kw struct parameters 
    mu  :: Float64
    rho :: Float64
    Lx  :: Float64
    Ly  :: Float64
    Lz  :: Float64
    tFinal :: Float64
    Nx  :: Int64
    Ny  :: Int64
    Nz  :: Int64
    stepMax :: Int64
    CFL :: Float64
    out_freq :: Int64
    tol :: Float64
    nprocx :: Int64
    nprocy :: Int64
    nprocz :: Int64
    xper :: Bool
    yper :: Bool
    zper :: Bool
end