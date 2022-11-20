using Parameters

@with_kw struct parameters 
    mu
    rho
    Lx
    Ly
    Lz
    tFinal
    Nx
    Ny
    Nz
    stepMax
    CFL
    out_freq
    tol
    nprocx
    nprocy
    nprocz
    xper
    yper
    zper
end