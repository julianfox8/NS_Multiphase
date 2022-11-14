using Parameters

@with_kw struct parameters 
    mu
    rho
    Lx
    Ly
    tFinal
    u_lef 
    u_bot
    u_top
    u_rig
    v_lef
    Nx
    Ny
    stepMax
    CFL
    out_freq
    nprocx
    nprocy
end