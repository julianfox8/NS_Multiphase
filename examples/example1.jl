"""
Example using 1 processors 

Run from terminal using >> mpiexecjl --project=. -np 1 julia examples/example1.jl
or REPL using julia> include("examples/example1.jl")
"""

using NavierStokes_Parallel

# Define parameters 
param = parameters(
    # Constants
    mu=0.1,       # Dynamic viscosity
    rho=1.0,           # Density
    Lx=3.0,            # Domain size
    Ly=3.0,
    Lz=1.0,
    tFinal=1.0,      # Simulation time

    # Discretization inputs
    Nx=10,           # Number of grid cells
    Ny=10,
    Nz=1,
    stepMax=10,   # Maximum number of timesteps
    CFL=0.5,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    out_freq=200,    # Number of steps between when plots are updated

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = false,
)

"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,
                xm,ym,zm,Lx,Ly,Lz = mesh
    # Pressure
    fill!(P,0.0)

    # Velocity
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_ 
        u[i,j,k] = -(ym[j] - Ly/2.0)
        v[i,j,k] =  (xm[i] - Lx/2.0)
        w[i,j,k] = 0.0
    end

    return nothing    
end

"""
Boundary conditions for velocity on cell faces
"""
function BC!(uf,vf,wf,mesh,par_env)
    @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
    @unpack imin,imax,jmin,jmax,kmin,kmax = mesh
    
    # Left 
    if irankx == 0 
        uf[imin  ,:,:] .= 0.0 # No flux
        vf[imin-1,:,:] = -vf[imin,:,:] # No slip
        wf[imin-1,:,:] = -wf[imin,:,:] # No slip
    end
    
    # Right
    if irankx == nprocx-1
        uf[imax+1,:,:] .= 0.0 # No flux
        vf[imax+1,:,:] = -vf[imax,:,:] # No slip
        wf[imax+1,:,:] = -wf[imax,:,:] # No slip
    end

    # Bottom 
    if iranky == 0 
        uf[:,jmin-1,:] = -uf[:,jmin,:] # No slip
        vf[:,jmin  ,:] .= 0.0 # No flux
        wf[:,jmin-1,:] = -wf[:,jmin,:] # No slip
    end
    
    # Top
    if iranky == nprocy-1
        uf[:,jmax+1,:] = -uf[:,jmax,:] # No slip
        vf[:,jmax+1,:] .= 0.0 # No flux
        wf[:,jmax+1,:] = -wf[:,jmax,:] # No slip
    end

    # Back 
    if irankz == 0 
        uf[:,:,kmin-1] = -uf[:,:,kmin] # No slip
        vf[:,:,kmin-1] = -vf[:,:,kmin] # No slip
        wf[:,:,kmin  ] .= 0.0 # No flux
    end
    
    # Front
    if irankz == nprocz-1
        uf[:,:,kmax+1] = -uf[:,:,kmax] # No slip
        vf[:,:,kmax+1] = -vf[:,:,kmax] # No slip
        wf[:,:,kmax+1] .= 0.0 # No flux
    end

    return nothing
end




# Simply run solver on 1 processor
run_solver(param, IC!, BC!)