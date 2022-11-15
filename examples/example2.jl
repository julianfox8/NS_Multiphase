"""
Example using 2 processors 

Run from terminal using >> mpiexecjl --project=. -np 2 julia examples/example2.jl
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
    stepMax=1000,   # Maximum number of timesteps
    CFL=0.5,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    out_freq=200,    # Number of steps between when plots are updated

    # Processors 
    nprocx = 2,
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
Boundary conditions for velocity 
"""
function BC!(u,v,w,mesh,par_env)
    @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
    @unpack imin_,  imax_,  jmin_,  jmax_,  kmin_,  kmax_ = mesh
    @unpack imino_, imaxo_, jmino_, jmaxo_, kmino_, kmaxo_ = mesh

    # Left 
    if irankx == 0 
        for i in imino_:imin_-1 # Loop over ghost cells
            u[i,:,:] = -u[imin_,:,:] # No flux
            v[i,:,:] = -v[imin_,:,:] # No slip
            w[i,:,:] = -w[imin_,:,:] # No slip
        end
    end
    
    # Right
    if irankx == nprocx-1
        for i in imax_+1:imaxo_ # Loop over ghost cells
            u[i,:,:] = -u[imax_,:,:] # No flux
            v[i,:,:] = -v[imax_,:,:] # No slip
            w[i,:,:] = -w[imax_,:,:] # No slip
        end
    end

    # Bottom 
    if iranky == 0 
        for j in jmino_:jmin_-1 # Loop over ghost cells
            u[:,j,:] = -u[:,jmin_,:] # No flux
            v[:,j,:] = -v[:,jmin_,:] # No slip
            w[:,j,:] = -w[:,jmin_,:] # No slip
        end
    end
    
    # Top
    if iranky == nprocy-1
        for j in jmax_+1:jmaxo_ # Loop over ghost cells
            u[:,j,:] = -u[:,jmax_,:] # No flux
            v[:,j,:] = -v[:,jmax_,:] # No slip
            w[:,j,:] = -w[:,jmax_,:] # No slip
        end
    end

    # Back 
    if irankz == 0 
        for k in kmino_:kmin_-1 # Loop over ghost cells
            u[:,:,k] = -u[:,:,kmin_] # No flux
            v[:,:,k] = -v[:,:,kmin_] # No slip
            w[:,:,k] = -w[:,:,kmin_] # No slip
        end
    end
    
    # Front
    if irankz == nprocz-1
        for k in kmax_+1:kmaxo_ # Loop over ghost cells
            u[:,:,k] = -u[:,:,kmax_] # No flux
            v[:,:,k] = -v[:,:,kmax_] # No slip
            w[:,:,k] = -w[:,:,kmax_] # No slip
        end
    end

    return nothing
end

# Simply run solver on 1 processor
run_solver(param, IC!, BC!)