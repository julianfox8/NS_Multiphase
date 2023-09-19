"""
Example using inflow/outflow
"""

using NavierStokes_Parallel
using Random

#Domain set up similar to "Numerical simulation of a single rising bubble by VOF with surface compression"
# Define parameters 
param = parameters(
    # Constants
    mu_liq=0.01,       # Dynamic viscosity
    mu_gas = 0.0001,
    rho_liq= 1000,           # Density
    rho_gas =0.1, 
    sigma = 0.0072, #surface tension coefficient
    gravity = 1e-2,
    Lx=5.0,            # Domain size 
    Ly=5.0,
    Lz=1.0,
    tFinal=100.0,      # Simulation time
 
    
    # Discretization inputsc
    Nx=10,           # Number of grid cells
    Ny=10,
    Nz=1,
    stepMax=3,   # Maximum number of timesteps
    max_dt = 1e-2,
    CFL=0.4,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    std_out_period = 0.0,
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-6,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = true,

    # pressureSolver = "NLsolve",
    pressureSolver = "sparseSecant",
    iter_type = "standard",
    VTK_dir= "VTK_example_static_bubble2"

)

"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,VF,mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_,
                imin_,imax_,jmin_,jmax_,kmin_,kmax_,
                x,y,z,xm,ym,zm,Lx,Ly,Lz = mesh
    # Pressure
    # rand!(P,1:10)
    fill!(P,0.0)

    # Velocity
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 
        u[i,j,k] = 0.0 #-(ym[j] - Ly/2.0)
        v[i,j,k] = 0.0 # (xm[i] - Lx/2.0)
        w[i,j,k] = 0.0
    end

    # fill!(VF,1.0)
    # Volume Fraction
    rad=0.5
    xo=2.5
    yo=2.5
    zo = 2.5
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        # VF[i,j,k]=VFbubble3d(x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],rad,xo,yo,zo)
        VF[i,j,k]=VFbubble2d(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
    end

    return nothing    
end

"""
Boundary conditions for velocity
"""
function BC!(u,v,w,mesh,par_env)
    @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
    @unpack imin,imax,jmin,jmax,kmin,kmax = mesh
    @unpack jmin_,jmax_ = mesh
    @unpack xm,ym = mesh
    

     # Left 
     if irankx == 0 
        i = imin-1
        u[i,:,:] = -u[imin,:,:] # No flux
        v[i,:,:] = -v[imin,:,:] # slip
        w[i,:,:] = -w[imin,:,:] # No slip
    end
    # Right
    if irankx == nprocx-1
        i = imax+1
        u[i,:,:] = -u[imax,:,:] # No flux
        v[i,:,:] = -v[imax,:,:] # slip
        w[i,:,:] = -w[imax,:,:] # No slip
    end
    # Bottom 
    if iranky == 0 
        j = jmin-1
        u[:,j,:] = -u[:,jmin,:] # No slip
        v[:,j,:] = -v[:,jmin,:] # No flux
        w[:,j,:] = -w[:,jmin,:] # No slip
    end
    # Top
    utop=1.0
    if iranky == nprocy-1
        j = jmax+1
        u[:,j,:] = -u[:,jmax,:]  # No slip
        v[:,j,:] = -v[:,jmax,:]  # No flux
        w[:,j,:] = -w[:,jmax,:] # No slip
    end
    # Back 
    if irankz == 0 
        k = kmin-1
        u[:,:,k] = -u[:,:,kmin] # No slip
        v[:,:,k] = -v[:,:,kmin] # slip
        w[:,:,k] = -w[:,:,kmin] # No flux
    end
    # Front
    if irankz == nprocz-1
        k = kmax+1
        u[:,:,k] = -u[:,:,kmax] # No slip
        v[:,:,k] = -v[:,:,kmax] # slip
        w[:,:,k] = -w[:,:,kmax] # No flux
    end

    return nothing
end

"""
Apply outflow correction to region
"""
function outflow_correction!(correction,uf,vf,wf,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack iranky,nprocy = par_env
    # Top is the outflow
    if iranky == nprocy-1
        vf[:,jmax_+1,:] .+= correction 
    end
end

"""
Define area of outflow region
"""
function outflow_area(mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack x,z = mesh 
    myArea = (x[imax_+1]-x[imin_]) * (z[kmax_+1]-z[kmin_])
    return NavierStokes_Parallel.parallel_sum_all(myArea,par_env)
end
outflow =(area=outflow_area,correction=outflow_correction!)

# Simply run solver on 1 processor
@time run_solver(param, IC!, BC!,outflow)