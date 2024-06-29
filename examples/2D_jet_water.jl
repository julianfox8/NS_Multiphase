"""
Example using inflow/outflow
"""

using NavierStokes_Parallel
using Random

#Domain set up similar to "Numerical simulation of a single rising bubble by VOF with surface compression"
# Define parameters 
param = parameters(
    # Constants
    mu_liq=0.00009,       # Dynamic viscosity
    mu_gas = 0.0001,
    rho_liq= 1000,           # Density
    rho_gas =0.1, 
    sigma = 0.00072, #surface tension coefficient
    gravity = 10,
    Lx=0.04,            # Domain size 
    Ly=0.01,
    Lz=1/50,
    tFinal=100.0,      # Simulation time
 
    
    # Discretization inputs
    Nx=8,#Nx=64,           # Number of grid cells
    Ny=32,#Ny=256,
    Nz=1,
    stepMax=5000,   # Maximum number of timesteps
    max_dt = 1e-3,
    CFL=0.4,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    std_out_period = 0.0,
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-6,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = true,
    yper = false,
    zper = true,

    # pressureSolver = "NLsolve",
    # pressureSolver = "Secant",
    # pressureSolver = "sparseSecant",
    pressureSolver = "hypreSecant",
    # pressureSolver = "GaussSeidel",
    # pressureSolver = "ConjugateGradient",
    # pressure_scheme = "finite-difference",
    iter_type = "standard",
    VTK_dir= "VTK_example_static_bubble21"

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
    rad=0.0006
    xo=0.0
    yo=0.005
    zo = 0.005
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        # VF[i,j,k]=VFbubble3d(x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],rad,xo,yo,zo)
        VF[i,j,k]=VFbubble2d(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
    end

    return nothing    
end

"""
Boundary conditions for velocity
"""
function BC!(u,v,w,t,mesh,par_env)
    @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
    @unpack imin,imax,jmin,jmax,kmin,kmax,kmin_,kmax_,jmin_,jmax_,y = mesh
    @unpack jmin_,jmax_ = mesh
    @unpack xm,ym = mesh
    

     # Left
    amp = 0.25
    f=15
    if irankx == 0 
        i = imin-1
        for k=kmin_:kmax_,j=jmin_:jmax_
            if y[j] >= 0.004 && y[j] <= 0.006
                ujet = 5.0 + amp*sin(2*pi*t*f)
            else
                ujet = 0.0
            end
            u[i,j,k] = ujet - u[imin,j,k]
            v[i,j,k] = ujet - v[imin,j,k] 
            w[i,j,k] = ujet - w[imin,j,k]
            # u[i,j,k] = ujet
            # v[i,j,k] = ujet  
            # w[i,j,k] = ujet
        end        
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
    # # Back 
    # if irankz == 0 
    #     k = kmin-1
    #     u[:,:,k] = -u[:,:,kmin] # No slip
    #     v[:,:,k] = -v[:,:,kmin] # slip
    #     w[:,:,k] = -w[:,:,kmin] # No flux
    # end
    # # Front
    # if irankz == nprocz-1
    #     k = kmax+1
    #     u[:,:,k] = -u[:,:,kmax] # No slip
    #     v[:,:,k] = -v[:,:,kmax] # slip
    #     w[:,:,k] = -w[:,:,kmax] # No flux
    # end

    return nothing
end

"""
Apply outflow correction to region
"""
function outflow_correction!(correction,uf,vf,wf,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack irankx,nprocx = par_env
    # Right is the outflow
    if irankx == nprocx-1
        uf[imax_+1,:,:] .+= correction 
    end
end

"""
Define area of outflow region
"""
function outflow_area(mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack y,z = mesh 
    myArea = (y[jmax_+1]-y[jmin_]) * (z[kmax_+1]-z[kmin_])
    return NavierStokes_Parallel.parallel_sum_all(myArea,par_env)
end
outflow =(area=outflow_area,correction=outflow_correction!)

# Simply run solver on 1 processor
run_solver(param, IC!, BC!,outflow)
