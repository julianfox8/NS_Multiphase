    """
Example using inflow/outflow
"""

using NavierStokes_Parallel

# Define parameters 
param = parameters(
    # Constants
    mu_liq=0.01,       # Dynamic viscosity
    mu_gas = 0.0001,
    rho_liq= 1000,           # Density
    rho_gas =1, 
    sigma = 0.0072, #surface tension coefficient
    gravity = 10,
    Lx=3.0,            # Domain size
    Ly=3.0,
    Lz=1.0,
    tFinal=1.0,      # Simulation time
    
    # Discretization inputs
    Nx=30,           # Number of grid cells
    Ny=30,
    Nz=1,
    stepMax=50,   # Maximum number of timesteps
    max_dt = 1e-4,
    CFL=0.1,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    std_out_period = 0.0,
    out_period=1,     # Number of steps between when plots are updated
    tol = 1e-8,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = false,
    yper = false,
    zper = true,

    # pressureSolver = "GaussSeidel",
    pressureSolver = "ConjugateGradient",
    # pressureSolver = "sparseSecant",
    # pressureSolver = "NLsolve",
    pressure_scheme = "finite-difference",
    iter_type = "standard",
    # VTK_dir= "VTK_example_3"
    VTK_dir= "VTK_example_static_lid_driven"

)

"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,VF,mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_,
    imin_,imax_,jmin_,jmax_,kmin_,kmax_,
    x,y,z,xm,ym,zm,Lx,Ly,Lz = mesh
    # Pressure
    fill!(P,0.0)

    # Velocity
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 
        u[i,j,k] = 0.0 #-(ym[j] - Ly/2.0)
        v[i,j,k] = 0.0 # (xm[i] - Lx/2.0)
        w[i,j,k] = 0.0
    end

    # Volume Fraction
    fill!(VF,1.0)
    # rad=0.5
    # xo=2.5
    # yo=2.5
    # zo = 2.5
    # for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
    #     # VF[i,j,k]=VFbubble3d(x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],rad,xo,yo,zo)
    #     VF[i,j,k]=VFbubble2d(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
    # end

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
        for j=jmin_:jmax_
            if ym[j] >= 1.5 
                uleft = 1.0
            else
                uleft = 0.0
            end
            u[i,j,:] .= 2uleft .- u[imin,j,:]
        end
        v[i,:,:] = -v[imin,:,:] # No slip
        w[i,:,:] = -w[imin,:,:] # No slip
    end
    # Right
    vright=0.0
    if irankx == nprocx-1
        i = imax+1
        for j=jmin_:jmax_
            if ym[j] >= 1.5
                uright = 1.0
            else
                uright = 0.0
            end
            u[i,j,:] .= 2uright .- u[imax,j,:]
        end
        v[i,:,:] = -v[imax,:,:] .+ 2vright # No slip
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
    utop=0.0
    if iranky == nprocy-1
        j = jmax+1
        u[:,j,:] = -u[:,jmax,:] .+ 2utop # No slip
        v[:,j,:] = -v[:,jmax,:] # No flux
        w[:,j,:] = -w[:,jmax,:] # No slip
    end
    # Back 
    if irankz == 0 
        k = kmin-1
        u[:,:,k] = -u[:,:,kmin] # No slip
        v[:,:,k] = -v[:,:,kmin] # No slip
        w[:,:,k] = -w[:,:,kmin] # No flux
    end
    # Front
    if irankz == nprocz-1
        k = kmax+1
        u[:,:,k] = -u[:,:,kmax] # No slip
        v[:,:,k] = -v[:,:,kmax] # No slip
        w[:,:,k] = -w[:,:,kmax] # No flux
    end

    return nothing
end

# Simply run solver on 1 processor
@time run_solver(param, IC!, BC!)

