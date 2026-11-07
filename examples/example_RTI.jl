"""
Example using inflow/outflow
"""

using NavierStokes_Parallel
using Random

# Define parameters 
param = parameters(
    # Constants  (He et al. 1999 / Lee & Kim 2013, At = 0.5, Re = 1024)
    rho_liq = 3.0,          # heavy fluid (on top);  At = (3-1)/(3+1) = 0.5
    rho_gas = 1.0,          # light fluid (bottom)
    mu_liq  = 3.0/1024,     # equal kinematic viscosity: nu = W*sqrt(gW)/Re = 1/1024
    mu_gas  = 1.0/1024,
    sigma   = 0.0,          # surface tension neglected
    grav_x  = 0.0,
    grav_y  = 1.0,          # |g| = 1, pointing down (use whatever sign your code needs for -y)
    grav_z  = 0.0,
    Lx = 1.0,               # W
    Ly = 4.0,               # 4W, vertical
    Lz = 1.0,               # W
    tFinal = 4.0,           # paper figures go to t = 4 (units of sqrt(W/g))

    # Discretization
    Nx = 24,#64,                # He et al. low-res: 64x64x256; high-res: 128x128x512
    Ny = 96,#256,
    Nz = 24,
    stepMax=1500,   # Maximum number of timesteps
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
    xper = true,
    yper = false,
    zper = true,

    # pressure_scheme = "finite-difference",
    pressure_scheme = "semi-lagrangian",


    # pressureSolver = "anderson",
    # pressureSolver = "Secant",
    pressureSolver = "res_iteration",
    # pressureSolver = "hypreSecant",
    # pressureSolver = "rbgs",
    # pressureSolver = "ConjugateGradient",
    # pressureSolver = "FC_hypre",
    # pressure_scheme = "finite-difference",


    pressurePrecond = "multigrid",
    # v1 = 10,
    # v2 = 10,
    # smoother = "rbgs",
    # coarse_sol = "rbgs",


    mg_lvl = 4,

    projection_method = "Heun",
    iter_type = "standard",
    VTK_dir= "VTK_example_RTI"

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
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        VF[i,j,k]=VFrti(x[i],x[i+1],y[j],y[j+1],z[i],z[i+1])
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
Boundary conditions for face velocity
"""
function face_BC!(uf,vf,wf,mesh,par_env)
    @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
    @unpack imin,imax,jmin,jmax,kmin,kmax,x,y,z = mesh

    # Bottom 
    if iranky == 0 
        j = jmin-1
        uf[:,j,:] .= 0.0
        vf[:,j,:] .= 0.0
        wf[:,j,:] .= 0.0
    end
    # Top
    if iranky == nprocy-1
        j = jmax+1
        uf[:,j,:] .= 0.0
        vf[:,j,:] .= 0.0
        wf[:,j,:] .= 0.0
    end

    return nothing
end 

# Simply run solver on 1 processor
@time run_solver(param, IC!, BC!;faceBC = face_BC!)
