using NavierStokes_Parallel


NS = NavierStokes_Parallel


param = parameters(
    # Constants
    mu=0.1,       # Dynamic viscosity
    rho=1.0,           # Density
    Lx=3.0,            # Domain size
    Ly=3.0,
    Lz=3.0,
    tFinal=100.0,      # Simulation time

    # Discretization inputs
    Nx=20,           # Number of grid cells
    Ny=10,
    Nz=1,
    stepMax=200,   # Maximum number of timesteps
    CFL=0.4,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
    std_out_period = 0.0,
    out_period=10,     # Number of steps between when plots are updated
    tol = 1e-3,

    # Processors 
    nprocx = 1,
    nprocy = 1,
    nprocz = 1,

    # Periodicity
    xper = true,
    yper = true,
    zper = true,

    # pressureSolver = "Secant",
    pressureSolver = "NLsolve",
    # pressureSolver = "GaussSeidel",

    # Iteration method used in @loop macro
    iter_type = "standard",
    # iter_type = "floop",
)
# Setup par_env
par_env = NS.parallel_init(param)

# Setup mesh
mesh = NS.create_mesh(param,par_env)


P,u,v,w,VF,uf,vf,wf = NS.initArrays(mesh)
    
"""
Initial conditions for pressure and velocity
"""
function IC!(P,u,v,w,VF,mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,
                xm,ym,zm,Lx,Ly,Lz = mesh
    # Pressure
    
    fill!(P,0.0)

    # Velocity
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_ 
        u[i,j,k] = 0.0 #-(ym[j] - Ly/2.0)
        v[i,j,k] = 0.0 # (xm[i] - Lx/2.0)
        w[i,j,k] = 0.0
    end

    # Volume Fraction
    fill!(VF,0.0)

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
            if ym[j] <= 1.5 
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
            if ym[j] <= 1.5
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

# # Create initial condition
t = 0.0 :: Float64
IC!(P,u,v,w,VF,mesh)
#printArray("VF",VF,par_env)

# Apply boundary conditions
BC!(u,v,w,mesh,par_env)

# update_borders!(u,mesh,par_env)
# update_borders!(v,mesh,par_env)
# update_borders!(w,mesh,par_env)
# update_borders!(P,mesh,par_env)

# Create face velocities
NS.interpolateFace!(u,v,w,uf,vf,wf,mesh)

# # Check divergence
# divg = divergence(uf,vf,wf,mesh,par_env)

# # Loop over time
# nstep = 0
# iter = 0
# while nstep<stepMax && t<tFinal

# Update step counter
#nstep += 1
# Compute timestep and update time
dt = NS.compute_dt(u,v,w,param,mesh,par_env)
t += dt
# println(P)
# Call pressure Solver (handles processor boundaries for P)
iter = NS.semi_lag_pressure_solver!(P,uf,vf,wf,dt,param,mesh,par_env)
println(iter)
# iter = NS.pressure_solver!(P,uf,vf,wf,dt,param,mesh,par_env)
#println(iter)
# end



