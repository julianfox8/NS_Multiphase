using NavierStokes_Parallel
using Test
using MPI

NS = NavierStokes_Parallel

function test_gravity_with_static_liq()
    param = parameters(
        # Constants
        mu_liq=1e-6,       # Dynamic viscosity
        mu_gas = 1e-9,
        rho_liq= 1.0,           # Density
        rho_gas =0.0001, 
        sigma = 0.0, #0.000072, #surface tension coefficient
        gravity = 1.0,
        Lx=5.0,            # Domain size
        Ly=1.0,
        Lz=1/50,
        tFinal=1.0,      # Simulation time
    
        
        # Discretization inputsc
        Nx=10,           # Number of grid cells
        Ny=10,
        Nz=1,
        stepMax=2,   # Maximum number of timesteps
        max_dt = 1e-3,
        CFL=0.1,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
        std_out_period = 0.0,
        out_period=1,     # Number of steps between when plots are updated
        tol = 1e-10,
    
        # Processors 
        nprocx = 1,
        nprocy = 1,
        nprocz = 1,
    
        # Periodicity
        xper = false,
        yper = false,
        zper = true,
    
        pressureSolver = "NLsolve",
        iter_type = "standard",
        # VTK_dir= "VTK_example_static_bubble"
    
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
            u[i,j,k] = 0.0 
            v[i,j,k] = 0.0 
            w[i,j,k] = 0.0
        end

        fill!(VF,1.0)
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
        
        vsides = 0.0
        # Left 
        if irankx == 0 
            i = imin-1
            u[i,:,:] = -u[imin,:,:] # No flux
            v[i,:,:] = -v[imin,:,:] # No slip
            w[i,:,:] = -w[imin,:,:] # No slip
        end
        # Right
        if irankx == nprocx-1
            i = imax+1
            u[i,:,:] = -u[imax,:,:] # No flux
            v[i,:,:] = -v[imax,:,:] # No slip
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

    # Setup par_env
    par_env = NS.parallel_init(param)

    mesh = NS.create_mesh(param,par_env)

    # Create work arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,Curve,sfx,sfy,sfz = NS.initArrays(mesh)


    # Create initial condition
    t = 0.0 :: Float64
    IC!(P,u,v,w,VF,mesh)
    #printArray("VF",VF,par_env)

    # Apply boundary conditions
    BC!(u,v,w,mesh,par_env)

    # Create face velocities
    NS.interpolateFace!(u,v,w,uf,vf,wf,mesh)

    # Compute band around interface
    NS.computeBand!(band,VF,param,mesh,par_env)

    # Compute interface normal 
    NS.computeNormal!(nx,ny,nz,VF,param,mesh,par_env)

    # Compute PLIC reconstruction 
    NS.computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)

    # # Check divergence
    # divg = divergence(uf,vf,wf,mesh,par_env)
    dt = NS.compute_dt(u,v,w,param,mesh,par_env)

    # Check semi-lagrangian divergence
    divg = NS.divergence(uf,vf,wf,dt,band,mesh,par_env)

    # Loop over time
    nstep = 0
    iter = 0

    while nstep<param.stepMax && t<param.tFinal

        # Update step counter
        nstep += 1

        # Compute timestep and update time
        # dt = compute_dt(u,v,w,param,mesh,par_env)
        t += dt

        # Predictor step (including VF transport)
        NS.transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,tmp1,tmp2,tmp3,tmp4,Curve,dt,param,mesh,par_env,BC!,sfx,sfy,sfz)

        # Create face velocities
        NS.interpolateFace!(us,vs,ws,uf,vf,wf,mesh)

        # # Call pressure Solver (handles processor boundaries for P)
        iter = NS.pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mesh,par_env)
        println("mean of dP @ iter ",nstep," = ", NS.mean(P[:,1,:]-P[:,end,:]))
    end
end

test_gravity_with_static_liq()