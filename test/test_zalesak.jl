using NavierStokes_Parallel
using Test
using MPI

NS = NavierStokes_Parallel

function test_pressure()
    # Define parameters 
    param = parameters(
        # Constants
        mu_liq=0.0,            # Dynamic viscosity
        mu_gas = 0.0,
        rho_liq=1.0,           # Density
        rho_gas = 1.0,
        sigma = 0.0, # surface tension coefficient (N/m)
        grav_x = 0.0,
        grav_y = 0.0,
        grav_z = 0.0, # Gravity (m/s^2)
        Lx=1.0,            # Domain size
        Ly=1.0,
        Lz=1/50,
        tFinal=1.0,      # Simulation time
        
        # Discretization inputs
        Nx=100,           # Number of grid cells
        Ny=100,
        Nz=1,
        stepMax=10000,   # Maximum number of timesteps
        max_dt = 1e-2,
        CFL=0.5,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
        std_out_period = 0.0,
        out_period=1,     # Number of steps between when plots are updated
        tol = 1e-11,

        # Processors 
        nprocx = 1,
        nprocy = 1,
        nprocz = 1,

        # Periodicity
        xper = false,
        yper = false,
        zper = true,

        # Turn off NS solver
        solveNS = false,
        VFVelocity = "rotation",

        # pressure_scheme = "finite-difference",
        pressure_scheme = "semi-lagrangian",
        # pressureSolver = "hypreSecant",
        pressureSolver = "res_iteration",

        # hypreSolver = "GMRES-AMG",
        hypreSolver = "BiCGSTAB",

        # Iteration method used in @loop macro
        iter_type = "standard",
        #iter_type = "floop",
        test_case = "Zalesak_psolve",
    )

    """
    Initial conditions for pressure and velocity
    """
    function IC!(P,u,v,w,VF,mesh)
        @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
        @unpack x,y,z = mesh
        @unpack xm,ym,zm = mesh

        # Velocity
        t=0.0
        u_fun = (x,y,z,t) -> 2π*(0.5 - y)
        v_fun = (x,y,z,t) -> 2π*(x - 0.5)
        w_fun = (x,y,z,t) -> 0.0
        # Set velocities (including ghost cells)
        for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
            u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
            v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
            w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
        end

        # Volume Fraction
        rad=0.15
        xo=0.5
        yo=0.75
        slot_width = 0.05
        slot_length = 0.24
        for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
            VF[i,j,k]=VFzalesak2d(x[i],x[i+1],y[j],y[j+1],rad,xo,yo,slot_width,slot_length)
            
        end

        return nothing    
    end


    # Setup par_env
    par_env = NS.parallel_init(param)

    # Setup mesh
    mg_mesh = NS.init_mg_mesh(param,par_env)
    mesh = mg_mesh.mesh_lvls[1]
    # Initialize arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,divg,mask,tets,verts,inds,vInds = NS.initArrays(mesh)

    @unpack x,y,z,dx,dy,dz,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    p_min,p_max = NS.prepare_indices(tmp5,par_env,mesh)
    mg_arrays = NS.mg_initArrays(mg_mesh,param,p_min,p_max,par_env)

    # Create initial condition
    t = 0.0 :: Float64
    IC!(P,u,v,w,VF,mesh)

    # Compute band around interface
    NS.computeBand!(band,VF,param,mesh,par_env)
    
    # Compute interface normal 
    NS.computeNormal!(nx,ny,nz,VF,param,mesh,par_env)

    # Compute PLIC reconstruction 
    NS.computePLIC!(D,nx,ny,nz,VF,param,mesh,par_env)

    # # Check divergence
    dt = NS.compute_dt(u,v,w,param,mesh,par_env)

    # Check semi-lagrangian divergence
    NS.divergence!(divg,uf,vf,wf,dt,band,verts,tets,param,mesh,par_env)

    # compute density and viscosity at intial conditions
    NS.compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

    # Loop over time
    nstep = 0
    iter = 0

    # Grab initial volume fraction sum
    VF_init = NS.parallel_sum(VF[mesh.imin_:mesh.imax_,mesh.jmin_:mesh.jmax_,mesh.kmin_:mesh.kmax_]*dx*dy*dz,par_env)

    # Output IC
    t_last =[-100.0,]
    h_last =[100]

    # Initialize VTK
    pvd,pvd_restart,pvd_PLIC = NS.VTK_init(param,par_env)
    NS.VTK(nstep,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_restart,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz,verts,tets)
    NS.std_out(h_last,t_last,nstep,t,P,VF,u,v,w,divg,VF_init,iter,param,mesh,par_env)
    while nstep<param.stepMax && t<param.tFinal

        # Update step counter
        nstep += 1

        # Set velocity for iteration using deformation field
        NS.defineVelocity!(t,u,v,w,uf,vf,wf,param,mesh)

        # Compute timestep and update time
        dt = NS.compute_dt(u,v,w,param,mesh,par_env)
        t += dt

        if param.pressure_scheme == "semi-lagrangian"

            # Determine pressure correction
            iter = NS.poisson_solve!(P,tmp9,uf,vf,wf,gradx,grady,gradz,band,dt,param,mg_mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,verts,tets,mg_arrays)
            
            # Corrector face velocities
            NS.corrector!(uf,vf,wf,P,dt,denx,deny,denz,mesh)

        end

        # Calculate divergence
        NS.divergence!(divg,uf,vf,wf,dt,band,verts,tets,param,mesh,par_env)

        # output before transport with divergence free velocity field    
        NS.std_out(h_last,t_last,nstep,t,P,VF,u,v,w,divg,VF_init,iter,param,mesh,par_env)

        # Predictor step (including VF transport)
        NS.transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,mask,dt,param,mesh,par_env,BC!,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,t,verts,tets,inds,vInds)
        
        # Update bands with transported VF
        NS.computeBand!(band,VF,param,mesh,par_env)
        
        # Update density and viscosity with transported VF
        NS.compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

        # VTK Output
        NS.VTK(nstep,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_restart,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz,verts,tets)
    
    end
end

test_pressure()