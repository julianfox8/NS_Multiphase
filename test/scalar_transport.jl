using NavierStokes_Parallel

NS = NavierStokes_Parallel

function scalar_advection!(uf,vf,wf,VF,nx,ny,nz,D,VFnew,dt,param,mesh,par_env,t,verts,tets,inds,vInds;pmesh=nothing)
    @unpack instability,grav_x,grav_y,grav_z,pressure_scheme,tesselation,VFlo,VFhi,rho_liq,rho_gas,nband = param
    @unpack irankx,isroot = par_env
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    
    if tesselation == "6_tets"
        num_tets = 6
    elseif tesselation == "5_tets"
        num_tets = 5
    elseif tesselation == "24_tets"
        num_tets = 24
    else
        error("Unknown Tesselation: $tesselation")
    end
    # Compute interface normal 
    fill!(nx,0.0)
    fill!(ny,0.0)
    fill!(nz,0.0)
    
    # Preallocate for cutTet
    nLevel=100
    nThread = Threads.nthreads()
    vert = Array{Float64}(undef, 3, 8,nLevel,nThread)
    vert_ind = Array{Int64}(undef, 3, 8, 2,nLevel,nThread)
    d = Array{Float64}(undef, 4,nThread)
    newtet = Array{Float64}(undef, 3, 4,nThread)

    # Perform transport
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
                # Calculate inertia near or away from the interface
        # if abs(band[i,j,k]) <= nband
        # Calculate inertia near or away from the interface        
        ntets = num_tets
        tetsign = NS.cell2tets!(verts,tets,i,j,k,param,mesh; 
            project_verts=true,uf=uf,vf=vf,wf=wf,dt=dt,
            compute_indices=true,inds=inds,vInds=vInds,pmesh=pmesh)

        if pressure_scheme == "finite-difference"
            # Add correction tets 
            tets,inds,ntets = NS.add_correction_tets(ntets,verts,tets,inds,i,j,k,uf,vf,wf,dt,param,mesh;pmesh=pmesh)
        end
        
        # Compute VF in semi-Lagrangian cell 
        vol  = 0.0
        vScal = 0.0
        
        for tet in eachindex(view(tets,1,1,1:ntets))
            tetVol, tetvScal, maxlvl = NS.cutTet_scal(tets[:,:,tet],inds[:,:,tet],
                                VF,
                                false,false,false,nx,ny,nz,D,mesh,param,
                                1,vert,vert_ind,d,newtet)
            vol  += tetsign * tetVol
            vScal += tetsign * tetvScal

        end
        VFnew[i,j,k] = vScal/vol

    end
    
    # Finish updating VF
    VF[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .= VFnew[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
    NS.Neumann!(VF,mesh,par_env)
    NS.update_VF_borders!(VF,mesh,par_env)
    # end

     return nothing
end

function VFsine(xc,yc,thickness;A=0.1,y0=0.5,k=1.0)

    ys = y0 + A*sin(2π*k*xc)

    if abs(yc - ys) < thickness
        return 1.0
    else
        return 0.0
    end
end

function test_scalar_transport()
    # Define parameters 
    param = parameters(
        # Constants
        mu_liq=1.0,            # Dynamic viscosity
        mu_gas = 1.0,
        rho_liq=1.0,           # Density
        rho_gas = 1.0,
        sigma = 0.0, # surface tension coefficient (N/m)
        grav_x = 0.0,
        grav_y = 0.0,
        grav_z = 0.0, # Gravity (m/s^2)
        Lx=1.0,            # Domain size
        Ly=1.0,
        Lz=1/50,
        tFinal=2.0,      # Simulation time
        
        # Discretization inputs
        Nx=64,           # Number of grid cells
        Ny=64,
        Nz=1,
        stepMax=10000,   # Maximum number of timesteps
        max_dt = 5e-2,
        CFL=0.5,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
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

        # Turn off NS solver
        solveNS = false,
        VFVelocity = "Deformation",

        # pressure_scheme = "finite-difference",
        pressure_scheme = "semi-lagrangian",
        # pressureSolver = "hypreSecant",
        pressureSolver = "res_iteration",

        hypreSolver = "GMRES-AMG",
        # hypreSolver = "BiCGSTAB",
        projection_method = "Euler",
        # projection_method = "RK4",
        # projection_method = "Midpoint",
        # projection_method = "Trapezoidal",

        # interpolation_method = "inv_dist_weighting",
        interpolation_method = "trilinear",
        # interpolation_method = "taylor",
        # interpolation_method = "taylorls",
        # interpolation_method = "mls7",

        # Iteration method used in @loop macro
        iter_type = "standard",
        #iter_type = "floop",
        # test_case = "Deformation_result_euler",
        test_case = "Deformation_transport_test",
    )

    """
    Initial conditions for pressure and velocity
    """
    function IC!(P,u,v,w,VF,mesh)
        @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
        @unpack x,y,z,dy = mesh
        @unpack xm,ym,zm = mesh

        # Velocity
        t=0.0
        u_fun(x,y,z,t) = -2(sin(π*x))^2*sin(π*y)*cos(π*y)*cos(π*t/2.0)
        v_fun(x,y,z,t) = +2(sin(π*y))^2*sin(π*x)*cos(π*x)*cos(π*t/2.0)
        w_fun(x,y,z,t) = 0.0
        # Set velocities (including ghost cells)
        for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
            u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
            v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
            w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
        end

        # Volume Fraction
        for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
            VF[i,j,k]=VFsine(xm[i],ym[j],dy/2)
        end

        return nothing    
    end

    """
    Boundary conditions for velocity
    """
    function BC!(u,v,w,mesh,par_env)
        # Not needed for deformation test
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
    NS.Neumann!(VF,mesh,par_env)
    NS.update_VF_borders!(VF,mesh,par_env)
    # Define velocity for deformation test
    NS.defineVelocity!(t,u,v,w,uf,vf,wf,param,mesh)

    # Compute band around interface
    # NS.computeBand!(band,VF,param,mesh,par_env)
    fill!(band,0.0)

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

        # Compute timestep and update time
        CFL_dt = param.CFL*max(dx/maximum(abs.(u)),dy/maximum(abs.(v)))
        if (param.tFinal-t) < param.max_dt && (param.tFinal-t) < CFL_dt
            dt = param.tFinal-t
        else
            dt = NS.compute_dt(u,v,w,param,mesh,par_env)
        end
        
        # Set velocity for iteration using deformation field
        NS.defineVelocity!(t+dt/2,u,v,w,uf,vf,wf,param,mesh)
        
        # Update time 
        t += dt

        if param.pressure_scheme == "semi-lagrangian"

            # Determine pressure correction
            iter = NS.pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mg_mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,gradx,grady,gradz,verts,tets,mg_arrays,BC!)#;pmesh=pmesh)
        
            # Corrector face velocities
            NS.corrector!(uf,vf,wf,P,dt,denx,deny,denz,mesh)
            # NS.pmesh2VTK(pmesh,"pressure_preimage",param)

        end

        # Calculate divergence
        NS.divergence!(divg,uf,vf,wf,dt,band,verts,tets,param,mesh,par_env)
        
        # output before transport with divergence free velocity field    
        NS.std_out(h_last,t_last,nstep,t,P,VF,u,v,w,divg,VF_init,iter,param,mesh,par_env)
        
        #Compute scalar advection
        scalar_advection!(uf,vf,wf,VF,nx,ny,nz,D,tmp1,dt,param,mesh,par_env,t,verts,tets,inds,vInds)
        
        # VTK Output
        NS.VTK(nstep,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_restart,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz,verts,tets)
    end
end

test_scalar_transport()