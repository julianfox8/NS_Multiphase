using NavierStokes_Parallel 
using LinearAlgebra
using DataFrames
using CSV
NS = NavierStokes_Parallel

include(joinpath(@__DIR__,"common.jl"))
include(joinpath(@__DIR__,"pre_image_tools.jl"))

function pre_image_err(dts,scheme,projection_method)
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
        Lz=1.0/100,
        tFinal=2.0,      # Simulation time
        
        # Discretization inputs
        Nx=32,           # Number of grid cells
        Ny=32,
        Nz=1,
        stepMax=1,   # Maximum number of timesteps
        max_dt = 6e-2,
        CFL=1.5,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
        std_out_period = 0.0,
        out_period=1,     # Number of steps between when plots are updated
        tol = 1e-7,

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
        # VFVelocity = "Deformation3D",
        pressure_scheme = scheme,
        # pressure_scheme = "finite-difference",
        # pressure_scheme = "semi-lagrangian",
        # pressureSolver = "hypreSecant",
        pressureSolver = "res_iteration_AA_con",

        hypreSolver = "GMRES-AMG",
        # hypreSolver = "LGMRES",
        # hypreSolver = "BiCGSTAB",
        # projection_method = "Euler",
        # projection_method = "RK4",
        # projection_method = "Midpoint",
        projection_method = projection_method,

        # Iteration method used in @loop macro
        iter_type = "standard",
        #iter_type = "floop",
        test_case = "test_error_pre_image",
    )

    """
    Initial conditions for pressure and velocity
    """
    function IC!(P,u,v,w,VF,mesh,param)
        @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
        @unpack x,y,z = mesh
        @unpack xm,ym,zm = mesh
        @unpack VFVelocity = param

        
        t=0.0

        # Velocity
        if VFVelocity == "Deformation"
            u_fun = (x,y,z,t) -> -2(sin(π*x))^2*sin(π*y)*cos(π*y)*cos(π*t/2.0)
            v_fun = (x,y,z,t) -> +2(sin(π*y))^2*sin(π*x)*cos(π*x)*cos(π*t/2.0)
            w_fun = (x,y,z,t) -> 0.0

            # Volume Fraction
            rad=0.15
            xo=0.5
            yo=0.75

            for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
                VF[i,j,k]=VFcircle(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
            end
        elseif VFVelocity == "Deformation3D"
            u_fun = (x,y,z,t) -> 2(sin(π*x))^2*sin(2π*y)*sin(2π*z)*cos(π*t/2.0)
            v_fun = (x,y,z,t) -> -(sin(π*y))^2*sin(2π*x)*sin(2π*z)*cos(π*t/2.0)
            w_fun = (x,y,z,t) -> -(sin(π*z))^2*sin(2π*x)*sin(2π*y)*cos(π*t/2.0)

            # Volume Fraction 3D
            rad=0.15
            xo=0.5
            yo=0.75
            zo=0.5

            for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
                VF[i,j,k]=VFbubble3d(x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],rad,xo,yo,zo)
            end

        end

        # Set velocities (including ghost cells)
        for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
            u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
            v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
            w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
        end
        
        return nothing    
    end

    """
    Boundary conditions for velocity
    """
    function BC!(u,v,w,t,mesh,par_env)
        # Not needed when solveNS=false
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
    mg_arrays = NS.mg_initArrays(mg_mesh,param,par_env)


    if param.pressure_scheme == "finite-difference"
        if param.tesselation == "6_tets"
            num_tets = 6
        elseif param.tesselation == "5_tets"
            num_tets = 5
        elseif param.tesselation == "24_tets"
            num_tets = 24
        end
    end

    # Create initial condition
    t = 0.0 :: Float64
    IC!(P,u,v,w,VF,mesh,param)


    # # Check divergence
    # dt = NS.compute_dt(u,v,w,param,mesh,par_env)
    @unpack dx,dy,Nx,Ny,Nz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # compute density and viscosity at intial conditions
    NS.compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

    #initialize sample points along cell face
    ns = 5
    nfaces = 6
    nfp = ns^2
    nverts = 8
    ntotal = nfaces * nfp
    sample_pts = Matrix{Float64}(undef, 3, ntotal)
    preimage_sample_pts = Matrix{Float64}(undef, 3, ntotal)
    
    # objects for barycentric coordinate computation
    tri_ids = Vector{Int}(undef, ntotal)
    lambdas = Matrix{Float64}(undef, 3, ntotal)


    pre_tri_verts = Array{eltype(tets)}(undef,3,nverts+6) ;fill!(pre_tri_verts, 0.0)
    tri_verts = Array{eltype(tets)}(undef,3,nverts+6) ;fill!(tri_verts, 0.0)

    # preallocate arrays for richardson extrapolation
    nsteps = 15
    I = [zeros(3) for _ in 1:nsteps, _ in 1:nsteps]

    # #initialize error
    # errs = zeros(length(test_dts))
    #! store copies of face velocity field 
    error_dt = zeros(length(dts))
    cfl_dt = zeros(length(dts))

    for (idx, dt) in enumerate(dts)
        # Set velocity for iteration using deformation field
        NS.defineVelocity!(t,u,v,w,uf,vf,wf,param,mesh)
        maxvel = max(maximum(abs.(u)),maximum(abs.(v)),maximum(abs.(w)))
        cfl = (dt*maxvel/dx)
        println("starting error calc for CFL = $cfl")
        uf_old = copy(uf)
        vf_old = copy(vf)
        wf_old = copy(wf)
        if param.pressure_scheme == "semi-lagrangian" #need to determine corrected field
            # Determine pressure correction
            iter = NS.pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mg_mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,gradx,grady,gradz,verts,tets,mg_arrays,BC!;)
            println("pressure solver converged in $iter iterations")
            # Corrector face velocities
            NS.corrector!(uf,vf,wf,P,dt,denx,deny,denz,mesh)
        end
        
        errors = tmp9; fill!(errors,0.0)
        # loop over domain and project vertices to test numerical integration 
        @time for k = kmin_:kmax_, j = jmin_:jmax_-1, i = imin_:imax_-1

            # Get cell vertices with triangulation ordering
            tetsign = NS.cell2verts!(verts,i,j,k,param,mesh)
            original_verts = copy(verts)
            
            tri_verts[:,1:8] = verts
            triangulate_face!(tetsign,tri_verts)
            
            # Sample face of cell with some number of points
            sample_cell_faces!(sample_pts, verts, tetsign, ns)
            
            # Compute barycentric coordinates 
            compute_barycentric!(tri_verts, sample_pts, tri_ids, lambdas, tetsign, ns)

            # Project sampled points to get analytic pre-image
            for pt_id in axes(sample_pts, 2)
                richardson_extrapolation_n!(I,@view(sample_pts[:,pt_id]), i, j, k, uf_old, vf_old, wf_old, dt, nsteps, param, mesh; tol = 1e-14)
            end
            
            # Compute numerical pre-image (either flux-corrected or pressure-corrected)
            tetsign = NS.cell2tets!(verts,tets,i,j,k,param,mesh; 
            project_verts=true,uf=uf,vf=vf,wf=wf,dt=dt,
            compute_indices=true,inds=inds,vInds=vInds,)

            pre_tri_verts[:,1:8] = verts

            if param.pressure_scheme == "finite-difference"
                # Add correction tets 
                tets,inds,ntets = NS.add_correction_tets(num_tets,verts,tets,inds,i,j,k,uf,vf,wf,dt,param,mesh;)  
            end
            
            # Triangulate pre-image along faces
            if param.pressure_scheme == "semi-lagrangian"
                triangulate_face!(tetsign,pre_tri_verts)
            elseif param.pressure_scheme == "finite-difference"
                triangulate_face_wtets!(tetsign,pre_tri_verts,tets)
            end

            # Map sample points to pre-image using barycentric coordinates and triangulation with tets
            map_sample_points_to_preimage!(preimage_sample_pts, pre_tri_verts, tri_ids, lambdas, tetsign, ns)

            # Compute error in sample points for i,j,k cell for leading edges(ignoring *max cells)
            d = preimage_sample_pts .- sample_pts
            errors[i,j,k] = sum(d[:,nfp+1:nfp*2].^2) + sum(d[:,nfp*3+1:nfp*4].^2) + sum(d[:,nfp*5+1:nfp*6].^2)
            
            # L-infinity norm if wanted
            # errors[i,j,k] = maximum(sqrt.(sum((preimage_sample_pts .- sample_pts).^2)))
        end
        # Compute L2 norm (pre-image error)
        error_dt[idx] = sqrt(sum(errors)/(ntotal*(Nx-1)*(Ny-1)*(Nz)))
        cfl_dt[idx] = cfl
    end
    return error_dt, cfl_dt
end

function sweep_preimage(; dts = [0.04, 0.03, 0.025, 0.02, 0.01, 0.0066, 0.005, 0.0025],
                          preimage_vars = [(tag = "SL", scheme = "semi-lagrangian"),(tag = "FD", scheme = "finite-difference"),],
                          projections = ["Euler", "Heun", "RK4"],
                          outfile = joinpath(DATA, "preimage_errors.csv"))

    rows = DataFrame(variant = String[], projection = String[],
                     dt = Float64[], cfl = Float64[], error = Float64[])

    for v in preimage_vars, m in projections
        @info "pre-image: $(v.tag) / $m"
        errs, cfls = pre_image_err(dts, v.scheme, m)
        append!(rows, DataFrame(variant     = v.tag,
                                projection  = m,
                                dt          = dts,
                                cfl         = cfls,
                                error       = errs))
    end

    mkpath(DATA)
    CSV.write(outfile, rows)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    sweep_preimage()
end
