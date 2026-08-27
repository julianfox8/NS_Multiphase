using NavierStokes_Parallel
using CSV
using DataFrames  


NS = NavierStokes_Parallel

include(joinpath(@__DIR__, "common.jl"))

function test_advection(n,scheme,fc,case,tag;tFinal = 1.0,pre_image_vis = false)
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
        tFinal=tFinal,   # Simulation time
        
        # Discretization inputs
        Nx=n,           # Number of grid cells
        Ny=n,
        Nz=1,
        stepMax=10000,   # Maximum number of timesteps
        max_dt =1e-1,
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

        # Set test case
        # `case` selects the physics (must match Tools.jl); `tag` distinguishes
        # the variant in the output directory name
        test_case = "$(case)_$(tag)",

        # Turn off NS solver
        solveNS = false,
        VFVelocity = case,

        pressure_scheme = scheme,
        
        # pressureSolver = "hypreSecant",
        pressureSolver = "res_iteration_AA_con",

        # projection_method = "Euler",
        # projection_method = "Midpoint",
        projection_method = "Heun",
        # projection_method = "RK4",

        flux_corrections = fc,
        hypreSolver = "LGMRES",
        # hypreSolver = "GMRES-AMG",
        # hypreSolver = "BiCGSTAB",
        mg_lvl = 1,
        # Iteration method used in @loop macro
        iter_type = "standard",
        #iter_type = "floop",

        # Output name for VTK and CSV
        VTK_root = RESULTS,
    )

    """
    Initial conditions for pressure and velocity
    """
    function IC!(P,u,v,w,VF,mesh,param)
        @unpack x,y,z,xm,ym,zm,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
        @unpack VFVelocity = param

        t=0.0
        # Velocity
        if VFVelocity == "Deformation"
            u_fun = (x,y,z,t) -> -2(sin(π*x))^2*sin(π*y)*cos(π*y)*cos(π*t/2.0)
            v_fun = (x,y,z,t) -> +2(sin(π*y))^2*sin(π*x)*cos(π*x)*cos(π*t/2.0)
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

            for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
                VF[i,j,k]=VFcircle(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
            end
        elseif VFVelocity == "Deformation3D"
            u_fun = (x,y,z,t) -> 2(sin(π*x))^2*sin(2π*y)*sin(2π*z)*cos(π*t/2.0)
            v_fun = (x,y,z,t) -> -(sin(π*y))^2*sin(2π*x)*sin(2π*z)*cos(π*t/2.0)
            w_fun = (x,y,z,t) -> -(sin(π*z))^2*sin(2π*x)*sin(2π*y)*cos(π*t/2.0)

            # Set velocities (including ghost cells)
            for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
                u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
                v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
                w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
            end
            
            # Volume Fraction 3D
            rad=0.15
            xo=0.5
            yo=0.75
            zo=0.5

            for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
                VF[i,j,k]=VFbubble3d(x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],rad,xo,yo,zo)
            end

        elseif VFVelocity == "Zalesak"        
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
        else 
            error("No test case for this flow field")
        end

        return nothing    
    end

    """
    Boundary conditions for velocity
    """
    function BC!(u,v,w,mesh,par_env)
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

    # Create initial condition
    t = 0.0 :: Float64
    IC!(P,u,v,w,VF,mesh,param)

    # Set velocity for iteration using deformation field
    NS.defineVelocity!(t,u,v,w,uf,vf,wf,param,mesh)

    # Compute band around interface
    # NS.computeBand!(band,VF,param,mesh,par_env)
    fill!(band,0.0)
    
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

    # Initialize struct to store vertices of pre-images based on domain size
    if pre_image_vis
        pmesh = NS.PreimageMesh(Vector{Vector{Float64}}(),
                Vector{NTuple{4,Int}}(),
                Int[],
                Vector{Vector{Int}}(),
                Int[],
                Vector{Vector{Vector{Float64}}}(),
                Int[])
        tpmesh = NS.PreimageMesh(Vector{Vector{Float64}}(),
                Vector{NTuple{4,Int}}(),
                Int[],
                Vector{Vector{Int}}(),
                Int[],
                Vector{Vector{Vector{Float64}}}(),
                Int[])
    end
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
            if !pre_image_vis
                iter = NS.pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mg_mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,gradx,grady,gradz,verts,tets,mg_arrays,BC!)
            else
                iter = NS.pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mg_mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,gradx,grady,gradz,verts,tets,mg_arrays,BC!;pmesh=pmesh)
                NS.pmesh2VTK(pmesh,"pressure_preimage",param)
            end
            # Corrector face velocities
            NS.corrector!(uf,vf,wf,P,dt,denx,deny,denz,mesh)
        end
        
        # Calculate divergence
        NS.divergence!(divg,uf,vf,wf,dt,band,verts,tets,param,mesh,par_env)

        # output before transport with divergence free velocity field    
        NS.std_out(h_last,t_last,nstep,t,P,VF,u,v,w,divg,VF_init,iter,param,mesh,par_env)

        # Predictor step (including VF transport)
        if !pre_image_vis
            NS.transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,mask,dt,param,mesh,par_env,BC!,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,t,verts,tets,inds,vInds)
        else
            NS.transport!(us,vs,ws,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,mask,dt,param,mesh,par_env,BC!,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,t,verts,tets,inds,vInds;pmesh=tpmesh)
            NS.pmesh2VTK(tpmesh,"transport_preimage",param)
            error("pre-images vtk files created")
        end

        # VTK Output
        NS.VTK(nstep,t,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp1,param,mesh,par_env,pvd,pvd_restart,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz,verts,tets)
    end
end

struct Variant
    tag::String; scheme::String; fc::Bool
end

const VARIANTS = [
    Variant("SL",   "semi-lagrangian",   false),   # fc ignored when scheme is SL
    Variant("FD",   "finite-difference", true),
    Variant("noFC", "finite-difference", false),
]

const CASES = Dict(
    "deformation" => (vf = "Deformation", tFinal = 2.0, Ns = [48,64,96,128]),
    "zalesak"     => (vf = "Zalesak",     tFinal = 1.0, Ns = [48,64,96,128]),
)

# ---------------------------------------------------------------------------
# Variant / job selection
# ---------------------------------------------------------------------------

"Path to the `Solver.pvd` a `(variant, N)` run writes."
pvd_path(spec, v::Variant, n::Integer) =
    joinpath(RESULTS,
             "VTK_$(spec.vf)_$(v.tag)_$(v.scheme)_$(n)_$(n)_$(1)",
             "Solver.pvd")

# ---------------------------------------------------------------------------
# Stage 1: Run cases to generate VTK results
# ---------------------------------------------------------------------------

"""
    run_cases(case; Ns = CASES[case].Ns, force = false)

Run every variant of `case` at each mesh size in `Ns`.

Jobs whose `Solver.pvd` already exists are skipped, so an interrupted sweep
resumes; to redo a subset, delete its result directories and call again.
`force = true` reruns regardless — needed after any solver change, since the
skip only checks that a file exists, not what produced it.
"""
function run_cases(case; Ns = CASES[case].Ns, force = false)
    spec = CASES[case]

    ran = skipped = 0
    for v in VARIANTS, n in Ns
        if !force && isfile(pvd_path(spec, v, n))
            @info "skipping $(spec.vf) $(v.tag) N=$n — results exist"
            skipped += 1
            continue
        end
        @info "running $(spec.vf) $(v.tag) N=$n"
        test_advection(n, v.scheme, v.fc, spec.vf, v.tag; tFinal = spec.tFinal)
        ran += 1
    end

    @info "$case: ran $ran, skipped $skipped"
    return nothing
end

# ---------------------------------------------------------------------------
# Stage 2: Produce all csv files for error analysis
# ---------------------------------------------------------------------------
function compute_shape_err(file_path, N, t_len)

    VF_init = zeros(Float32, N, N, 1)
    VF_t    = similar(VF_init)

    NS.fillArray!(VF_t, t_len, file_path)
    NS.fillArray!(VF_init, 0, file_path)

    return sum(abs.(VF_t .- VF_init)) / (N*N) 
end

"""
    compute_errors(case, Ns = CASES[case].Ns)

Rebuild `data/<case>_errors.csv` from the VTK results on disk, warning past any
run that is missing. Derived data — rerun it any time to repair the file.
"""
function compute_errors(case, Ns = CASES[case].Ns)
    spec = CASES[case]
    rows = DataFrame(variant = String[], N = Int[], E_shape = Float64[])

    for v in VARIANTS, n in Ns
        pvd = pvd_path(spec, v, n)
        isfile(pvd) || (@warn "missing $(v.tag) N=$n, skipping" pvd; continue)
        push!(rows, (v.tag, n, compute_shape_err(pvd, n, spec.tFinal)))
    end

    mkpath(DATA)
    CSV.write(joinpath(DATA, "$(case)_errors.csv"), rows)
    return rows
end


if abspath(PROGRAM_FILE) == @__FILE__
    run_cases(ARGS[1])
    compute_errors(ARGS[1])
end
