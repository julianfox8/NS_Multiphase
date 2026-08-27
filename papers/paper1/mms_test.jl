"""
Method of manufactured solutions for pressure test script
"""

using NavierStokes_Parallel
using Random
using CairoMakie
using Statistics
using LaTeXStrings

using CSV
using DataFrames

NS = NavierStokes_Parallel

include(joinpath(@__DIR__, "common.jl"))

# Define parameters 
function test_psolve(Nx,Ny,scheme,solver,lvl;plt=false)
    println("Starting MMS test for pressure with Nx = $Nx and Ny = $Ny ")

    param = parameters(
        # Constants
        mu_liq=0.0,       # Dynamic viscosity of liquid (N/m)
        mu_gas = 0.0,   # Dynamic viscosity of gas (N/m)
        rho_liq= 1.0,     # Density of liquid (kg/m^3)
        rho_gas = 1.0,  # Density of gas (kg/m^3)
        sigma = 0.0,    # Surface tension coefficient (N/m)
        grav_x = 0.0,   # Gravity  (m/s^2)
        grav_y = 0.0,   # Gravity (m/s^2)
        grav_z = 0.0,   # Gravity (m/s^2)
        Lx=1.0,        # Domain size of 8Dx30Dx8D where D is bubble diameter(m)
        Ly=1.0,             
        Lz=1.0/100,
        tFinal=100.0,      # Simulation time

        
        # Discretization inputs
        Nx=Nx,           # Number of grid cells
        Ny=Ny,
        Nz=1,
        stepMax=50,   # Maximum number of timesteps
        max_dt = 2.5e-2,
        CFL=1,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
        std_out_period = 0.0,
        out_period=1,     # Number of steps between when plots are updated
        tol = 1e-5,

        # Processors 
        nprocx = 1,
        nprocy = 1,
        nprocz = 1,

        # Periodicity
        xper = false,
        yper = false,
        zper = true,

        pressure_scheme = scheme,
        pressureSolver = solver,

        hypreSolver = "GMRES-AMG",
        mg_lvl = lvl,
        # projection_method = "RK4",
        projection_method = "Heun",
        tesselation = "5_tets",
        
        iter_type = "standard",
        test_case = "psolve_test", 

    )


    """
    Compute manufactured solution and source term
    """
    function compute_MMS!(u,v,w,uf,vf,wf,RHS,VF,dt,exact,denx,deny,denz,mesh,par_env)
        @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
        @unpack jmin_,jmax_,xm,ym,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_,dy,dx,dz,x,y = mesh
        @unpack xper,yper,zper,rho_gas,rho_liq,pressure_scheme = param
        

        # this for loop is used for MMS applied strictly to RHS
        for k = kmino_:kmaxo_, j = jmino_:jmaxo_,i = imino_:imaxo_
            exact[i,j,1] = cos(2π*ym[j])*cos(2π*xm[i])
            uf[i,j,k] = -2π*sin(2π*x[i])*cos(2π*ym[j])*dt/(denx[i,j,k])
            vf[i,j,k] = -2π*sin(2π*y[j])*cos(2π*xm[i])*dt/(deny[i,j,k])
            wf[i,j,k] = 0.0
        end


        for  k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            RHS[i,j,k] = (vf[i,j+1,k]-vf[i,j,k])/dy + (uf[i+1,j,k]-uf[i,j,k])/dx
        end

        return nothing
    end

    """
    VF IC
    """
    function IC!(VF,mesh)
        @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,
                    xm,ym,y,Lx,Ly,Lz,dy = mesh

        # Volume Fraction
        rad=0.25
        xo=0.5
        yo=0.5

        for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
            VF[i,j,k]=VFcircle(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
        end

        return nothing
    end
    # Setup par_env
    par_env = NS.parallel_init(param)

    # Setup mesh
    mg_mesh = NS.init_mg_mesh(param,par_env)

    # Initialize work arrays for finest level along with subset of arrays for coarser levels
    mesh = mg_mesh.mesh_lvls[1]

    # Initialize arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,RHS,tmp2,exact_sol,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,divg,mask,tets,verts,inds,vInds = NS.initArrays(mesh)

    @unpack x,y,z,dx,dy,dz,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    p_min,p_max = NS.prepare_indices(tmp5,par_env,mesh)
    mg_arrays = NS.mg_initArrays(mg_mesh,param,par_env)

    # Compute dt
    dt = NS.compute_dt(u,v,w,param,mesh,par_env)
    t = 0.0 :: Float64

    # Create source term/exact solution and apply BC to Pressure
    IC!(VF,mesh) 
    # fill!(VF,0.0)
    # NS.update_borders!(VF,mesh,par_env)

    NS.compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)
    
    compute_MMS!(u,v,w,uf,vf,wf,RHS,VF,dt,exact_sol,denx,deny,denz,mesh,par_env)

    # Compute band around interface
    # NS.computeBand!(band,VF,param,mesh,par_env)
    fill!(band,0.0)

    # Loop over time
    nstep = 0
    iter = 0

    # # Call pressure Solver (handles processor boundaries for P)
    if param.mg_lvl > 1
        iter = NS.mg_cycler(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,mg_arrays,mg_mesh,VF,verts,tets,param,par_env) 
    elseif param.pressure_scheme == "finite-difference"
        if param.pressureSolver == "FC_hypre"
            iter = NS.FC_hypre_solver(P,RHS,tmp2,denx,deny,denz,tmp5,mg_arrays[1].jacob,mg_arrays[1].x_vec,mg_arrays[1].b_vec,dt,param,mesh,par_env,20000)
        elseif param.pressureSolver == "gauss-seidel"
            iter = NS.gs(P,RHS,tmp2,denx,deny,denz,dt,param,mg_mesh.mesh_lvls[1],par_env;max_iter=100000)
        elseif param.pressureSolver == "jacobi"
            iter = NS.jacobi(P,tmp6,RHS,tmp2,denx,deny,denz,dt,param,mg_mesh.mesh_lvls[1],par_env;max_iter=100000)
        elseif param.pressureSolver == "geometric_mg"
            iter = NS.mg_geometric!(P, RHS, denx, deny, denz, dt, param, mg_mesh.mesh_lvls[1], par_env)
        elseif param.pressureSolver == "congugateGradient"
            iter = NS.cg!(P, RHS, denx, deny, denz,tmp6,dt, param, mg_mesh.mesh_lvls[1], par_env)  
        end
    elseif param.pressure_scheme == "semi-lagrangian"
        if param.pressureSolver == "res_iteration_AA_con"
            iter = NS.res_iteration_AA_con(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp6,tmp2,tmp7,tmp4,verts,tets,param,mg_mesh.mesh_lvls[1],par_env;) 
        elseif param.pressureSolver == "res_iteration_AA_uncon"
            iter = NS.res_iteration_AA_uncon(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp6,tmp2,tmp7,tmp4,verts,tets,param,mg_mesh.mesh_lvls[1],par_env;) 
        elseif param.pressureSolver == "hypreSecant"
            iter = NS.Secant_jacobian_hypre!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp6,tmp2,tmp7,tmp4,verts,tets,mg_arrays[1].jacob,mg_arrays[1].x_vec,mg_arrays[1].b_vec,param,mg_mesh.mesh_lvls[1],par_env)
        end
    end
    println("solver: $(param.pressureSolver) converged in $iter iterations")
    # Check solution error
    L2_error = sqrt(sum((exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]).^2))/sqrt(sum(exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_].^2))

    Linf_error = maximum(abs.(exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]))/maximum(abs.(exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_]))
      # Plotting routine
    P_slice = P[imin_:imax_,jmin_:jmax_,1]
    exact_slice = exact_sol[imin_:imax_,jmin_:jmax_,1]
    VF_slice = VF[imin_:imax_,jmin_:jmax_,1]
    RHS_slice = RHS[imin_:imax_,jmin_:jmax_,1]
    error_slice = (exact_slice - P_slice)
    

    # Create x and y coordinates for plotting
    x_plot = x[imin_:imax_]
    y_plot = y[jmin_:jmax_]

    # Pass plt = true to visualize a single MMS run
    if plt 
        denx_cell = similar(P_slice)
        deny_cell = similar(P_slice)

        for j ∈ jmin_:jmax_, i ∈ imin_:imax_
            denx_cell[i,j] = (denx[i,j,1] + denx[i+1,j,1])/2
            deny_cell[i,j] = (deny[i,j,1] + deny[i,j+1,1])/2
        end
        
        # Create subplots
        fig = Figure(size = (1000, 800))

        # -----------------------------
        # Heatmap 1: Computed P
        # -----------------------------
        ax1 = Axis(fig[1, 1],
            xlabel = "x",
            ylabel = "y",
            title = "Computed P",
            aspect = DataAspect()
        )

        hm1 = heatmap!(ax1, x_plot, y_plot, P_slice)
        Colorbar(fig[1, 1, Right()], hm1)

        # -----------------------------
        # Heatmap 2: Exact Solution
        # -----------------------------
        ax2 = Axis(fig[1, 2],
            xlabel = "x",
            ylabel = "y",
            title = "Exact Solution",
            aspect = DataAspect()
        )

        hm2 = heatmap!(ax2, x_plot, y_plot, exact_slice)
        Colorbar(fig[1, 2, Right()], hm2)

        # -----------------------------
        # Heatmap 3: Source / Error
        # -----------------------------
        ax3 = Axis(fig[2, 1],
            xlabel = "x",
            ylabel = "y",
            title = "Error",
            aspect = DataAspect()
        )

        hm3 = heatmap!(ax3, x_plot, y_plot, error_slice, colormap = :hot)
        Colorbar(fig[2, 1, Right()], hm3)

        # -----------------------------
        # Line plot: Centerline
        # -----------------------------
        ax4 = Axis(fig[2, 2],
            xlabel = "y",
            ylabel = "P",
            title = "Centerline (x=0.5)"
        )

        mid = div(Nx, 2)

        lines!(ax4, x_plot, P_slice[mid, :],
            label = "Computed P",
            linewidth = 2
        )

        lines!(ax4, x_plot, exact_slice[mid, :],
            label = "Exact",
            linewidth = 2,
            linestyle = :dash
        )



        axislegend(ax4, position = :rt)


        ax5 = Axis(fig[3, 1],
            xlabel = "x",
            ylabel = "y",
            title = "denx",
            aspect = DataAspect()
        )

        hm5 = heatmap!(ax5, x_plot, y_plot, denx_cell)
        Colorbar(fig[3, 1, Right()], hm5)


        ax6 = Axis(fig[3, 2],
            xlabel = "x",
            ylabel = "y",
            title = "VF",
            aspect = DataAspect()
        )

        hm6 = heatmap!(ax6, x_plot, y_plot, VF_slice)
        Colorbar(fig[3, 2, Right()], hm6)
        # -----------------------------
        # Global title
        # -----------------------------
        Label(fig[0, :],
            "Mesh: $(Nx)x$(Ny), L∞ error: $(round(Linf_error, sigdigits=4)), interpolation: $(param.interpolation_method)",
            fontsize = 18
        )

        display(fig)
        # savefig("MMS_comparison_$(Nx)x$(Ny).png")
        # println("Saved plot: MMS_comparison_$(Nx)x$(Ny).png")
    end
    return L2_error,Linf_error
end

# ---------------------------------------------------------------------------
# Sweep: mesh refinement x discretisation variant -> data/mms_errors.csv
#
# Tags match the other paper CSVs ("SL", "FD"), so make_plots.jl can reuse the
# same legend labels. Plotting lives in make_plots.jl, not here.
# ---------------------------------------------------------------------------

const MMS_VARIANTS = [
    (tag = "FD", scheme = "finite-difference", solver = "FC_hypre",             lvl = 1),
    (tag = "SL", scheme = "semi-lagrangian",   solver = "res_iteration_AA_con", lvl = 1),
]

"""
    sweep_mms(; Ns, variants, outfile)

Run the variable-density pressure MMS over the mesh x variant matrix and write
`variant, N, L2, Linf, time` to `outfile`, replacing it. Point a narrowed sweep
at a scratch `outfile` — it would otherwise truncate the full-matrix CSV.
"""
function sweep_mms(; Ns       = [16, 32, 64, 128],
                     variants = MMS_VARIANTS,
                     outfile  = joinpath(DATA, "mms_errors.csv"))

    rows = DataFrame(variant = String[], N = Int[],
                     L2 = Float64[], Linf = Float64[], time = Float64[])

    for v in variants, n in Ns
        @info "MMS: $(v.tag) N=$n"
        t0 = time()
        L2, Linf = test_psolve(n, n, v.scheme, v.solver, v.lvl)
        push!(rows, (v.tag, n, L2, Linf, time() - t0))
    end

    mkpath(DATA)
    CSV.write(outfile, rows)
    @info "wrote $outfile"
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    sweep_mms()
end
