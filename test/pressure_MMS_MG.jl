"""
Method of manufactured solutions for pressure test script
"""

using NavierStokes_Parallel
using Random
using CairoMakie
using Statistics
using LinearAlgebra

NS = NavierStokes_Parallel



# Define parameters 
function test_psolve(Nx,Ny,scheme,solver,lvl,n_cycles)
    println("Starting MMS test for pressure with Nx = $Nx and Ny = $Ny for scheme $scheme")
    ##! still need to determine the best way to upate Nx,Ny,Nz

    param = parameters(
        # Constants
        mu_liq=0.0,       # Dynamic viscosity of liquid (N/m)
        mu_gas = 0.0,   # Dynamic viscosity of gas (N/m)
        rho_liq= 1000.0,     # Density of liquid (kg/m^3)
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
        max_dt = 1e-4,
        CFL=0.4,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
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
    function compute_MMS!(uf,vf,wf,RHS,dt,exact,denx,deny,denz,mesh,par_env)
        @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
        @unpack jmin_,jmax_,xm,ym,imin_,imax_,jmin_,jmax_,kmin_,kmax_,dy,dx,dz = mesh
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

        # for k = kmin_:kmax_,i = imin_:imax_
        #     #height
        #     height = Ly/2 

        #     for j = jmin_:jmax_        
        #         if y[j] < height && y[j+1] > height 
        #             VF[i,j,k] = (height - y[j])/(y[j+1]-y[j]) 
        #         elseif y[j] < height
        #             VF[i,j,k] = 1.0
        #         else
        #             VF[i,j,k] = 0.0
        #         end
        #     end
        # end



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
    #! need to apply BC to VF (potentially)
    # fill!(VF,0.0)
    IC!(VF,mesh)
    # NS.update_borders!(VF,mesh,par_env)

    # compute density and viscosity at intial conditions
    NS.compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

    # compute_MMS!(uf,vf,wf,RHS,VF,dt,exact_sol,mesh,par_env)
    compute_MMS!(uf,vf,wf,RHS,dt,exact_sol,denx,deny,denz,mesh,par_env)
    
    # Compute band around interface
    NS.computeBand!(band,VF,param,mesh,par_env)
    # fill!(band,1.0)

    # Loop over time
    nstep = 0
    iter = 0

    # Copy all level-1 fields into mg_arrays once (densities, velocities, band stay fixed)
    fields = (P = P, uf = uf, vf = vf, wf = wf,
              denx = denx, deny = deny, denz = denz,
              gradx = gradx, grady = grady, gradz = gradz,
              band = band)
    NS.copy_to_mg!(mg_arrays, fields, 1)

    # # Call pressure Solver (handles processor boundaries for P)
    # Initial error norm
    e0 = norm(exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-mg_arrays[1].P_h[imin_:imax_, jmin_:jmax_, kmin_:kmax_])


    # Run V-cycles and record per-cycle convergence factor
    err_norms = zeros(n_cycles + 1)
    err_norms[1] = e0
    n_done = n_cycles
    # Each call to mg_fas_lin!/mg_fas! executes one full V-cycle:
    for c in 1:n_cycles
        if scheme == "finite-difference"
            converged = NS.mg_fas_lin!(1, mg_arrays, mg_mesh, dt, VF, verts, tets,
                           nothing, param, par_env, 0; iter = c)
        elseif scheme == "semi-lagrangian"
            converged = NS.mg_fas!(1, mg_arrays, mg_mesh, dt, VF, verts, tets,
                       nothing, param, par_env, 0; iter =  c)
        end
        err_norms[c+1] = norm(exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-mg_arrays[1].P_h[imin_:imax_, jmin_:jmax_, kmin_:kmax_])
        # println("Cycle $c: Error = $(err_norms[c+1])")
        if converged[]
            n_done = c
        #     println("solver: $(param.pressureSolver) converged in $iter iterations")
            break
        end
    end

    
    # Check solution error
    # L2_error = sqrt(sum((exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-mg_arrays[1].P_h[imin_:imax_,jmin_:jmax_,kmin_:kmax_]).^2))/sqrt(sum(exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_].^2))

    # Plotting routine
    P_slice = mg_arrays[1].P_h[imin_:imax_,jmin_:jmax_,1]
    exact_slice = exact_sol[imin_:imax_,jmin_:jmax_,1]
    error_slice = (exact_slice - P_slice)
    

    # Create x and y coordinates for plotting
    x_plot = x[imin_:imax_]
    y_plot = y[jmin_:jmax_]
    plt = false
    if plt 
        # Create subplots
        p1 = heatmap(x_plot, y_plot, P_slice, 
                    xlabel="x", ylabel="y", title="Computed P",
                    aspect_ratio=:equal, color=:viridis)
        
        p2 = heatmap(x_plot, y_plot, exact_slice, 
                    xlabel="x", ylabel="y", title="Exact Solution",
                    aspect_ratio=:equal, color=:viridis)
        
        p3 = heatmap(x_plot, y_plot, error_slice, 
                    xlabel="x", ylabel="y", title="Source",
                    aspect_ratio=:equal, color=:hot)
        
        p4 = plot(x_plot, P_slice[div(Nx,2),:], label="Computed P", 
                xlabel="y", ylabel="P", title="Centerline (x=0.5)",
                linewidth=2, legend=:bottomright)
        plot!(p4, x_plot, exact_slice[div(Nx,2),:], label="Exact", 
            linewidth=2, linestyle=:dash)


        plot(p1, p2, p3, p4, layout=(2,2), size=(1000,800),
            plot_title="Mesh: $(Nx)x$(Ny), L2 error: $(round(L2_error, sigdigits=4))")
        
        savefig("MMS_comparison_$(Nx)x$(Ny).png")
        println("Saved plot: MMS_comparison_$(Nx)x$(Ny).png")
    end
    return err_norms[1:n_done+1]
end

mesh_size = 64
schemes = ["semi-lagrangian","finite-difference"]
solver = ["res_iteration","gauss-seidel"]

tags    = ["WJ-SL","GS-FD"]
markers = [:circle, :pentagon, :diamond, :dtriangle]
lvl = 2
n_cycles = 50
t_start = time()
err_norms = [@time test_psolve(mesh_size,mesh_size,schemes[i],solver[i],lvl,n_cycles) for i in eachindex(schemes)]
times = time() - t_start
println("time taken: $times seconds")
println(err_norms)

# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: ‖e‖₂ vs iteration (normalized to e₀)
# Expected: single-level plateaus early (LF mode stalls); two-grid keeps falling
# ──────────────────────────────────────────────────────────────────────────────

f1 = Figure(size=(1100, 650))
ax1 = Axis(f1[1,1],
    xlabel = "Iteration / V-cycle",
    ylabel = "‖e‖₂ / ‖e₀‖₂",
    title  = "MMS-mode convergence: two-grid  (mesh=$mesh_size)",
    yscale = log10,
)

for j in eachindex(schemes)
    e0 = err_norms[j][1]
    n_pts = length(err_norms[j])
    scatterlines!(ax1, 0:(n_pts-1), err_norms[j] ./ e0;
        label     = "$(tags[j]) — two-grid",
        marker    = markers[j],
        linestyle = :solid,
    )
end

axislegend(ax1, position=:rc)
save("MMS_mode_convergence.png", f1)
println("Saved: MMS_mode_convergence.png")

# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: per-step convergence factor ρ = ‖eₙ‖/‖eₙ₋₁‖
# Expected: single-level ρ → 1 as smooth mode dominates; two-grid ρ stays small
# ──────────────────────────────────────────────────────────────────────────────

f2 = Figure(size=(1100, 650))
ax2 = Axis(f2[1,1],
    xlabel = "Iteration / V-cycle",
    ylabel = "Per-step factor  ‖eₙ‖ / ‖eₙ₋₁‖",
    title  = "Per-step convergence factor  (mesh=$mesh_size)",
    yscale = log10,
)

for j in eachindex(schemes)
    tg_factors = err_norms[j][2:end] ./ err_norms[j][1:end-1]
    scatterlines!(ax2, 1:length(tg_factors), tg_factors;
        label     = "$(tags[j]) — two-grid",
        marker    = markers[j],
        linestyle = :solid,
    )
end

hlines!(ax2, [1.0], linestyle=:dot, color=:red, label="ρ = 1  (no reduction)")
axislegend(ax2, position=:rb)
save("MMS_mode_conv_factor.png", f2)
println("Saved: MMS_mode_conv_factor.png")