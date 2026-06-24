"""
Test the smoothening property of a solver
"""

using NavierStokes_Parallel
using Random
using CairoMakie
using Statistics
using LinearAlgebra

NS = NavierStokes_Parallel

# Define parameters 
function test_smoothing(modes,sweep,Nx,Ny,scheme,solver,lvl,xper,yper,zper;plt=false)
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
        max_dt = 2.5e-3,
        CFL=1.0,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
        std_out_period = 0.0,
        out_period=1,     # Number of steps between when plots are updated
        tol = 1e-8,

        # Processors 
        nprocx = 1,
        nprocy = 1,
        nprocz = 1,

        # Periodicity
        xper = xper,
        yper = yper,
        zper = zper,

        pressure_scheme = scheme,
        pressureSolver = solver,

        hypreSolver = "GMRES-AMG",
        mg_lvl = lvl,
        # projection_method = "RK4",
        # projection_method = "Euler",
        projection_method = "Heun",
        tesselation = "5_tets",
        
        iter_type = "standard",
        test_case = "psolve_test", 

    )


    """
    Fourier mode injection
    """
    function set_modes!(modes,P,mesh,par_env)
        @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
        @unpack jmin_,jmax_,xm,ym,imin_,imax_,jmin_,jmax_,kmin_,kmax_,dy,dx,dz = mesh
        @unpack xper,yper,zper,rho_gas,rho_liq,pressure_scheme = param
        k = 1

        # this for loop is used for MMS applied strictly to RHS
        # for mode in modes
        for i = imin_:imax_, j = jmin_:jmax_
            P[i,j,k] = sin(2π*modes[1]*xm[i])
            # P[i,j,k] += sin(2*mode*π*ym[j])sin(2*mode*π*xm[i]) 
        end
        
        # apply BC
        NS.Neumann!(P,mesh,par_env)
        NS.update_borders!(P,mesh,par_env)
        
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

    # Initialize volume fraction
    fill!(VF,0.0)
    
    # compute density and viscosity at intial conditions
    NS.compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

    # Inject fourier modes and apply neumann or periodic BC
    set_modes!(modes,P,mesh,par_env)
    
    # Compute initial error
    init_L2_err = norm(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
    exact_sol = copy(P)

    # Compute band around interface
    NS.computeBand!(band,VF,param,mesh,par_env)
    # fill!(band,1.0)

    # Loop over time
    nstep = 0
    iter = 0

    # # Call pressure Solver (handles processor boundaries for P)
    if param.mg_lvl > 1
        iter = NS.mg_cycler(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,mg_arrays,mg_mesh,VF,verts,tets,param,par_env) 
    elseif param.pressure_scheme == "finite-difference"
        if param.pressureSolver == "gauss-seidel"
            iter = NS.gs(P,RHS,tmp2,denx,deny,denz,dt,param,mg_mesh.mesh_lvls[1],par_env;max_iter=sweep)
        elseif param.pressureSolver == "jacobi"
            iter = NS.jacobi(P,tmp6,RHS,tmp2,denx,deny,denz,dt,param,mg_mesh.mesh_lvls[1],par_env;max_iter=sweep)
        end
    elseif param.pressure_scheme == "semi-lagrangian"
        if param.pressureSolver == "res_iteration"
            iter = NS.res_iteration(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp6,tmp2,tmp7,tmp4,verts,tets,param,mg_mesh.mesh_lvls[1],par_env;max_iter = sweep) 
        elseif param.pressureSolver == "res_iteration_AA"
            iter = NS.res_iteration_AA(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp6,tmp2,tmp7,tmp4,verts,tets,param,mg_mesh.mesh_lvls[1],par_env;max_iter = sweep) 
        elseif param.pressureSolver == "hypreSecant"
            iter = NS.Secant_jacobian_hypre!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp6,tmp2,tmp7,tmp4,verts,tets,mg_arrays.jacob[1],mg_arrays.x_vec[1],mg_arrays.b_vec[1],param,mg_mesh.mesh_lvls[1],par_env)
        end
    end
    
    # Check solution error
    L2_err = norm(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
    # err_reduction = L2_err^(1/sweep)/init_L2_err^(1/sweep)
    err_reduction = L2_err/init_L2_err
    # println("Error after $(sweep) sweeps: $(L2_err)")

    # Plotting routine
    P_slice = P[imin_:imax_,jmin_:jmax_,1]
    exact_slice = exact_sol[imin_:imax_,jmin_:jmax_,1]
    
    
    # Create x and y coordinates for plotting
    x_plot = x[imin_:imax_]
    y_plot = y[jmin_:jmax_]

    if plt
        # Shared color range across both panels
        clim = (-1.0, 1.0)

        fig = Figure(size = (1200, 500))

        # -----------------------------
        # Heatmap 1: Computed P
        # -----------------------------
        ax1 = Axis(fig[1, 1],
            xlabel = "x",
            ylabel = "y",
            title = "Computed P",
            aspect = DataAspect()
        )
        hm1 = heatmap!(ax1, x_plot, y_plot, P_slice;
            colormap = :RdBu,
            colorrange = clim
        )
        Colorbar(fig[1, 2], hm1; label = "P")

        # -----------------------------
        # Heatmap 2: Exact solution
        # -----------------------------
        ax2 = Axis(fig[1, 3],
            xlabel = "x",
            ylabel = "y",
            title = "Exact",
            aspect = DataAspect()
        )
        hm2 = heatmap!(ax2, x_plot, y_plot, exact_slice;
            colormap = :RdBu,
            colorrange = clim
        )
        Colorbar(fig[1, 4], hm2; label = "P")

        # -----------------------------
        # Global title
        # -----------------------------
        Label(fig[0, :],
            "Mesh: $(Nx)x$(Ny), L2 error: $(round(L2_err, sigdigits=4))",
            fontsize = 18
        )

        display(fig)
    end
    return err_reduction
end

# Set simulation parameter
mesh_size = 64
sweeps = 1
xper = true ; yper = true ; zper = true

# test smoothing factor across all high-frequency modes for a fixed sweep count
schemes = ["finite-difference","finite-difference","semi-lagrangian"]
solvers = ["gauss-seidel","jacobi","res_iteration"]
tags = ["GS","Jacobi","SL-SL"]
hf_modes = collect(1 : mesh_size)
smoothing_factor = zeros(length(schemes), length(hf_modes))
markers = [:circle,:diamond,:dtriangle,:pentagon]

# set plots
mode_plt = true
plt = false

for j in eachindex(schemes)
    for (mi, m) in enumerate(hf_modes)
        smoothing_factor[j, mi] = test_smoothing([m], sweeps, mesh_size, mesh_size, schemes[j], solvers[j], 1, xper, yper, zper; plt=false)
    end
end

# ------------------------
# Smoothing factor vs mode plot
# ------------------------
if mode_plt
    f2 = Figure(size = (1000, 800))
    ax2 = Axis(f2[1,1],
        xlabel = "Fourier mode k",
        ylabel = "smoothing factor (L2 reduction after $sweeps sweeps)",
        title = "Smoothing factor, mesh=$mesh_size",
        # yscale = log10,
    )

    for j in eachindex(schemes)
        scatterlines!(ax2, hf_modes, smoothing_factor[j, :],
            label = tags[j],
            marker = markers[j],
        )
    end

    vlines!(ax2, [mesh_size÷4 + 0.5, 3*(mesh_size÷4) - 0.5]; color=:black, linestyle=:dash, label="HF boundary")

    axislegend(ax2, position = :rt)
    # display(f2)
    save("smoothing_factor_vs_mode.png", f2)
    println("Saved smoothing factor plot: smoothing_factor_vs_mode.png")
end

#! test error reduction for a single mode with various numbers of sweeps
# tags = ["GS","Jacobi","SL-SL"]#,"SL_SL AA"]
# schemes = ["finite-difference","finite-difference","semi-lagrangian","semi-lagrangian"]
# solvers = ["gauss-seidel","jacobi","res_iteration","res_iteration_AA"]
# modes = [mesh_size/3]

# err_reduction  = zeros(length(schemes), length(range(1,sweeps)))

# for j in eachindex(schemes)
#     for i in range(1,sweeps)
#        err_reduction[j,i] = test_smoothing(modes,i,mesh_size,mesh_size,schemes[j],solvers[j],1;plt=false)    
#     end
# end

# conv_plot = false

# if conv_plot
#     # ---------------------------------------------
#     # Convergence Analysis and Plotting (log-log)
#     # ---------------------------------------------
#     println("\nMesh sizes = ", mesh_size)

#     # ------------------------
#     # L2 convergence plot (error vs sweeps, single mode)
#     # ------------------------
#     f = Figure(size = (1000, 800))
#     p_red = Axis(f[1,1],
#         xlabel = "Sweeps",
#         ylabel = "error reduction",
#         yscale = log10,
#         )

#     for j in eachindex(schemes)
#         scatterlines!(p_red, range(1,sweeps), err_reduction[j, :],
#             label = "$(tags[j])",
#         )
#     end
#     axislegend(p_red, position = :rt)
#     save("err_reduct_test.png",f)
#     println("Saved error reduction plot: err_reduct_test.png")

# end