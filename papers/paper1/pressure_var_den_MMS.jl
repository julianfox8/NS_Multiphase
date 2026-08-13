"""
Method of manufactured solutions for pressure test script
"""

using NavierStokes_Parallel
using Random
using CairoMakie
using Statistics
using LaTeXStrings

NS = NavierStokes_Parallel



# Define parameters 
function test_psolve(Nx,Ny,scheme,solver,lvl)
    println("Starting MMS test for pressure with Nx = $Nx and Ny = $Ny ")
    ##! still need to determine the best way to upate Nx,Ny,Nz

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
            # u[i,j,k] = -2π*sin(2π*xm[i])*cos(2π*ym[j])*dt*(2/(denx[i,j,k]+denx[i+1,j,k]))
            # v[i,j,k] = -2π*sin(2π*ym[j])*cos(2π*xm[i])*dt*(2/(deny[i,j,k]+deny[i,j+1,k]))
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
        # Volume Fraction
        rad=0.25
        xo=0.5
        yo=0.5

        for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
            VF[i,j,k]=VFcircle(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
        end
        # rad=0.25
        # xo=0.5
        # yo=0.5
        # zo=0.5
        # for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        #     VF[i,j,k]=VFbubble3d(x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],rad,xo,yo,zo)
        #     # VF[i,j,k]=VFbubble2d(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
        # end
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
    
    # compute_MMS!(uf,vf,wf,RHS,VF,dt,exact_sol,mesh,par_env)
    compute_MMS!(u,v,w,uf,vf,wf,RHS,VF,dt,exact_sol,denx,deny,denz,mesh,par_env)
    # println("u-face vel along boundary: ", uf[imin_,:,1])
    # println("u-face vel along boundary: ", uf[imin_-1,:,1])
    # println("v-face vel along boundary: ", vf[:,jmin_,1])
    # Compute band around interface
    NS.computeBand!(band,VF,param,mesh,par_env)
    # fill!(band,0.0)

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
    plt = false
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
        # hm3 = heatmap!(ax3, x_plot, y_plot, RHS_slice', colormap = :hot)
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

# mesh_size = 48
# scheme = "finite-difference"
# # solver = "geometric_mg"
# # solver = "FC_hypre"
# solver = "gauss-seidel"
# # scheme = "semi-lagrangian"
# # solver = "res_iteration_AA2"
# lvl = 4

# @time L2_err, Linf_err = test_psolve(mesh_size,mesh_size,scheme,solver,lvl)    

# times = time() - t_start
# println("time taken: $times seconds")
mesh_sizes =[16,32,64,128]
lvl = [1,1]#,3]
schemes = ["finite-difference","semi-lagrangian"]#,"semi-lagrangian"]
solvers = ["FC_hypre","res_iteration_AA_con"]#,"res_iteration"]
# tags = ["SL-FV","SL-SL"]
# schemes = ["semi-lagrangian"]#,"semi-lagrangian"]
# solvers = ["res_iteration"]#,"res_iteration"]
tags = ["SL-FV","SL-SL"]

# lvl = [1,1,1,3,1]
# schemes = ["finite-difference","finite-difference","semi-lagrangian","semi-lagrangian","semi-lagrangian"]
# solvers = ["gauss-seidel","congugateGradient","res_iteration","res_iteration","hypreSecant"]
# tags = ["FD","SL"]
# solver_tag = ["gauss-seidel","CG","NL Jacobi","FAS","Secant"]

markers = [:diamond,:circle,:diamond,:dtriangle,:pentagon]
linestyle = [:dot,:dot,:dot,:dashdot,:dashdotdot]
L2_err   = zeros(length(schemes), length(mesh_sizes))
Linf_err = zeros(length(schemes), length(mesh_sizes))
times = zeros(length(schemes), length(mesh_sizes))
for j in eachindex(schemes)
    for i in eachindex(mesh_sizes)
        t_start = time()
        L2_err[j,i], Linf_err[j,i] = test_psolve(mesh_sizes[i],mesh_sizes[i],schemes[j],solvers[j],lvl[j])    
        times[j,i] = time() - t_start
    end
end
conv_plot = true
timing_plot = false

if conv_plot
    # ---------------------------------------------
    # Convergence Analysis and Plotting (log-log)
    # ---------------------------------------------
    println("\nMesh sizes = ", mesh_sizes)
    println("L2 errors  = ", L2_err)
    println("L∞ errors  = ", Linf_err)

    # Vertical offsets for separating overlapping curves
    offsets = [0.15, 0.145]   # apply only to the first scheme

    # ------------------------
    # L2 convergence plot
    # ------------------------
    Ns = round.(Int, 1 .// mesh_sizes) 
    f = Figure(size = (700,500))
    pL2 = Axis(
        f[1,1],
        xlabel = L"Mesh \ Size",
        ylabel = L"L_2 \ error",
        xscale = log10,
        yscale = log10,
        # title = "MMS pressure solver",
        xticks = (mesh_sizes),
        # titlesize = 30,
        xlabelsize = 28,
        ylabelsize = 32,
        xticklabelsize = 22,
        yticklabelsize = 28
    )


    # colors = ["blue","orange"]
    for j in eachindex(schemes)

        
        vals = L2_err[j, :] .* 10^offsets[j]

        # lines!(pL2, mesh_sizes, vals,
        #     linestyle = linestyle[j],
        #     label = tags[j],
        #     # color = colors[j],
        #     # linewidth = 3
        # )

        # scatter!(pL2, mesh_sizes, vals,
        #     marker = markers[j],
        #     # color = colors[j],
        #     markersize = 12
        # )
        scatterlines!(pL2, mesh_sizes, vals,
            linestyle = linestyle[j],
            label = tags[j],
            marker = markers[j],
            linewidth = 2,
            # color = colors[j],
            markersize = 12
        )

    end
    # Reference slopes (plotted as lines)
    ref1_L2 = (L2_err[1,1] .* (mesh_sizes ./ mesh_sizes[1]).^(-1)).*10^0.3  # 1st-order slope
    ref2_L2 = L2_err[1,1] .* (mesh_sizes ./ mesh_sizes[1]).^(-2)  # 2nd-order slope

    lines!(pL2, mesh_sizes, ref1_L2, linestyle=:dash, label="O(Δx)",linewidth=2)
    lines!(pL2, mesh_sizes, ref2_L2, linestyle=:dash, label="O(Δx²)",linewidth=2)

    axislegend(pL2, position = :rt,labelsize = 20)
    save( "L2_convergence.png",f)
    println("Saved L2 plot: L2_convergence.png")

    # ------------------------
    # L∞ convergence plot
    # ------------------------
    # pLinf = plot(
    #     xlabel = "N",
    #     ylabel = "L∞ error",
    #     xscale = :log10,
    #     yscale = :log10,
    #     legend = :bottomleft
    # )

    # # Reference slopes
    # ref1_Linf = ( Linf_err[1,1] .* (mesh_sizes ./ mesh_sizes[1]).^(-1)) .*10^0.3 # 1st-order
    # ref2_Linf = Linf_err[1,1] .* (mesh_sizes ./ mesh_sizes[1]).^(-2)  # 2nd-order

    # plot!(pLinf, mesh_sizes, ref1_Linf, linestyle=:dash, label="1st order")
    # plot!(pLinf, mesh_sizes, ref2_Linf, linestyle=:dash, label="2nd order")

    # # Plot scheme results
    # for j in eachindex(schemes)
    #     plot!(pLinf, mesh_sizes, Linf_err[j, :] .* 10^offsets[j],
    #         label = "$(tags[j])",
    #         linewidth = 3,
    #         markershape = markers[j]
    #     )
    # end

    # savefig(pLinf, "Linf_convergence.png")
    # println("Saved L∞ plot: Linf_convergence.png")

end

if timing_plot 
    # ---------------------------------------------
    # Timing Plot
    # ---------------------------------------------
    println("\nTiming results (seconds):")
    println(times)
    f = Figure(size = (900,600))
    pTime = Axis(f[1,1],
        xlabel = "N",
        ylabel = "Wall-clock time (s)",
        xscale = log10,
        yscale = log10,
        xticks = (mesh_sizes[2:end]),
        title = "Timing vs Resolution",
        titlesize = 30,
        xlabelsize = 24,
        ylabelsize = 24,
        xticklabelsize = 18,
        yticklabelsize = 18
    )

    # Optional: offsets to separate curves vertically (log space)
    time_offsets = [0.0, 0.0, 0.0, 0.0, 0.0]   # adjust if needed, same length as schemes

    for j in eachindex(schemes)
        scatterlines!(
            pTime,
            mesh_sizes[2:end],
            times[j, 2:end] .* 10 .^ time_offsets[j],
            label = "$(tags[j])",
            linewidth = 3
            # markershape = markers[j]
        )
    end
    axislegend(pTime, position = :lt,labelsize = 24)
    save( "timing_plot.png",f)
    println("Saved timing plot: timing_plot.png")
end