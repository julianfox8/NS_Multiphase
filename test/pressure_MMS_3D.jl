"""
Method of manufactured solutions for pressure test script
"""

using NavierStokes_Parallel
using Random
using Plots
using Statistics
using Profile
using ProfileView
using FlameGraphs

NS = NavierStokes_Parallel



# Define parameters 
function test_psolve(Nx,Ny,Nz,scheme,solver,lvl)
    println("Starting MMS test for pressure with Nx = $Nx and Ny = $Ny ")
    ##! still need to determine the best way to upate Nx,Ny,Nz

    param = parameters(
        # Constants
        mu_liq=1,       # Dynamic viscosity of liquid (N/m)
        mu_gas = 0.1,   # Dynamic viscosity of gas (N/m)
        rho_liq= 1.0,     # Density of liquid (kg/m^3)
        rho_gas = 1.0,  # Density of gas (kg/m^3)
        sigma = 0.0,    # Surface tension coefficient (N/m)
        grav_x = 0.0,   # Gravity  (m/s^2)
        grav_y = 0.0,   # Gravity (m/s^2)
        grav_z = 0.0,   # Gravity (m/s^2)
        Lx=1.0,        # Domain size of 8Dx30Dx8D where D is bubble diameter(m)
        Ly=1.0,             
        Lz=1.0,
        tFinal=100.0,      # Simulation time

        
        # Discretization inputs
        Nx=Nx,           # Number of grid cells
        Ny=Ny,
        Nz=Nz,
        stepMax=50,   # Maximum number of timesteps
        max_dt = 1e-3,
        CFL=0.4,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
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

        pressure_scheme = scheme,
        pressureSolver = solver,
        # pressureSolver = "hypreSecant",
        # pressurePrecond = "nl_jacobi",

        # pressure_scheme = "finite-difference",
        # pressureSolver = "congugateGradient",
        # pressureSolver = "FC_hypre",
        # pressureSolver = "gauss-seidel",

        hypreSolver = "GMRES-AMG",
        mg_lvl = lvl,
        projection_method = "RK4",
        tesselation = "5_tets",
        
        iter_type = "standard",
        test_case = "psolve_test", 

    )



    """
    Compute manufactured solution and source term
    """
    function compute_MMS!(uf,vf,wf,RHS,VF,dt,exact,mesh,par_env)
        @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
        @unpack jmin_,jmax_,xm,ym,imin_,imax_,jmin_,jmax_,kmin_,kmax_,dy,dx,dz = mesh
        @unpack xper,yper,zper,rho_gas,rho_liq,pressure_scheme = param
        k = 1

        # this for loop is used for MMS applied strictly to RHS
        for i = imin_:imax_, j = jmin_:jmax_+1
            rho = (VF[i,j,1] == 1.0) ? rho_liq : rho_gas
            exact[i,j,1] = cos(2π*ym[j])
            if pressure_scheme == "semi-lagrangian"
                vf[i,j,k] = -2π*sin(2π*y[j])/rho*dt
            elseif pressure_scheme == "finite-difference"
                vf[i,j,k] = -2π*sin(2π*y[j])/rho
            end
            # vf[i,j,k] = -2π*sin(2π*y[j])/rho
        #     RHS[i,j,1] = -(4*π^2/rho)*cos(2π*ym[j])*dt
        #     # RHS[i,j,1] = -(8*π^2/rho)*cos(2π*ym[j])*cos(2π*xm[i])
            
        end
        for i = imin_:imax_, j = jmin_:jmax_
            RHS[i,j,k] = (vf[i,j+1,k]-vf[i,j,k])/dy
        end
        # for  j = jmin_-1:jmax_+2,i = imin_-1:imax_+1
        #     rho = (VF[i,j,1] == 1.0) ? rho_liq : rho_gas
        #     if pressure_scheme == "semi-lagrangian"
        #         vf[i,j,1] = -2*π*sin(2π*y[j])cos(2π*x[i])/rho*dt
        #     elseif pressure_scheme == "finite-difference"
        #         vf[i,j,1] = -2*π*sin(2π*y[j])cos(2π*x[i])/rho
        #     end
        # end

        # for j = jmin_-1:jmax_+1, i = imin_-1:imax_+2
        #     rho = (VF[i,j,1] == 1.0) ? rho_liq : rho_gas
        #     if pressure_scheme == "semi-lagrangian"
        #         uf[i,j,1] = 0.0#-2*π*sin(2π*x[i])cos(2π*y[j])/rho*dt
        #     elseif pressure_scheme == "finite-difference"
        #         uf[i,j,1] = 0.0 #-2*π*sin(2π*x[i])cos(2π*y[j])/rho
        #     end
        # end

        # for i = imin_:imax_, j = jmin_:jmax_
        #     exact[i,j,1] = cos(2π*ym[j])*cos(2π*xm[i])
        #     RHS[i,j,k] = (vf[i,j+1,k]-vf[i,j,k])/dy # + (uf[i+1,j,k]-uf[i,j,k])/dx
        # end

        return nothing
    end


    """
    Compute manufactured solution and source term
    """
    function compute_MMS!(u,v,w,uf,vf,wf,RHS,VF,dt,exact,denx,deny,denz,mesh,par_env)
        @unpack irankx, iranky, irankz, nprocx, nprocy, nprocz = par_env
        @unpack jmin_,jmax_,xm,ym,imin_,imax_,jmin_,jmax_,kmin_,kmax_,dy,dx,dz = mesh
        @unpack xper,yper,zper,rho_gas,rho_liq,pressure_scheme = param
        k = 1

        # this for loop is used for MMS applied strictly to RHS
        for k = kmin_-1:kmax_+1,j = jmin_-1:jmax_+1,i = imin_-1:imax_+1 
            exact[i,j,1] = cos(2π*ym[j])
            if pressure_scheme == "semi-lagrangian"
                v[i,j,k] = -2π*sin(2π*ym[j])*dt*(2/(deny[i,j,k]+deny[i,j+1,k]))
            elseif pressure_scheme == "finite-difference"
                v[i,j,k] = -2π*sin(2π*ym[j])*(2/(deny[i,j,k]+deny[i,j+1,k]))
            end
        end

        # apply periodic BC
        NS.update_borders!(u,mesh,par_env)
        NS.update_borders!(v,mesh,par_env)
        NS.update_borders!(w,mesh,par_env)

        # Create face velocities
        NS.interpolateFace!(u,v,w,uf,vf,wf,mesh)

        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            # RHS[i,j,k] = 2/(deny[i,j,k]+deny[i,j+1,k])*(vf[i,j+1,k]-vf[i,j,k])/dy
            RHS[i,j,k] = (vf[i,j+1,k]-vf[i,j,k])/dy
        end

        # for  j = jmin_-1:jmax_+2,i = imin_-1:imax_+1
        #     rho = (VF[i,j,1] == 1.0) ? rho_liq : rho_gas
        #     if pressure_scheme == "semi-lagrangian"
        #         vf[i,j,1] = -2*π*sin(2π*y[j])cos(2π*x[i])/rho*dt
        #     elseif pressure_scheme == "finite-difference"
        #         vf[i,j,1] = -2*π*sin(2π*y[j])cos(2π*x[i])/rho
        #     end
        # end

        # for j = jmin_-1:jmax_+1, i = imin_-1:imax_+2
        #     rho = (VF[i,j,1] == 1.0) ? rho_liq : rho_gas
        #     if pressure_scheme == "semi-lagrangian"
        #         uf[i,j,1] = 0.0#-2*π*sin(2π*x[i])cos(2π*y[j])/rho*dt
        #     elseif pressure_scheme == "finite-difference"
        #         uf[i,j,1] = 0.0 #-2*π*sin(2π*x[i])cos(2π*y[j])/rho
        #     end
        # end

        # for i = imin_:imax_, j = jmin_:jmax_
        #     exact[i,j,1] = cos(2π*ym[j])*cos(2π*xm[i])
        #     RHS[i,j,k] = (vf[i,j+1,k]-vf[i,j,k])/dy # + (uf[i+1,j,k]-uf[i,j,k])/dx
        # end

        return nothing
    end

    """
    VF IC
    """
    function IC!(VF,mesh)
        @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,
                    xm,ym,y,Lx,Ly,Lz,dy = mesh

        for k = kmin_:kmax_,i = imin_:imax_
            #height
            height = Ly/2 

            for j = jmin_:jmax_        
                if y[j] < height && y[j+1] > height 
                    VF[i,j,k] = (height - y[j])/(y[j+1]-y[j]) 
                elseif y[j] < height
                    VF[i,j,k] = 1.0
                else
                    VF[i,j,k] = 0.0
                end
            end
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
    mg_arrays = NS.mg_initArrays(mg_mesh,param,p_min,p_max,par_env)

    # Compute dt
    dt = NS.compute_dt(u,v,w,param,mesh,par_env)
    t = 0.0 :: Float64

    # Create source term/exact solution and apply BC to Pressure
    # VF[:,:,:] .= 0.0
    IC!(VF,mesh)
    # NS.update_borders!(VF,mesh,par_env)
    # compute density and viscosity at intial conditions
    NS.compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

    # compute_MMS!(uf,vf,wf,RHS,VF,dt,exact_sol,mesh,par_env)
    compute_MMS!(u,v,w,uf,vf,wf,RHS,VF,dt,exact_sol,denx,deny,denz,mesh,par_env)
    
    # stats = check_neumann!(uf, vf, mesh; tol=1e-10)

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
        if param.pressureSolver == "FC_hypre"
            iter = NS.FC_hypre_solver(P,RHS,tmp2,denx,deny,denz,tmp5,mg_arrays.jacob[1],mg_arrays.x_vec[1],mg_arrays.b_vec[1],dt,param,mesh,par_env,1000)
        elseif param.pressureSolver == "gauss-seidel"
            iter = NS.gs(P,RHS,tmp2,denx,deny,denz,dt,param,mg_mesh.mesh_lvls[1],par_env;max_iter=100000)
        elseif param.pressureSolver == "congugateGradient"
            iter = NS.cg!(P, RHS, denx, deny, denz,tmp6,dt, param, mg_mesh.mesh_lvls[1], par_env)  
        end
    elseif param.pressure_scheme == "semi-lagrangian"
        if param.pressureSolver == "res_iteration"
            iter = NS.res_iteration(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp6,tmp2,tmp7,tmp4,verts,tets,param,mg_mesh.mesh_lvls[1],par_env;) 
        elseif param.pressureSolver == "hypreSecant"
            iter = NS.Secant_jacobian_hypre!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp6,tmp2,tmp7,tmp4,verts,tets,mg_arrays.jacob[1],mg_arrays.x_vec[1],mg_arrays.b_vec[1],param,mg_mesh.mesh_lvls[1],par_env)
        end
    end
    println("solver: $(param.pressureSolver) converged in $iter iterations")
    # Check solution error
    L2_error = sqrt(sum((exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]).^2))/sqrt(sum(exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_].^2))

    Linf_error = maximum(abs.(exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_]-P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]))/maximum(abs.(exact_sol[imin_:imax_,jmin_:jmax_,kmin_:kmax_]))
      # Plotting routine
    P_slice = P[imin_:imax_,jmin_:jmax_,div(Nz,2)]
    exact_slice = exact_sol[imin_:imax_,jmin_:jmax_,div(Nz,2)]
    RHS_slice = RHS[imin_:imax_,jmin_:jmax_,div(Nz,2)]
    error_slice = (exact_slice - P_slice)
    

    # Create x and y coordinates for plotting
    x_plot = x[imin_:imax_]
    y_plot = y[jmin_:jmax_]
    plt = false
    if plt 
        # Create subplots
        p1 = heatmap(x_plot, y_plot, P_slice', 
                    xlabel="x", ylabel="y", title="Computed P",
                    aspect_ratio=:equal, color=:viridis)
        
        p2 = heatmap(x_plot, y_plot, exact_slice', 
                    xlabel="x", ylabel="y", title="Exact Solution",
                    aspect_ratio=:equal, color=:viridis)
        
        p3 = heatmap(x_plot, y_plot, error_slice', 
                    xlabel="x", ylabel="y", title="Source",
                    aspect_ratio=:equal, color=:hot)
        
        p4 = plot(x_plot, P_slice[div(Nx,2),:], label="Computed P", 
                xlabel="y", ylabel="P", title="Centerline (x=0.5)",
                linewidth=2, legend=:bottomright)
        plot!(p4, x_plot, exact_slice[div(Nx,2),:], label="Exact", 
            linewidth=2, linestyle=:dash)

        # p4 = plot(x_plot, P_slice[:,div(Ny,2)], label="Computed P", 
        #         xlabel="y", ylabel="P", title="Centerline (y=0.5)",
        #         linewidth=2, legend=:bottomright)
        # plot!(p4, x_plot, exact_slice[:,div(Ny,2)], label="Exact", 
        #     linewidth=2, linestyle=:dash)

        plot(p1, p2, p3, p4, layout=(2,2), size=(1000,800),
            plot_title="Mesh: $(Nx)x$(Ny), L2 error: $(round(L2_error, sigdigits=4))")
        
        savefig("MMS_comparison_$(Nx)x$(Ny).png")
        println("Saved plot: MMS_comparison_$(Nx)x$(Ny).png")
    end
    return L2_error,Linf_error
end

# mesh_sizes =[24,32,48,64]
# lvl = [1,3,1,1]
# schemes = ["semi-lagrangian","semi-lagrangian","semi-lagrangian","finite-difference"]
# solvers = ["hypreSecant","res_iteration","res_iteration","FC_hypre"]
# solver_tag = ["Secant","FAS","NLW Jacobi","GMRES"]

mesh_sizes =[24,32,48,64]
lvl = [1,3,1,1]
schemes = ["semi-lagrangian","semi-lagrangian","finite-difference"]#,"finite-difference"]
solvers = ["res_iteration","res_iteration","FC_hypre"]#,"gauss-seidel"]
solver_tag = ["NLW Jacobi","FAS","GMRES","GS"]

markers = [:circle,:square,:diamond,:dtriangle,:pentagon]
L2_err   = zeros(length(schemes), length(mesh_sizes))
Linf_err = zeros(length(schemes), length(mesh_sizes))
times = zeros(length(schemes), length(mesh_sizes))
for j in eachindex(schemes)
    for i in eachindex(mesh_sizes)
        t_start = time()
        L2_err[j,i], Linf_err[j,i] = test_psolve(mesh_sizes[i],mesh_sizes[i],mesh_sizes[i],schemes[j],solvers[j],lvl[j])    
        times[j,i] = time() - t_start
    end
end
conv_plot = false
timing_plot = true

if conv_plot
    # ---------------------------------------------
    # Convergence Analysis and Plotting (log-log)
    # ---------------------------------------------
    println("\nMesh sizes = ", mesh_sizes)
    println("L2 errors  = ", L2_err)
    println("L∞ errors  = ", Linf_err)

    # Vertical offsets for separating overlapping curves
    offsets = [0.15, 0.1]   # apply only to the first scheme

    # ------------------------
    # L2 convergence plot
    # ------------------------
    pL2 = plot(
        xlabel = "N",
        ylabel = "L₂ error",
        xscale = :log10,
        yscale = :log10,
        legend = :bottomleft
    )

    # Reference slopes (plotted as lines)
    ref1_L2 = (L2_err[1,1] .* (mesh_sizes ./ mesh_sizes[1]).^(-1)).*10^0.3  # 1st-order slope
    ref2_L2 = L2_err[1,1] .* (mesh_sizes ./ mesh_sizes[1]).^(-2)  # 2nd-order slope

    plot!(pL2, mesh_sizes, ref1_L2, linestyle=:dash, label="1st order")
    plot!(pL2, mesh_sizes, ref2_L2, linestyle=:dash, label="2nd order")

    # Plot scheme results
    for j in eachindex(schemes)
        plot!(pL2, mesh_sizes, L2_err[j, :] .* 10^offsets[j],
            label = "$(tags[j])",
            linewidth = 3,
            markershape = markers[j]
        )
    end

    savefig(pL2, "L2_convergence.png")
    println("Saved L2 plot: L2_convergence.png")

    # ------------------------
    # L∞ convergence plot
    # ------------------------
    pLinf = plot(
        xlabel = "N",
        ylabel = "L∞ error",
        xscale = :log10,
        yscale = :log10,
        legend = :bottomleft
    )

    # Reference slopes
    ref1_Linf = ( Linf_err[1,1] .* (mesh_sizes ./ mesh_sizes[1]).^(-1)) .*10^0.3 # 1st-order
    ref2_Linf = Linf_err[1,1] .* (mesh_sizes ./ mesh_sizes[1]).^(-2)  # 2nd-order

    plot!(pLinf, mesh_sizes, ref1_Linf, linestyle=:dash, label="1st order")
    plot!(pLinf, mesh_sizes, ref2_Linf, linestyle=:dash, label="2nd order")

    # Plot scheme results
    for j in eachindex(schemes)
        plot!(pLinf, mesh_sizes, Linf_err[j, :] .* 10^offsets[j],
            label = "$(tags[j])",
            linewidth = 3,
            markershape = markers[j]
        )
    end

    savefig(pLinf, "Linf_convergence.png")
    println("Saved L∞ plot: Linf_convergence.png")

end

if timing_plot 
    # ---------------------------------------------
    # Timing Plot
    # ---------------------------------------------
    println("\nTiming results (seconds):")
    println(times)

    pTime = plot(
        xlabel = "N",
        ylabel = "Wall-clock time (s)",
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        title = "Timing vs Resolution"
    )

    # Optional: offsets to separate curves vertically (log space)
    time_offsets = [0.0, 0.0, 0.0, 0.0, 0.0]   # adjust if needed, same length as schemes

    for j in eachindex(schemes)
        plot!(
            pTime,
            mesh_sizes,
            times[j, :] .* 10 .^ time_offsets[j],
            label = "$(solver_tag[j])",
            linewidth = 3,
            markershape = markers[j]
        )
    end

    savefig(pTime, "timing_plot_3D.png")
    println("Saved timing plot: timing_plot_3D.png")
end