"""
Two-grid convergence factor analysis for linear multigrid solvers.

Theory: for the homogeneous problem (RHS=0, exact P*=0), the error IS P.
Inject a pure Fourier mode, apply one V-cycle, measure ρ = ‖e_after‖/‖e_before‖.

Two analyses:
  1. ρ vs Fourier mode k over the high-frequency band [N/4+1, N/2]
  2. Per-cycle ρ vs V-cycle number for the worst-case mode
"""

using NavierStokes_Parallel
using CairoMakie
using LinearAlgebra

NS = NavierStokes_Parallel

# ──────────────────────────────────────────────────────────────────────────────
# Core test function
# ──────────────────────────────────────────────────────────────────────────────

"""
    test_twogrid(mode, n_cycles, Nx, scheme, solver) -> conv_per_cycle::Vector

Injects Fourier mode `mode` as the initial error on a 1D mesh of size `Nx`,
then applies `n_cycles` two-grid V-cycles (mg_lvl=2, 5 pre/post GS smooths,
exact coarse solve). Returns a vector of per-cycle convergence factors
ρ[c] = ‖e_c‖/‖e_{c-1}‖.
"""
function test_twogrid(mode, n_cycles, Nx, scheme, solver)
    param = parameters(
        mu_liq   = 0.0,
        mu_gas   = 0.0,
        rho_liq  = 1.0,
        rho_gas  = 1.0,
        sigma    = 0.0,
        grav_x   = 0.0,
        grav_y   = 0.0,
        grav_z   = 0.0,
        Lx = 1.0, Ly = 1.0, Lz = 1.0/100,
        tFinal   = 100.0,
        Nx = Nx, Ny = Nx, Nz = 1,
        stepMax  = 50,
        max_dt   = 2.5e-4,
        CFL      = 1.0,
        std_out_period = 0.0,
        out_period     = 1,
        tol = 1e-14,   # keep small so pre-smoother never exits early
        nprocx = 1, nprocy = 1, nprocz = 1,
        xper = false, yper = false, zper = true,
        pressure_scheme = scheme,
        pressureSolver  = solver,
        hypreSolver     = "GMRES-AMG",
        mg_lvl          = 2,          # two-grid: fine + one coarse level
        projection_method = "Heun",
        tesselation = "5_tets",
        iter_type   = "standard",
        test_case   = "psolve_test",
    )

    # 1D-aware mode injection: Ny=1 means ym[j]=0.5, so sin(2kπ·0.5)=0 for integer k.
    # Only apply the sine in active (>1 cell) directions.
    function set_mode!(mode, P, mesh, par_env)
        @unpack imin_, imax_, jmin_, jmax_, xm, ym = mesh
        kz = 1
        for i = imin_:imax_, j = jmin_:jmax_
            xterm = sin(2 * mode * π * xm[i])
            yterm = param.Ny > 1 ? sin(2 * mode * π * ym[j]) : 1.0
            P[i, j, kz] += xterm * yterm
        end
        NS.update_borders!(P, mesh, par_env)
    end

    par_env = NS.parallel_init(param)
    mg_mesh = NS.init_mg_mesh(param, par_env)
    mesh    = mg_mesh.mesh_lvls[1]
    @unpack imin_, imax_, jmin_, jmax_, kmin_, kmax_ = mesh

    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,RHS,tmp2,exact_sol,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,
    tmplrg,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,divg,mask,
    tets,verts,inds,vInds = NS.initArrays(mesh)

    p_min, p_max = NS.prepare_indices(tmp5, par_env, mesh)
    mg_arrays    = NS.mg_initArrays(mg_mesh, param, par_env)

    dt = NS.compute_dt(u, v, w, param, mesh, par_env)

    fill!(VF,   0.0)
    fill!(band, 1.0)
    NS.compute_props!(denx, deny, denz, viscx, viscy, viscz, VF, param, mesh)

    # Inject pure error mode (RHS=0, exact P*=0, so error = P)
    fill!(P, 0.0)
    set_mode!(mode, P, mesh, par_env)
    NS.Neumann!(P, mesh, par_env)

    # Copy all level-1 fields into mg_arrays once (densities, velocities, band stay fixed)
    fields = (P = P, uf = uf, vf = vf, wf = wf,
              denx = denx, deny = deny, denz = denz,
              gradx = gradx, grady = grady, gradz = gradz,
              band = band)
    NS.copy_to_mg!(mg_arrays, fields, 1)

    # Initial error norm
    e0 = norm(mg_arrays[1].P_h[imin_:imax_, jmin_:jmax_, kmin_:kmax_])

    if e0 < 1e-14
        return fill(NaN, n_cycles)
    end

    # Run V-cycles and record per-cycle convergence factor
    err_norms = zeros(n_cycles + 1)
    err_norms[1] = e0

    # Each call to mg_fas_lin!/mg_fas! executes one full V-cycle:
    #   lvl=1         → always start at finest level
    #   pvtk_iter=0   → dummy VTK counter (pvd_data=nothing so unused)
    #   iter=nothing  → no outer convergence logging needed
    # The number of pre/post smooths (v1=v2=5) is hardcoded inside mg_fas_lin!/mg_fas!
    for c in 1:n_cycles
        if scheme == "finite-difference"
            NS.mg_fas_lin!(1, mg_arrays, mg_mesh, dt, VF, verts, tets,
                           nothing, param, par_env, 0; iter = nothing)
        elseif scheme == "semi-lagrangian"
            NS.mg_fas!(1, mg_arrays, mg_mesh, dt, VF, verts, tets,
                       nothing, param, par_env, 0; iter = nothing)
        end
        err_norms[c+1] = norm(mg_arrays[1].P_h[imin_:imax_, jmin_:jmax_, kmin_:kmax_])
    end

    # Per-cycle factors ρ[c] = ‖e_c‖/‖e_{c-1}‖
    conv_per_cycle = err_norms[2:end] ./ err_norms[1:end-1]
    return conv_per_cycle
end

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

mesh_size = 64
n_cycles  = 5     # V-cycles to run per mode

# tags    = ["GS-FD"]
# schemes = ["finite-difference"]
# solvers = ["gauss-seidel"]
tags    = ["WJ-SL","GS-FD"]
schemes = ["semi-lagrangian","finite-difference"]
solvers = ["res_iteration","gauss-seidel"]
markers = [:circle, :square, :diamond, :dtriangle, :pentagon]

hf_modes = collect(mesh_size÷4+1 : mesh_size÷2)   # high-frequency band

# ──────────────────────────────────────────────────────────────────────────────
# Analysis 1: convergence factor vs Fourier mode (1 V-cycle each)
# ──────────────────────────────────────────────────────────────────────────────
println("=== Two-grid convergence factor vs mode (1 V-cycle) ===")

conv_vs_mode = zeros(length(schemes), length(hf_modes))

for j in eachindex(schemes)
    for (mi, m) in enumerate(hf_modes)
        cpc = test_twogrid(m, 1, mesh_size, schemes[j], solvers[j])
        
        conv_vs_mode[j, mi] = cpc[1]
        print("  $(tags[j])  k=$m  ρ=$(round(cpc[1], sigdigits=4))\n")
    end
end

f1 = Figure(size = (1000, 600))
ax1 = Axis(f1[1,1],
    xlabel = "Fourier mode k",
    ylabel = "Convergence factor  ρ = ‖e₁‖ / ‖e₀‖",
    title  = "Two-grid convergence factor per mode  (mesh=$mesh_size, 1 V-cycle, 5 pre/post smooths)",
    yscale = log10,
)
for j in eachindex(schemes)
    scatterlines!(ax1, hf_modes, conv_vs_mode[j, :],
        label  = tags[j],
        marker = markers[j],
    )
end
hlines!(ax1, [1.0], linestyle = :dash, color = :red, label = "ρ = 1  (no reduction)")
axislegend(ax1, position = :rt)
save("twogrid_conv_vs_mode.png", f1)
println("Saved: twogrid_conv_vs_mode.png")

# ──────────────────────────────────────────────────────────────────────────────
# Analysis 2: worst-case mode — convergence factor vs V-cycle number
# ──────────────────────────────────────────────────────────────────────────────
worst_idx  = argmax(vec(conv_vs_mode[1, :]))
worst_mode = hf_modes[worst_idx]
println("\n=== Worst-case mode: k=$worst_mode  ρ=$(round(conv_vs_mode[1,worst_idx], sigdigits=4)) ===")
println("Computing convergence vs cycle number for k=$worst_mode...")

conv_vs_cycle = zeros(length(schemes), n_cycles)

for j in eachindex(schemes)
    cpc = test_twogrid(worst_mode, n_cycles, mesh_size, schemes[j], solvers[j])
    conv_vs_cycle[j, :] .= cpc
end

f2 = Figure(size = (1000, 600))
ax2 = Axis(f2[1,1],
    xlabel = "V-cycle number",
    ylabel = "Per-cycle convergence factor  ‖eₙ‖ / ‖eₙ₋₁‖",
    title  = "Two-grid convergence per cycle  (worst-case mode k=$worst_mode, mesh=$mesh_size)",
    yscale = log10,
)
for j in eachindex(schemes)
    scatterlines!(ax2, 1:n_cycles, conv_vs_cycle[j, :],
        label  = tags[j],
        marker = markers[j],
    )
end
hlines!(ax2, [1.0], linestyle = :dash, color = :red, label = "ρ = 1")
axislegend(ax2, position = :rb)
save("twogrid_conv_vs_cycle.png", f2)
println("Saved: twogrid_conv_vs_cycle.png")

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
println("\n=== Summary (mesh=$mesh_size) ===")
for j in eachindex(schemes)
    ρ_max = maximum(conv_vs_mode[j, :])
    ρ_avg = mean(conv_vs_mode[j, :])
    println("  $(tags[j]):  spectral radius ρ_max = $(round(ρ_max, sigdigits=4))  (mean = $(round(ρ_avg, sigdigits=4)))")
end
