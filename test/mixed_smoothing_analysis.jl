"""
Mixed-mode convergence speedup test.

Injects a superposition of a low-frequency (smooth) and high-frequency (oscillatory)
Fourier mode as the initial error, then compares:
  - Single-level smoother only: decays HF component fast, stalls on LF component
  - Two-grid V-cycle: decays both components

The asymptotic convergence factor of each method is compared to confirm the
speedup predicted by the smoothing factor analysis.
"""

using NavierStokes_Parallel
using CairoMakie
using LinearAlgebra
using Statistics
using Printf

NS = NavierStokes_Parallel

# ──────────────────────────────────────────────────────────────────────────────
# Shared: mixed-mode injection
# ──────────────────────────────────────────────────────────────────────────────

"""
Inject P = sin(2π·lf·x)sin(2π·lf·y) + sin(2π·hf·x)sin(2π·hf·y)
as the initial error.  RHS=0 everywhere so exact P*=0 and error = P.
"""
function set_mixed_modes!(lf_mode, hf_mode, P, mesh, par_env)
    @unpack imin_, imax_, jmin_, jmax_, xm, ym = mesh
    kz = 1
    for i = imin_:imax_, j = jmin_:jmax_
        lf = sin(2π * lf_mode * xm[i]) * sin(2π * lf_mode * ym[j])
        hf = sin(2π * hf_mode * xm[i]) * sin(2π * hf_mode * ym[j])
        P[i, j, kz] = lf + hf
    end
    NS.update_borders!(P, mesh, par_env)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# Single-level: track ‖P‖₂ after each smoother sweep
# ──────────────────────────────────────────────────────────────────────────────

"""
    test_mixed_singlelevel(lf_mode, hf_mode, n_sweeps, Nx, scheme, solver)

Returns a vector of length n_sweeps+1 containing ‖P‖₂ at sweep 0,1,...,n_sweeps.
"""
function test_mixed_singlelevel(lf_mode, hf_mode, n_sweeps, Nx, scheme, solver)
    param = parameters(
        mu_liq=0.0, mu_gas=0.0,
        rho_liq=1.0, rho_gas=1.0,
        sigma=0.0,
        grav_x=0.0, grav_y=0.0, grav_z=0.0,
        Lx=1.0, Ly=1.0, Lz=1.0/100,
        tFinal=100.0,
        Nx=Nx, Ny=Nx, Nz=1,
        stepMax=50,
        max_dt=2.5e-4,
        CFL=1.0,
        std_out_period=0.0,
        out_period=1,
        tol=1e-14,   # keep small so smoother never exits early
        nprocx=1, nprocy=1, nprocz=1,
        xper=false, yper=false, zper=true,
        pressure_scheme=scheme,
        pressureSolver=solver,
        hypreSolver="GMRES-AMG",
        mg_lvl=1,
        projection_method="Heun",
        tesselation="5_tets",
        iter_type="standard",
        test_case="psolve_test",
    )

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
    fill!(VF, 0.0)
    fill!(band, 1.0)
    NS.compute_props!(denx, deny, denz, viscx, viscy, viscz, VF, param, mesh)

    fill!(P, 0.0)
    set_mixed_modes!(lf_mode, hf_mode, P, mesh, par_env)
    NS.Neumann!(P, mesh, par_env)

    norms = zeros(n_sweeps + 1)
    norms[1] = norm(P[imin_:imax_, jmin_:jmax_, kmin_:kmax_])

    for s in 1:n_sweeps
        if scheme == "finite-difference"
            if solver == "gauss-seidel"
                NS.gs(P, RHS, tmp2, denx, deny, denz, dt, param, mesh, par_env; max_iter=1)
            elseif solver == "jacobi"
                NS.jacobi(P, tmp6, RHS, tmp2, denx, deny, denz, dt, param, mesh, par_env; max_iter=1)
            end
        elseif scheme == "semi-lagrangian"
            if solver == "res_iteration"
                NS.res_iteration(P, uf, vf, wf, gradx, grady, gradz, band, dt, denx, deny, denz,
                                 tmp6, tmp2, tmp7, tmp4, verts, tets, param, mesh, par_env; max_iter=1)
            end
        end
        NS.Neumann!(P, mesh, par_env)
        norms[s+1] = norm(P[imin_:imax_, jmin_:jmax_, kmin_:kmax_])
    end

    return norms
end

# ──────────────────────────────────────────────────────────────────────────────
# Two-level: track ‖P‖₂ after each V-cycle
# ──────────────────────────────────────────────────────────────────────────────

"""
    test_mixed_twogrid(lf_mode, hf_mode, n_cycles, Nx, scheme, solver)

Returns a vector of length n_cycles+1 containing ‖P‖₂ at cycle 0,1,...,n_cycles.
"""
function test_mixed_twogrid(lf_mode, hf_mode, n_cycles, Nx, scheme, solver)
    param = parameters(
        mu_liq=0.0, mu_gas=0.0,
        rho_liq=1.0, rho_gas=1.0,
        sigma=0.0,
        grav_x=0.0, grav_y=0.0, grav_z=0.0,
        Lx=1.0, Ly=1.0, Lz=1.0/100,
        tFinal=100.0,
        Nx=Nx, Ny=Nx, Nz=1,
        stepMax=50,
        max_dt=2.5e-4,
        CFL=1.0,
        std_out_period=0.0,
        out_period=1,
        tol=1e-14,
        nprocx=1, nprocy=1, nprocz=1,
        xper=false, yper=false, zper=true,
        pressure_scheme=scheme,
        pressureSolver=solver,
        hypreSolver="GMRES-AMG",
        mg_lvl=2,
        projection_method="Heun",
        tesselation="5_tets",
        iter_type="standard",
        test_case="psolve_test",
    )

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
    fill!(VF, 0.0)
    fill!(band, 1.0)
    NS.compute_props!(denx, deny, denz, viscx, viscy, viscz, VF, param, mesh)

    fill!(P, 0.0)
    set_mixed_modes!(lf_mode, hf_mode, P, mesh, par_env)
    NS.Neumann!(P, mesh, par_env)

    fields = (P=P, uf=uf, vf=vf, wf=wf,
              denx=denx, deny=deny, denz=denz,
              gradx=gradx, grady=grady, gradz=gradz,
              band=band)
    NS.copy_to_mg!(mg_arrays, fields, 1)

    norms = zeros(n_cycles + 1)
    norms[1] = norm(mg_arrays[1].P_h[imin_:imax_, jmin_:jmax_, kmin_:kmax_])

    for c in 1:n_cycles
        if scheme == "finite-difference"
            NS.mg_fas_lin!(1, mg_arrays, mg_mesh, dt, VF, verts, tets,
                           nothing, param, par_env, 0; iter=nothing)
        elseif scheme == "semi-lagrangian"
            NS.mg_fas!(1, mg_arrays, mg_mesh, dt, VF, verts, tets,
                       nothing, param, par_env, 0; iter=nothing)
        end
        norms[c+1] = norm(mg_arrays[1].P_h[imin_:imax_, jmin_:jmax_, kmin_:kmax_])
    end

    return norms
end

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

mesh_size = 48
n_cycles    = 20             # sweeps (single-level) / V-cycles (two-level)

lf_mode   = 2               # smooth component — smoother barely touches this
hf_mode   = mesh_size ÷ 3 + 1   # oscillatory component — well inside high-freq band

tags    = ["GS-FD", "WJ-SL"]
schemes = ["finite-difference", "semi-lagrangian"]
solvers = ["gauss-seidel", "res_iteration"]
markers = [:circle, :pentagon, :diamond, :dtriangle]

println("=== Mixed-mode analysis: lf=$lf_mode  hf=$hf_mode  mesh=$mesh_size ===\n")

# ──────────────────────────────────────────────────────────────────────────────
# Run tests
# ──────────────────────────────────────────────────────────────────────────────

sl_norms = [test_mixed_singlelevel(lf_mode, hf_mode, n_cycles, mesh_size, schemes[j], solvers[j])
            for j in eachindex(schemes)]

tg_norms = [test_mixed_twogrid(lf_mode, hf_mode, n_cycles, mesh_size, schemes[j], solvers[j])
            for j in eachindex(schemes)]

# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: ‖e‖₂ vs iteration (normalized to e₀)
# Expected: single-level plateaus early (LF mode stalls); two-grid keeps falling
# ──────────────────────────────────────────────────────────────────────────────

f1 = Figure(size=(1100, 650))
ax1 = Axis(f1[1,1],
    xlabel = "Iteration / V-cycle",
    ylabel = "‖e‖₂ / ‖e₀‖₂",
    title  = "Mixed-mode convergence: single-level vs two-grid  (lf=$lf_mode, hf=$hf_mode, mesh=$mesh_size)",
    yscale = log10,
)

for j in eachindex(schemes)
    e0 = sl_norms[j][1]
    scatterlines!(ax1, 0:n_cycles, sl_norms[j] ./ e0;
        label     = "$(tags[j]) — single-level",
        marker    = markers[j],
        linestyle = :dash,
    )
    scatterlines!(ax1, 0:n_cycles, tg_norms[j] ./ e0;
        label     = "$(tags[j]) — two-grid",
        marker    = markers[j],
        linestyle = :solid,
    )
end

axislegend(ax1, position=:rc)
# display(f1)
save("mixed_mode_convergence.png", f1)
println("Saved: mixed_mode_convergence.png")

# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: per-step convergence factor ρ = ‖eₙ‖/‖eₙ₋₁‖
# Expected: single-level ρ → 1 as smooth mode dominates; two-grid ρ stays small
# ──────────────────────────────────────────────────────────────────────────────

f2 = Figure(size=(1100, 650))
ax2 = Axis(f2[1,1],
    xlabel = "Iteration / V-cycle",
    ylabel = "Per-step factor  ‖eₙ‖ / ‖eₙ₋₁‖",
    title  = "Per-step convergence factor  (lf=$lf_mode, hf=$hf_mode, mesh=$mesh_size)",
    yscale = log10,
)

for j in eachindex(schemes)
    sl_factors = sl_norms[j][2:end] ./ sl_norms[j][1:end-1]
    tg_factors = tg_norms[j][2:end] ./ tg_norms[j][1:end-1]
    scatterlines!(ax2, 1:n_iter, sl_factors;
        label     = "$(tags[j]) — single-level",
        marker    = markers[j],
        linestyle = :dash,
    )
    scatterlines!(ax2, 1:n_iter, tg_factors;
        label     = "$(tags[j]) — two-grid",
        marker    = markers[j],
        linestyle = :solid,
    )
end

hlines!(ax2, [1.0], linestyle=:dot, color=:red, label="ρ = 1  (no reduction)")
axislegend(ax2, position=:rb)
# display(f2)
save("mixed_mode_conv_factor.png", f2)
println("Saved: mixed_mode_conv_factor.png")

# ──────────────────────────────────────────────────────────────────────────────
# Summary: asymptotic rates (averaged over last 5 steps)
# ──────────────────────────────────────────────────────────────────────────────

# println("\n=== Raw error norms (mesh=$mesh_size, lf=$lf_mode, hf=$hf_mode) ===")
# for j in eachindex(schemes)
#     println("\n  $(tags[j]) — single-level ‖P‖₂ per sweep:")
#     for (s, n) in enumerate(sl_norms[j])
#         ρ = s == 1 ? "-" : @sprintf("%.4e", sl_norms[j][s] / sl_norms[j][s-1])
#         println("    sweep $(lpad(s-1,3)):  ‖P‖₂ = $(@sprintf("%.6e", n))   ρ = $ρ")
#     end
#     println("\n  $(tags[j]) — two-grid ‖P‖₂ per V-cycle:")
#     for (c, n) in enumerate(tg_norms[j])
#         ρ = c == 1 ? "-" : @sprintf("%.4e", tg_norms[j][c] / tg_norms[j][c-1])
#         println("    cycle $(lpad(c-1,3)):  ‖P‖₂ = $(@sprintf("%.6e", n))   ρ = $ρ")
#     end
# end

println("\n=== Asymptotic convergence factors (last 5 steps, mesh=$mesh_size) ===")
println("  lf_mode=$lf_mode  hf_mode=$hf_mode\n")

for j in eachindex(schemes)
    sl_asym = mean(sl_norms[j][end-4:end] ./ sl_norms[j][end-5:end-1])
    tg_asym = mean(tg_norms[j][end-4:end] ./ tg_norms[j][end-5:end-1])
    speedup = sl_asym / tg_asym
    println("  $(tags[j]):")
    println("    single-level ρ_asymp ≈ $(round(sl_asym, sigdigits=4))")
    println("    two-grid     ρ_asymp ≈ $(round(tg_asym, sigdigits=4))")
    println("    speedup             ≈ $(round(speedup, sigdigits=3))×")
end
