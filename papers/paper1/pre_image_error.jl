using CairoMakie
using DataFrames
using CSV
using LaTeXStrings

FD_euler = "/Users/julia/repo/NS_Multiphase/paper_1_result/finite-difference_Euler_2D_errors_test2.csv"
FD_heun  = "/Users/julia/repo/NS_Multiphase/paper_1_result/finite-difference_Heun_2D_errors_test2.csv"
FD_rk4   = "/Users/julia/repo/NS_Multiphase/paper_1_result/finite-difference_RK4_2D_errors_test2.csv"
SL_euler = "/Users/julia/repo/NS_Multiphase/paper_1_result/semi-lagrangian_Euler_2D_errors_test2.csv"
SL_heun  = "/Users/julia/repo/NS_Multiphase/paper_1_result/semi-lagrangian_Heun_2D_errors_test2.csv"
SL_rk4   = "/Users/julia/repo/NS_Multiphase/paper_1_result/semi-lagrangian_RK4_2D_errors_test2.csv"

data = Dict(
    "SL-FV" => Dict(
        "1st" => FD_euler,
        "2nd" => FD_heun,
        "4th" => FD_rk4
    ),
    "SL-SL" => Dict(
        "1st" => SL_euler,
        "2nd" => SL_heun,
        "4th" => SL_rk4
    )
)

"""
    deformation_cfl_factor(N, L; t=0.0)

Maximum CFL coefficient (per unit Δt) of the 2D deformation flow field on an
N×N mesh over a domain of size L, evaluated at the first time step (t=0).
The field matches `example4_deformation.jl`:
    u = -2 sin²(πx) sin(πy) cos(πy) cos(πt/8)
    v = +2 sin²(πy) sin(πx) cos(πx) cos(πt/8)
Returns the factor `k` such that  CFL(Δt) = k * Δt.

Definition: CFL = Δt * maxvel / dx, where `maxvel` is the maximum velocity
*magnitude* |u| = sqrt(u² + v² + w²) over the field (not just the largest
single component).
"""
function deformation_cfl_factor(N::Int, L::Real; t::Real = 0.0)
    dx = L / N
    u_fun(x, y) = -2 * sin(π*x)^2 * sin(π*y) * cos(π*y) * cos(π*t/2.0)
    v_fun(x, y) = +2 * sin(π*y)^2 * sin(π*x) * cos(π*x) * cos(π*t/2.0)

    maxvel = 0.0
    for j in 1:N, i in 1:N
        xm = (i - 0.5) * dx        # cell-center coordinates
        ym = (j - 0.5) * dx
        mag = sqrt(u_fun(xm, ym)^2 + v_fun(xm, ym)^2)
        maxvel = max(maxvel, mag)
    end
    return maxvel / dx
end

"""
    plot_convergence(data; N=32, L=1.0, xtick_mode=:data, nticks=6)

`xtick_mode` selects how the CFL x-axis ticks are placed:
- `:data`    : ticks sit at each Δt data value, labeled with its CFL (log x-scale).
- `:uniform` : ticks are uniformly spaced in CFL from 0 to the max CFL of the
               input (linear x-scale). `nticks` sets how many.

The data is always plotted in Δt coordinates; only the tick placement/scale change.
"""
function plot_convergence(data; N = 32, L = 1.0, xtick_mode::Symbol = :data, nticks::Int = 6)
    # --- CFL conversion for the 2D, 1/32 mesh at the first time step ---
    cfl_factor = deformation_cfl_factor(N, L; t = 0.0)
    @info "CFL factor (per unit Δt) for $(N)×$(N) mesh, dx = $(L/N)" cfl_factor

    # Fixed plotting order so colors/markers/legend are deterministic
    schemes = ["SL-FV", "SL-SL"]
    interps = ["1st", "2nd", "4th"]

    # Color grouped by order (Okabe-Ito colorblind-friendly palette); marker grouped by scheme
    colors = Dict("1st" => "#0072B2", "2nd" => "#E69F00", "4th" => "#009E73")
    markers = Dict("SL-FV" => :circle, "SL-SL" => :utriangle)
    linestyle = :dot

    # Read everything first so we can build the CFL ticks from the Δt values
    dfs = Dict((s,it) => CSV.read(data[s][it], DataFrame) for s in schemes for it in interps)

    dt_vals = sort(unique(reduce(vcat, [df.dts for df in values(dfs)])))

    # Build tick positions (in Δt coords) + CFL labels, per mode
    if xtick_mode == :data
        # ticks at each Δt value, labeled with its CFL
        tick_pos    = dt_vals
        tick_labels = [string(round(cfl_factor * dt, digits = 2)) for dt in dt_vals]
        xscale      = log10
    elseif xtick_mode == :uniform
        # ticks uniform in CFL from 0 to 1 at 0.2 increments (needs linear scale)
        cfl_vals    = 0.0:0.2:1.2
        tick_pos    = collect(cfl_vals) ./ cfl_factor          # back to Δt coords
        tick_labels = [string(round(c, digits = 1)) for c in cfl_vals]
        xscale      = identity
    else
        error("xtick_mode must be :data or :uniform, got :$xtick_mode")
    end

    for dt in dt_vals
        println("Δt = $dt  ->  CFL = $(round(cfl_factor * dt, digits = 3))")
    end

    f = Figure(size = (700, 500))
    ax = Axis(
        f[1,1],
        yscale = log10,
        xscale = xscale,
        xlabel = L"CFL",
        # ylabel = "L₂ error",
        ylabel = L"E_{\text{pre-image}}",
        # title  = "Pre-image error 2D deformation flow",
        xticks = (tick_pos, tick_labels),   # positions in Δt, labels show CFL
        xticklabelrotation = π/4,
        xlabelsize = 28,
        ylabelsize = 32,
        xticklabelsize = 22,
        yticklabelsize = 28
    )

    for s in schemes, it in interps
        df = dfs[(s,it)]
        scatterlines!(ax, df.dts, df.errors;
            label      = "$(s)-$(it)",
            marker     = markers[s],
            color      = colors[it],
            linestyle  = linestyle,
            markersize = 12,
        )
    end

    axislegend(ax, position = :rb,labelsize = 20)
    save("pre_image_errors_$(xtick_mode)2.png", f)
    return f
end

# Flip xtick_mode between :data and :uniform to compare the two tick styles
# plot_convergence(data; xtick_mode = :data)
plot_convergence(data; xtick_mode = :uniform)
