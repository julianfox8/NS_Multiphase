using CairoMakie
using NavierStokes_Parallel 

NS = NavierStokes_Parallel

function compute_def_err(file_path, t_len)
    
    parts = split(dirname(file_path), "_")
    Nx,Ny,Nz = parse.(Int, parts[end-2:end])

    VF_init = zeros(Float32, Nx, Ny, Nz)
    VF_t    = similar(VF_init)

    Eshape_vec = zeros(length(2:2:t_len))

    NS.fillArray!(VF_init, 0, file_path)

    for (idx, t) in enumerate(2:2:t_len)
        NS.fillArray!(VF_t, Float16(t), file_path)

        Eshape_vec[idx] = sqrt(sum((VF_t .- VF_init).^2) / (Nx*Ny*Nz))
    end

    return Eshape_vec
end

# -------------------------
# USER OPTIONS
# -------------------------
average = false
plot_all_periods = true

# -------------------------
# DATA
# -------------------------
# pvd_48 = "/Users/julia/repo/NS_Multiphase/results/VTK_Deformation2_finite-difference_48_48_1/Solver.pvd"
# pvd_64 = "/Users/julia/repo/NS_Multiphase/results/VTK_Deformation2_finite-difference_64_64_1/Solver.pvd"
SL_pvd_48 = " /home/g91k227/repo/NS_Multiphase/results/VTK_Deformation_40_semi-lagrangian_48_48_1/Solver.pvd"
SL_pvd_32 = " /home/g91k227/repo/NS_Multiphase/results/VTK_Deformation_40_semi-lagrangian_32_32_1/Solver.pvd"
SL_pvd_96 = " /home/g91k227/repo/NS_Multiphase/results/VTK_Deformation_40_semi-lagrangian_96_96_1/Solver.pvd"
SL_pvd_128 = " /home/g91k227/repo/NS_Multiphase/results/VTK_Deformation_40_semi-lagrangian_128_128_1/Solver.pvd"
FD_pvd_48 = " /home/g91k227/repo/NS_Multiphase/results/VTK_Deformation_40_finite-difference_48_48_1/Solver.pvd"
FD_pvd_32 = " /home/g91k227/repo/NS_Multiphase/results/VTK_Deformation_40_finite-difference_32_32_1/Solver.pvd"
FD_pvd_96 = " /home/g91k227/repo/NS_Multiphase/results/VTK_Deformation_40_finite-difference_96_96_1/Solver.pvd"
FD_pvd_128 = " /home/g91k227/repo/NS_Multiphase/results/VTK_Deformation_40_finite-difference_128_128_1/Solver.pvd"

data = Dict(
    "FD" => Dict(
        48  => FD_pvd_48,
        64  => FD_pvd_64,
        96  => FD_pvd_96,
        128 => FD_pvd_128
    ),
    "SL" => Dict(
        48  => SL_pvd_48,
        64  => SL_pvd_64,
        96  => SL_pvd_96,
        128 => SL_pvd_128
    )
)

# data = Dict(
#     "FD" => Dict(
#         48 => pvd_48,
#         64 => pvd_64
#     )
# )

function compute_all_errors(data, t_len)

    results = Dict()

    for (scheme, res_dict) in data

        Nx_vals = sort(collect(keys(res_dict)))
        n_res = length(Nx_vals)

        # Compute one to get n_periods
        E_sample = compute_def_err(res_dict[Nx_vals[1]], t_len)
        n_periods = length(E_sample)

        err = zeros(n_periods, n_res)

        for (j, Nx) in enumerate(Nx_vals)
            E = compute_def_err(res_dict[Nx], t_len)
            err[:, j] .= E
        end

        err_mean = NS.mean(err, dims=1) |> vec

        results[scheme] = (
            Nx = Nx_vals,
            err = err,
            err_mean = err_mean
        )
    end

    return results
end
# -------------------------
# AVERAGING
# -------------------------
err_mean = NS.mean(err, dims=1) |> vec
using CairoMakie
using Statistics

function plot_convergence(results;
    mode = :average,              # :average, :periods, :both
    save_fig = false,
    save_path = "convergence.png"
)

    f = Figure(size = (600,600))
    ax = Axis(f[1,1],
        title = "Deformation test case Eₛ",
        ylabel = "E_shape",
        xlabel = "Nx",
        xscale = log10,
        yscale = log10
    )

    for (scheme, res) in results

        Nx = res.Nx
        err = res.err
        err_mean = res.err_mean
        n_periods = size(err, 1)

        # -------------------------
        # PERIOD CURVES
        # -------------------------
        if mode == :periods || mode == :both
            for p in 1:n_periods
                lines!(ax, Nx, err[p, :],
                    label = "$scheme Period $p",
                    linewidth = 1
                )
            end
        end

        # -------------------------
        # AVERAGE CURVE
        # -------------------------
        if mode == :average || mode == :both
            lines!(ax, Nx, err_mean,
                linewidth = 3,
                label = "$scheme Avg"
            )
            scatter!(ax, Nx, err_mean)
        end

        # -------------------------
        # REFERENCE LINES
        # -------------------------
        x_ref = Nx[1]

        y_ref = if mode == :average
            err_mean[1]
        else
            err[1,1]   # first period as anchor
        end

        ref1 = y_ref .* (Nx ./ x_ref).^(-1)
        ref2 = y_ref .* (Nx ./ x_ref).^(-2)

        lines!(ax, Nx, ref1, linestyle=:dash, label = "$scheme O(Δx)")
        lines!(ax, Nx, ref2, linestyle=:dot,  label = "$scheme O(Δx²)")
    end

    axislegend(ax, position = :rb)

    # -------------------------
    # SAVE FIGURE
    # -------------------------
    if save_fig
        save(save_path, f)
        println("Figure saved to: $save_path")
    end

    return f
end


t_len = 20.0

results = compute_all_errors(data, t_len)

f = plot_convergence(results;
    mode = :average,
    save_fig = true,
    save_path = "avg_convergence.png"
)

f