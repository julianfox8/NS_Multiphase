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
pvd_48 = "/Users/julia/repo/NS_Multiphase/results/VTK_Deformation2_finite-difference_48_48_1/Solver.pvd"
pvd_64 = "/Users/julia/repo/NS_Multiphase/results/VTK_Deformation2_finite-difference_64_64_1/Solver.pvd"

t_len = 8.0

E_48 = compute_def_err(pvd_48, t_len)
E_64 = compute_def_err(pvd_64, t_len)

Nx = [48, 64]

n_periods = length(E_48)

err = zeros(n_periods, 2)
err[:,1] .= E_48
err[:,2] .= E_64

# -------------------------
# AVERAGING
# -------------------------
err_mean = NS.mean(err, dims=1) |> vec

# -------------------------
# PLOTTING
# -------------------------
f = Figure(size = (600,600))
ax = Axis(f[1,1],
    title = "Deformation test case Eₛ",
    ylabel = "E_shape",
    xlabel = "Nx",
    xscale = log10,
    yscale = log10
)

# Plot individual periods
if plot_all_periods && !average
    for p in 1:n_periods
        lines!(ax, Nx, err[p, :], label = "Period $p")
        scatter!(ax, Nx, err[p, :])
    end
end

# Plot averaged curve
if average
    lines!(ax, Nx, err_mean, linewidth=3, label="Average")
    scatter!(ax, Nx, err_mean)
end

# -------------------------
# REFERENCE LINES
# -------------------------
# Use first resolution as anchor
x_ref = Nx[1]
y_ref = average ? err_mean[1] : err[2,1]

ref1 = y_ref .* (Nx ./ x_ref).^(-1)
ref2 = y_ref .* (Nx ./ x_ref).^(-2)

lines!(ax, Nx, ref1, linestyle=:dash, label="O(Δx)")
lines!(ax, Nx, ref2, linestyle=:dash, label="O(Δx²)")

axislegend(ax, position = :rb)

f