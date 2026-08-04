using CairoMakie
using NavierStokes_Parallel 
using LaTeXStrings

NS = NavierStokes_Parallel

function compute_def_err(file_path, t_start,t_len)
    
    parts = split(dirname(file_path), "_")
    Nx,Ny,Nz = parse.(Int, parts[end-2:end])

    VF_init = zeros(Float32, Nx, Ny, Nz)
    VF_t    = similar(VF_init)

    Eshape_vec = zeros(length(t_start:2:t_len))

    NS.fillArray!(VF_init, 0, file_path)

    for (idx, t) in enumerate(t_start:2:t_len)
        NS.fillArray!(VF_t, Float16(t), file_path)

        # Eshape_vec[idx] = sqrt(sum((VF_t .- VF_init).^2)) / (Nx*Ny)
        Eshape_vec[idx] = sum(abs.(VF_t .- VF_init)) / (Nx*Ny)
    end

    return Eshape_vec
end

# -------------------------
# USER OPTIONS
# -------------------------
average = false
plot_all_periods = true

# -------------------------
# Zalesak Data
# -------------------------
# SL_pvd_48  = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_result_semi-lagrangian_48_48_1/Solver.pvd"
# SL_pvd_64  = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_result_semi-lagrangian_64_64_1/Solver.pvd"
# SL_pvd_96  = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_result_semi-lagrangian_96_96_1/Solver.pvd"
# SL_pvd_128 = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_result_semi-lagrangian_128_128_1/Solver.pvd"
# FD_pvd_48  = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_result_finite-difference_48_48_1/Solver.pvd"
# FD_pvd_64  = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_result_finite-difference_64_64_1/Solver.pvd"
# FD_pvd_96  = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_result_finite-difference_96_96_1/Solver.pvd"
# FD_pvd_128 = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_result_finite-difference_128_128_1/Solver.pvd"
# noFC_pvd_48 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_no_FC_finite-difference_48_48_1/Solver.pvd"
# noFC_pvd_64 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_no_FC_finite-difference_64_64_1/Solver.pvd"
# noFC_pvd_96 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_no_FC_finite-difference_96_96_1/Solver.pvd"
# noFC_pvd_128 = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Zalesak_no_FC_finite-difference_128_128_1/Solver.pvd"

# -------------------------
# Deformation Data
# -------------------------
SL_pvd_48 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_2_midpt2_semi-lagrangian_48_48_1/Solver.pvd"
SL_pvd_64 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_result_semi-lagrangian_64_64_1/Solver.pvd"
SL_pvd_96 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_result_semi-lagrangian_96_96_1/Solver.pvd"
SL_pvd_128 = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_result_semi-lagrangian_128_128_1/Solver.pvd"
FD_pvd_48 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_2_midpt2_finite-difference_48_48_1/Solver.pvd"
FD_pvd_64 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_2_midpt2_finite-difference_64_64_1/Solver.pvd"
FD_pvd_96 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_result_finite-difference_96_96_1/Solver.pvd"
FD_pvd_128 = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_result_finite-difference_128_128_1/Solver.pvd"
noFC_pvd_48 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_no_FC_finite-difference_48_48_1/Solver.pvd"
noFC_pvd_64 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_no_FC_finite-difference_64_64_1/Solver.pvd"
noFC_pvd_96 =  "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_no_FC_finite-difference_96_96_1/Solver.pvd"
noFC_pvd_128 = "/Users/julia/repo/NS_Multiphase/paper_1_result/VTK_Deformation_no_FC_finite-difference_128_128_1/Solver.pvd"

data = Dict(
    "FD" => Dict(
        48  => FD_pvd_48,
        64  => FD_pvd_64,
        # 32  => pvd_32,
        96  => FD_pvd_96,
        128 => FD_pvd_128
    ),
    "SL" => Dict(
        48  => SL_pvd_48,
        64  => SL_pvd_64,
        96  => SL_pvd_96,
        128 => SL_pvd_128
    ),
    "noFC" => Dict(
        48  => noFC_pvd_48,
        64  => noFC_pvd_64,
        96  => noFC_pvd_96,
        128 => noFC_pvd_128
    )

)

marker_map = Dict(
    "FD" => :diamond,
    "SL" => :circle,
    "noFC" => :rect
)
linestyle_map = Dict(
    "FD" => :solid,
    "SL" => :dash,
    "noFC" => :dashdot
)
# data = Dict(
#     "FD" => Dict(
#         48 => pvd_48,
#         32 => pvd_32
#     )
# )

function compute_all_errors(data, t_start,t_len)

    results = Dict()

    for (scheme, res_dict) in data
        println("Processing scheme: $scheme")
        Nx_vals = sort(collect(keys(res_dict)))
        n_res = length(Nx_vals)

        # Compute one to get n_periods
        E_sample = compute_def_err(res_dict[Nx_vals[1]], t_start, t_len)
        n_periods = length(E_sample)
        
        err = zeros(n_periods, n_res)

        for (j, Nx) in enumerate(Nx_vals)
            E = compute_def_err(res_dict[Nx], t_start, t_len)
            err[:, j] .= E
        end
        println("Scheme: $scheme, Nx: $Nx_vals, Errors: $err")
        err_mean = NS.mean(err, dims=1) |> vec

        results[scheme] = (
            Nx = Nx_vals,
            err = err,
            err_mean = err_mean
        )
    end

    return results
end


function plot_convergence(results;
    save_fig = false,
    save_path = "test_convergence.png"
)

    f = Figure(size = (700,500))
    ax = Axis(f[1,1],
        # title = "2D deformation test case Eₛ",
        ylabel = "L₁ error",
        xlabel = "Mesh Size",
        # xlabel = "CFL",
        xscale = log10,
        yscale = log10,
        xticks = (mesh_sizes),
        # xticks = (cfl_list),

        # titlesize = 30,
        xlabelsize = 34,
        ylabelsize = 34,
        xticklabelsize = 24,
        yticklabelsize = 24
    )

    for (scheme, res) in results
        if scheme =="FD"
            scheme_label = "SL-FV w/ FC"
        elseif scheme == "SL"
            scheme_label = "SL-SL"
        elseif scheme == "noFC"
            scheme_label = "SL_FV w/o FC"
        end
        Nx = res.Nx
        err = res.err
        n_periods = size(err, 1)

        # -------------------------
        # PERIOD CURVE
        # -------------------------
        for p in 1:n_periods
            scatterlines!(ax, Nx, err[p,:],linestyle = get(linestyle_map, scheme, :solid),marker = get(marker_map, scheme, :utriangle),label = "$scheme_label")
        end

        # -------------------------
        # REFERENCE LINES
        # -------------------------
        if scheme == "SL"
            x_ref = Nx[1]
            y_ref = err[1,1]   # first period as anchor

            ref1 = (y_ref+1e-4) .* (Nx ./ x_ref).^(-1)
            ref2 = (y_ref-1e-4) .* (Nx ./ x_ref).^(-2) 

            lines!(ax, Nx, ref1, linestyle=:dash, label = "O(Δx)",linewidth = 2)
            lines!(ax, Nx, ref2, linestyle=:dot,  label = "O(Δx²)",linewidth = 2)
        # elseif scheme == "noFC"
        #     x_ref = Nx[1]
        #     y_ref = err[1,1]   # first period as anchor

        #     ref1 = (y_ref+1e-2) .* (Nx ./ x_ref).^(-1)
        
        #     lines!(ax, Nx, ref1, linestyle=:dash, label = "O(Δx)",linewidth = 2)
        end
    end

    axislegend(ax, position = :lb,labelsize = 16)

    # -------------------------
    # SAVE FIGURE
    # -------------------------
    if save_fig
        save(save_path, f)
        println("Figure saved to: $save_path")
    end

    return f
end

mesh_sizes = [48, 64, 96, 128]
# cfl_list = [0.25, 0.5, 0.75, 1.0,1.5]
# cfl_list = [0.5, 1.0]
cfl_list = [0.75]
t_len = 2.0
t_start = 2.0
results = compute_all_errors(data, t_start, t_len)

f = plot_convergence(results;
    save_fig = true,
    save_path = "def_convergence.png"
)

f