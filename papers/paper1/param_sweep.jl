using CairoMakie
using NavierStokes_Parallel 
using LaTeXStrings
using CSV
using DataFrames

NS = NavierStokes_Parallel

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "advection_tests.jl"))


struct Variant
    tag::String; scheme::String; fc::Bool
end

# figure label — presentation only, derived from the scheme/fc combination
label(v::Variant) = v.scheme == "semi-lagrangian" ? "SL-SL" :
                    v.fc                          ? "SL-FV w/ FC" :
                                                    "SL-FV w/o FC"

const VARIANTS = [
    Variant("SL",   "semi-lagrangian",   false),   # fc ignored when scheme is SL
    Variant("FD",   "finite-difference", true),
    Variant("noFC", "finite-difference", false),
]

const CASES = Dict(
    "deformation" => (vf = "Deformation", tFinal = 2.0, Ns = [48,64,96,128]),
    "zalesak"     => (vf = "Zalesak",     tFinal = 1.0, Ns = [48,64,96,128]),
)

# ---------------------------------------------------------------------------
# Stage 1: Run all cases to generate VTK results
# ---------------------------------------------------------------------------

function run_cases(case, Ns = CASES[case].Ns)
    spec = CASES[case]
    for v in VARIANTS, n in Ns
        @info "running $(spec.vf) $(v.tag) N=$n"
        test_advection(n, v.scheme, v.fc, spec.vf, v.tag; tFinal = spec.tFinal)
    end
end

# ---------------------------------------------------------------------------
# Stage 2: Produce all csv files for error analysis
# ---------------------------------------------------------------------------
function compute_shape_err(file_path, N, t_len)

    VF_init = zeros(Float32, N, N, 1)
    VF_t    = similar(VF_init)

    NS.fillArray!(VF_t, t_len, file_path)
    NS.fillArray!(VF_init, 0, file_path)

    return sum(abs.(VF_t .- VF_init)) / (N*N) 
end

function compute_errors(case, Ns = CASES[case].Ns)
    spec = CASES[case]
    rows = DataFrame(variant = String[], N = Int[], E_shape = Float64[])

    for v in VARIANTS, n in Ns
        pvd = joinpath(RESULTS,
                       "VTK_$(spec.vf)_$(v.tag)_$(v.scheme)_$(n)_$(n)_$(1)",
                       "Solver.pvd")
        isfile(pvd) || (@warn "missing, skipping"; continue)
        push!(rows, (v.tag, n, compute_shape_err(pvd, n, spec.tFinal)))
    end

    mkpath(DATA)
    CSV.write(joinpath(DATA, "$(case)_errors.csv"), rows)
    return rows
end


if abspath(PROGRAM_FILE) == @__FILE__
    run_cases(ARGS[1])
    compute_errors(ARGS[1])
end