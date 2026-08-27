#=
Builds the paper figures from the CSVs in data/.

Reads only CSVs — no VTK, no solver — so it is fast and can be run on any
machine that has the repo, without the Zenodo archive.

    julia --project=. papers/paper1/make_plots.jl                    # every figure
    julia --project=. papers/paper1/make_plots.jl deformation        # just one
    julia --project=. papers/paper1/make_plots.jl mms preimage       # a subset

or from the REPL:

    include("papers/paper1/make_plots.jl")
    make_plots()                 # all
    make_plots("deformation")    # one
=#

using CairoMakie
using LaTeXStrings
using CSV
using DataFrames

include(joinpath(@__DIR__, "common.jl"))

# ---------------------------------------------------------------------------
# Legend labels
#
# Keyed by the `variant` / `projection` values as they appear in the CSVs, so
# this script needs nothing from the case scripts or the solver.
# ---------------------------------------------------------------------------

const VARIANT_LABEL = Dict(
    "SL"   => "SL-SL",
    "FD"   => "SL-FV w/ FC",
    "noFC" => "SL-FV w/o FC",
)

# projection method → order of accuracy, which is what the legend should show
const PROJ_ORDER = Dict(
    "Euler" => "1st",
    "Heun"  => "2nd",
    "RK4"   => "4th",
)

# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

"Read a CSV from data/, or return `nothing` if that sweep hasn't been run yet."
function load(name)
    path = joinpath(DATA, "$(name)_errors.csv")
    isfile(path) || (@warn "no data for '$name' — skipping (expected $path)"; return nothing)
    return CSV.read(path, DataFrame)
end

"Shape-error convergence for an advection case (`deformation` or `zalesak`)."
function plot_advection(case)
    df = load(case)
    df === nothing && return nothing

    convergence_plot(df;
        x = :N, y = :E_shape, group = :variant,
        xlabel     = L"Mesh\ Size",
        ylabel     = L"E_{\text{shape}}",
        labels     = VARIANT_LABEL,
        xticks     = sort(unique(df.N)),
        refgroup   = "SL",
        refoffsets = [1.5, 0.7],
        savepath   = joinpath(FIGS, "$(case)_convergence.png"))
end

"L2 convergence of the variable-density pressure MMS."
function plot_mms()
    df = load("mms")
    df === nothing && return nothing

    convergence_plot(df;
        x = :N, y = :L2, group = :variant,
        xlabel     = L"Mesh\ Size",
        ylabel     = L"L_2\ error",
        labels     = VARIANT_LABEL,
        xticks     = sort(unique(df.N)),
        refoffsets = [1.25, 0.75],  
        legendpos  = :rt,
        savepath   = joinpath(FIGS, "mms_convergence.png"))
end

"Pre-image error vs CFL, one curve per variant × projection order."
function plot_preimage()
    df = load("preimage")
    df === nothing && return nothing

    labels = Dict((v, m) => "$(VARIANT_LABEL[v])-$(PROJ_ORDER[m])"
                  for v in unique(df.variant), m in unique(df.projection))

    convergence_plot(df;
        x = :cfl, y = :error, group = [:variant, :projection],
        xlabel    = L"CFL",
        ylabel    = L"E_{\text{pre-image}}",
        labels    = labels,
        savepath  = joinpath(FIGS, "preimage_error.png"))
end

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

const FIGURES = Dict(
    "deformation" => () -> plot_advection("deformation"),
    "zalesak"     => () -> plot_advection("zalesak"),
    "mms"         => plot_mms,
    "preimage"    => plot_preimage,
)

"""
    make_plots(which...)

Build the named figures, or all of them if called with no arguments.
Sweeps whose CSV doesn't exist yet are skipped with a warning rather than
erroring, so `make_plots()` works before every case has been run.
"""
function make_plots(which...)
    names = isempty(which) || "all" in which ? sort(collect(keys(FIGURES))) : collect(which)

    for name in names
        haskey(FIGURES, name) ||
            error("unknown figure '$name'. options: $(join(sort(collect(keys(FIGURES))), ", ")), all")
    end

    mkpath(FIGS)
    return Dict(name => FIGURES[name]() for name in names)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_plots(ARGS...)
end
