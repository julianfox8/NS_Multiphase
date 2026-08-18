using CairoMakie
using LaTeXStrings
using DataFrames

# ---------------------------------------------------------------------------
# Directory layout
#
# @__DIR__ resolves to papers/paper1/ regardless of which script includes this
# file or where julia was launched from, so output never depends on cwd.
#
#   results/  regenerable VTK scratch  — gitignored, archived to Zenodo (~12 GB)
#   data/     error CSVs               — committed, small; figures rebuild from these alone
#   figures/  final paper figures      — committed
#
# Callers are responsible for mkpath() before writing.
# ---------------------------------------------------------------------------

const PAPER_DIR = @__DIR__
const RESULTS   = joinpath(PAPER_DIR, "results")
const DATA      = joinpath(PAPER_DIR, "data")
const FIGS      = joinpath(PAPER_DIR, "figures")

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

"""
Shared Makie theme for every figure in the paper.

Apply per-script with `set_theme!(MIST_THEME)`, or scope it to one figure with
`with_theme(MIST_THEME) do ... end`. 
"""
const MIST_THEME = Theme(
    size = (1200, 1000),
    Axis = (
        xlabelsize     = 28,
        ylabelsize     = 32,
        xticklabelsize = 22,
        yticklabelsize = 28,
    ),
    Legend = (
        labelsize = 20,
    ),
    Lines = (
        linewidth = 2,
    ),
    ScatterLines = (
        linewidth  = 2,
        markersize = 12,
    ),
)

"""
    slope_label(p)

LaTeX label for a reference line of order `p`: `O(Δx)`, `O(Δx²)`, ...
"""
slope_label(p::Integer) = p == 1 ? L"O(\Delta x)" : latexstring("O(\\Delta x^{$p})")

"""
    add_refslopes!(ax, x, y0; orders = [1, 2], offsets = nothing)

Draw dashed reference-convergence lines on a log-log axis.

Each line passes through `(x[1], y0 * offset)` and falls off as `x^(-p)` for
each `p` in `orders`.

- `x`       : the x values the data is plotted against (mesh sizes, CFL, ...)
- `y0`      : anchor value, normally the first data point of the reference series
- `offsets` : per-order *multiplicative* nudges to separate the lines from the
              data.
"""
function add_refslopes!(ax, x, y0::Real;
                        orders  = [1, 2],
                        offsets = nothing,
                        color   = :gray)

    xs = collect(float.(x))
    isempty(xs) && return ax

    for (i, p) in enumerate(orders)
        off = offsets === nothing ? 1.0 : offsets[i]
        ys  = (y0 * off) .* (xs ./ xs[1]) .^ (-p)

        lines!(ax, xs, ys; linestyle = :dash, linewidth = 2, color,
               label = slope_label(p))
    end

    return ax
end

# ---------------------------------------------------------------------------
# Convergence plotting
# ---------------------------------------------------------------------------

"""
    convergence_plot(df; x, y, group, kwargs...)

Log-log convergence figure from a long-format DataFrame, one curve per group.

All four paper CSVs share this shape — a grouping column, an independent
variable, and an error column — so one function covers every figure:

    advection : x = :N,   y = :E_shape, group = :variant
    MMS       : x = :N,   y = :L2,      group = :variant
    pre-image : x = :cfl, y = :error,   group = [:variant, :projection]

Required
- `x`, `y`  : column names for the independent variable and the error
- `group`   : a Symbol, or a vector of Symbols for a multi-key grouping

Optional
- `xlabel`, `ylabel` : axis labels, default to the column names
- `labels`    : `Dict` mapping a group key to its legend label. Keys are the raw
                column value for a single grouping column (`"SL" => "SL-SL"`), or
                a Tuple for a multi-key grouping (`("SL","Heun") => "SL-SL, Heun"`).
                Unmapped keys fall back to the value itself, joined by `" / "`.
- `refslopes` : reference-line orders; `[]` to omit them entirely
- `refoffsets`: multiplicative nudges for those lines, one per order
- `refgroup`  : which group anchors the reference lines. Defaults to the first.
- `xticks`    : explicit tick positions, e.g. the mesh sizes
- `legendpos` : `axislegend` position
- `savepath`  : if given, save the figure there (parent directory is created)

Returns the `Figure`.
"""
function convergence_plot(df::DataFrame;
                          x::Symbol,
                          y::Symbol,
                          group,
                          xlabel     = string(x),
                          ylabel     = string(y),
                          labels     = Dict(),
                          refslopes  = [1, 2],
                          refoffsets = nothing,
                          refgroup   = nothing,
                          xticks     = nothing,
                          markers    = [:circle, :diamond, :rect, :utriangle, :star5, :cross],
                          linestyle  = :dot,
                          legendpos  = :lb,
                          savepath   = nothing)

    groupcols = group isa Symbol ? [group] : collect(group)
    gdf = groupby(df, groupcols; sort = true)   # sort => deterministic colour/legend order

    legend_label(key) = get(labels, length(key) == 1 ? key[1] : key,
                            join(string.(key), " / "))
    keyof(sub) = Tuple(first(sub)[c] for c in groupcols)

    fig = with_theme(MIST_THEME) do
        f  = Figure()
        ax = Axis(f[1, 1]; xlabel, ylabel, xscale = log10, yscale = log10)
        xticks === nothing || (ax.xticks = xticks)

        for (i, sub) in enumerate(gdf)
            s = sort(sub, x)
            scatterlines!(ax, s[!, x], s[!, y];
                          marker = markers[mod1(i, length(markers))],
                          linestyle,
                          label = legend_label(keyof(sub)))
        end

        if !isempty(refslopes)
            anchor = refgroup === nothing ? first(gdf) :
                     only(g for g in gdf if keyof(g) == (refgroup isa Tuple ? refgroup : (refgroup,)))
            s = sort(anchor, x)
            add_refslopes!(ax, s[!, x], first(s[!, y]);
                           orders = refslopes, offsets = refoffsets)
        end

        axislegend(ax; position = legendpos)
        f
    end

    if savepath !== nothing
        mkpath(dirname(savepath))
        save(savepath, fig)
        @info "saved $savepath"
    end

    return fig
end

