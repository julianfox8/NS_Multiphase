include(joinpath(@__DIR__, "test_deformation.jl"))
include(joinpath(@__DIR__, "test_zalesak.jl"))

struct Variant
    tag::String; label::String; scheme::String; fc::Bool 
end

const VARIANTS = [
    Variant("SL", "SL-SL",       "semi-lagrangian",     false),
    Variant("FD", "SL-FV w/ FC", "finite-difference",   true),
    Variant("noFC", "SL-FV w/o FC", "finite-difference",false),  
]

const CASES = Dict(
    "deformation" => (runner = test_deformation, prefix = "Deformation", Ns = [48,64,96,128]),
    "zalesak"     => (runner = test_zalesak,     prefix = "Zalesak",     Ns = [48,64,96,128]),
)

function run_cases(case, Ns = CASES[case].Ns)
    spec = CASES[case]
    for v in VARIANTS, n in Ns
        @info "running $(v.tag) N=$n"
        spec.runner(n, v.scheme, v.fc,"$(spec.prefix)_$(v.tag)";)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_cases(ARGS)
end