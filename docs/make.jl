using Documenter, NavierStokes_Parallel 

makedocs(; sitename = "Mist.jl",
    #modules = [NavierStokes_Parallel],
    #doctest = true,
    authors = "Julian Fox and Mark Owkes",
    pages = Any[
        "Home" => "index.md",
        "Examples" => [
        "Overview" => "examples/index.md",
        "2D Deformation" => "examples/2d_deformation.md"],
        "Numerics" => "numerics/numerics.md"
        
    ],

) 

deploydocs(; repo ="https://github.com/julianfox8/NS_Multiphase", push_preview = false)