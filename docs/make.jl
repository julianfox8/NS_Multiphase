using Documenter, NavierStokes_Parallel 

makedocs(; sitename = "Mist.jl",
    #modules = [NavierStokes_Parallel],
    #doctest = true,
    authors = "Julian Fox and Mark Owkes",
    pages = Any[
        "Home" => "index.md",
        "Examples" => [
        "Overview" => "examples/index.md",
        ],
        "Numerics" => "numerics/numerics.md",
        "Verification" => [
            "Overview" => "verification/index.md",
            "Advection" => "verification/advection.md",
            "Zalesak Disk" => "verification/zalesak.md",
            "2D Deformation" => "verification/2d_deformation.md",
            "Pressure MMS" => "verification/mms.md",
            "Pre-image error" => "verification/preimage.md",]
    ],

) 

deploydocs(; repo ="https://github.com/julianfox8/NS_Multiphase", push_preview = false)