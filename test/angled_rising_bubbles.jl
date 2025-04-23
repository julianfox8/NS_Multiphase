
#! need to modify origin and output file
function write_input_file(param_file,rad)

    # identify lines to change
    new_file = String[]
    lines = readlines(param_file)
    for line ∈ lines
        if occursin("grav_x = ",line)
            push!(new_file,"    grav_x = $(round(9.8*cos(rad),digits = 4)),")
        elseif occursin("grav_y = ",line)
            push!(new_file,"    grav_y = $(round(9.8*sin(rad),digits = 4)),")
        elseif occursin("test_case = ",line)
            angle = replace(string(rad2deg(rad)),"." => "_")
            push!(new_file,"    test_case = \"viscous_bubble_rise_$(angle)\", ")
        else
            push!(new_file,line)
        end
    end

    #re-write entire file
    open(param_file,"w") do file
        for line ∈ new_file
            println(file,line)
        end
    end
    return nothing 
end

function run_sims(param_file)
    for i ∈ 1:10
        dθ = 89/10
        θ = i*dθ
        rad = deg2rad(θ)
        write_input_file(param_file,rad)
            # run simulation
        run(`mpiexecjl --project=. -n 1 julia $param_file`)
    end
end
param_file = "examples/example_viscous_bubble3D_coarse.jl"
run_sims(param_file)