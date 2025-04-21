using JSON

# Load the JSON dictionary
jacobians = JSON.parsefile("jacob_comp_dict_24tets.json")

# Helper function to group Jacobian values by x-location
function group_by_x(data)
    grouped = Dict{String, Float64}()
    for (index, value) in data
        grouped[index] = value
    end
    return grouped
end

# Grab data corresponding to each cell
group_12 = group_by_x(jacobians["12,11,12"])
group_14 = group_by_x(jacobians["12,15,12"])

# Define the symmetry relationship
halfway = 13

# Prepare the table header
println(rpad("12,11,12", 10), rpad(" J_val", 30), rpad("12,15,12", 10), rpad(" J_val", 30), "matches")
println("-"^90)

function match_counter(group_1,group_2)
    match_count = 0

    # Compare and print each row
    for x_12_s in keys(group_1)
        x_12_x,x_12_y,x_12_z = parse.(Int,split(x_12_s,","))
        x_14 = halfway + (halfway - x_12_y)  # Symmetry pair
        value_12 = group_1[x_12_s]
        value_14 = get(group_2, "$x_12_x,$x_14,$x_12_z", NaN)  # Handle missing values gracefully
        matches = !isnan(value_14) && isapprox(value_12, value_14; atol=1e-5)
        if matches
            match_count+=1
        end
        # Print each row
        println(rpad( "$x_12_x,$x_12_y,$x_12_z", 10), 
                rpad(string(value_12), 30), 
                rpad( "$x_12_x,$x_14,$x_12_z", 10), 
                rpad(string(value_14), 30), 
                matches)    
    end
    
    println("Found $match_count matches")
end

match_counter(group_12,group_14)
