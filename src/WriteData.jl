
function VTK(P,mesh,mesh_par,par_env)
    @unpack x,y,z = mesh 

    vtk_grid("fields", x, y, z) do vtk
        vtk["temperature"] = rand(length(x), length(y), length(z))
    end


end
