using ReadVTK

function gather_restart_files(restart_files)
    cell_data_restart = PVTKFile(restart_files.cell_data)
    xFace_data_restart = PVTKFile(restart_files.xFace_data)
    yFace_data_restart = PVTKFile(restart_files.yFace_data)
    zFace_data_restart = PVTKFile(restart_files.zFace_data)
    pvd_file_restart = PVDFile(restart_files.pvd_data)

    # Create VTK object and fill dict 
    cell_data = get_cell_data(cell_data_restart)
    dict_data = Dict()

    xpt_data = get_cell_data(xFace_data_restart)
    ypt_data = get_cell_data(yFace_data_restart)
    zpt_data = get_cell_data(zFace_data_restart)

    pt_cell_keys = union(keys(xpt_data),keys(ypt_data),keys(zpt_data),keys(cell_data))
    for k in pt_cell_keys
        if k in keys(cell_data) 
            dict_data[k] = get_data(cell_data[k])
        elseif k in keys(xpt_data)
            dict_data[k] = get_data(xpt_data[k])
        elseif k in keys(ypt_data)
            dict_data[k] = get_data(ypt_data[k])
        elseif k in keys(zpt_data)
            dict_data[k] = get_data(zpt_data[k])
        end
    end

    return cell_data_restart,pvd_file_restart, dict_data
end

function domain_check(mesh,pvtk_dict)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    # println(pvtk_dict)
    imin = Int(first(pvtk_dict["i_index"][1]))
    imax = Int(last(pvtk_dict["i_index"][1]))
    jmin = Int(first(pvtk_dict["j_index"][1]))
    jmax = Int(last(pvtk_dict["j_index"][1]))
    kmin = Int(first(pvtk_dict["k_index"][1]))
    kmax = Int(last(pvtk_dict["k_index"][1]))

    checks = [
        (imin_, imin, "imin"),
        (imax_, imax, "imax"),
        (jmin_, jmin, "jmin"),
        (jmax_, jmax, "jmax"),
        (kmin_, kmin, "kmin"),
        (kmax_, kmax, "kmax")
    ]

    for (mesh_val, data_val, name) in checks
        if mesh_val != data_val
            println("Mismatch in $name: input mesh value is $mesh_val, restart value is $data_val")
        end
    end


end

# function fillArrays(pvtk_file,pvd_file,P,u,v,w,VF,nx,ny,nz,band,Curve,sfx,sfy,sfz,file,vec_index,param,mesh,par_env)
function fillArrays(pvtk_file,pvd_file,pvtk_dict,P,uf,vf,wf,VF,vec_index,vec_x_indexo,vec_y_indexo,vec_z_indexo,param,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack irank = par_env

    MPI.Barrier(par_env.comm)

    # Fill work arrays
    for k=kmin_:kmax_,j=jmin_:jmax_,i=imin_:imax_
        P[i,j,k] = pvtk_dict["Pressure"][irank+1][Int(vec_index[i,j,k]-(size(pvtk_dict["Pressure"][irank+1])[1]*irank))]
        VF[i,j,k] = pvtk_dict["VF"][irank+1][Int(vec_index[i,j,k]-(size(pvtk_dict["VF"][irank+1])[1]*irank))]
    end

    for k=kmin_-1:kmax_+1,j=jmin_-1:jmax_+1,i=imin_-1:imax_+2
        uf[i,j,k] = pvtk_dict["X_F_Velocity"][irank+1][Int(vec_x_indexo[i,j,k]-(size(pvtk_dict["X_F_Velocity"][irank+1])[1]*irank))]
    end

    for k=kmin_-1:kmax_+1,j=jmin_-1:jmax_+2,i=imin_-1:imax_+1
        vf[i,j,k] = pvtk_dict["Y_F_Velocity"][irank+1][Int(vec_y_indexo[i,j,k]-(size(pvtk_dict["Y_F_Velocity"][irank+1])[1]*irank))]
    end

    for k=kmin_-1:kmax_+2,j=jmin_-1:jmax_+1,i=imin_-1:imax_+1
        wf[i,j,k] = pvtk_dict["Z_F_Velocity"][irank+1][Int(vec_z_indexo[i,j,k]-(size(pvtk_dict["Z_F_Velocity"][irank+1])[1]*irank))]
    end





    # Grab iteration and timestep 
    pvd_time_index = findfirst(x -> x == basename(pvtk_file.filename),pvd_file.vtk_filenames)
    t = pvd_file.timesteps[pvd_time_index]
    n_step = pvd_time_index-1

    return t,n_step
end
 