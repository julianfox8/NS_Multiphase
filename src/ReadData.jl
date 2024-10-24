using ReadVTK

function gather_restart_files(restart_files,mesh,par_env)
    @unpack irank = par_env
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Read VTK files
    cell_data_restart = PVTKFile(restart_files.cell_data)
    xFace_data_restart = PVTKFile(restart_files.xFace_data)
    yFace_data_restart = PVTKFile(restart_files.yFace_data)
    zFace_data_restart = PVTKFile(restart_files.zFace_data)
    pvd_file_restart = PVDFile(restart_files.pvd_data)

    # Create cell data objects and fill dict 
    cell_data = get_cell_data(cell_data_restart)
    xpt_data = get_cell_data(xFace_data_restart)
    ypt_data = get_cell_data(yFace_data_restart)
    zpt_data = get_cell_data(zFace_data_restart)
    pvtk_data = Dict()

    # Need keys from each cell data object
    pt_cell_keys = union(keys(xpt_data),keys(ypt_data),keys(zpt_data),keys(cell_data))

    for k in pt_cell_keys
        if k in keys(cell_data) 
            pvtk_data[k] = get_data(cell_data[k])
        elseif k in keys(xpt_data)
            pvtk_data[k] = get_data(xpt_data[k])
        elseif k in keys(ypt_data)
            pvtk_data[k] = get_data(ypt_data[k])
        elseif k in keys(zpt_data)
            pvtk_data[k] = get_data(zpt_data[k])
        end
    end

    # Grab extents and reshape VTK data arrays
    global_extents = ReadVTK.get_extents(cell_data.parent_xml)
    ext_x = global_extents[irank+1][1][end]-global_extents[irank+1][1][1]
    ext_y = global_extents[irank+1][2][end]-global_extents[irank+1][2][1]
    ext_z = global_extents[irank+1][3][end]-global_extents[irank+1][3][1]

    pvtk_data["Pressure"][irank+1] = reshape(pvtk_data["Pressure"][irank+1],(ext_x,ext_y,ext_z))
    pvtk_data["VF"][irank+1] = reshape(pvtk_data["VF"][irank+1],(ext_x,ext_y,ext_z))

    #Grab x-face extents and reshape VTK data array 
    xface_ghost_ext = ReadVTK.get_extents(xpt_data.parent_xml)
    xghost_ext_x = xface_ghost_ext[irank+1][1][end]-xface_ghost_ext[irank+1][1][1]+1
    xghost_ext_y = xface_ghost_ext[irank+1][2][end]-xface_ghost_ext[irank+1][2][1]+1
    xghost_ext_z = xface_ghost_ext[irank+1][3][end]-xface_ghost_ext[irank+1][3][1]+1

    pvtk_data["X_F_Velocity"][irank+1] = reshape(pvtk_data["X_F_Velocity"][irank+1],(xghost_ext_x,xghost_ext_y,xghost_ext_z))

    #Grab x-face extents and reshape VTK data array
    yface_ghost_ext = ReadVTK.get_extents(ypt_data.parent_xml)
    yghost_ext_x = yface_ghost_ext[irank+1][1][end]-yface_ghost_ext[irank+1][1][1]+1
    yghost_ext_y = yface_ghost_ext[irank+1][2][end]-yface_ghost_ext[irank+1][2][1]+1
    yghost_ext_z = yface_ghost_ext[irank+1][3][end]-yface_ghost_ext[irank+1][3][1]+1

    pvtk_data["Y_F_Velocity"][irank+1] = reshape(pvtk_data["Y_F_Velocity"][irank+1],(yghost_ext_x,yghost_ext_y,yghost_ext_z))

    #Grab x-face extents and reshape VTK data array
    zface_ghost_ext = ReadVTK.get_extents(zpt_data.parent_xml)
    zghost_ext_x = zface_ghost_ext[irank+1][1][end]-zface_ghost_ext[irank+1][1][1]+1
    zghost_ext_y = zface_ghost_ext[irank+1][2][end]-zface_ghost_ext[irank+1][2][1]+1
    zghost_ext_z = zface_ghost_ext[irank+1][3][end]-zface_ghost_ext[irank+1][3][1]+1

    pvtk_data["Z_F_Velocity"][irank+1] = reshape(pvtk_data["Z_F_Velocity"][irank+1],(zghost_ext_x,zghost_ext_y,zghost_ext_z))

    return cell_data_restart,pvd_file_restart, pvtk_data
end

function domain_check(mesh,pvtk_data)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    # println(pvtk_data)
    imin = Int(first(pvtk_data["i_index"][1]))
    imax = Int(last(pvtk_data["i_index"][1]))
    jmin = Int(first(pvtk_data["j_index"][1]))
    jmax = Int(last(pvtk_data["j_index"][1]))
    kmin = Int(first(pvtk_data["k_index"][1]))
    kmax = Int(last(pvtk_data["k_index"][1]))

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

function reshape_pvtk_data(data, pt_data, irank)
    extents = ReadVTK.get_extents(pt_data.parent_xml)
    ext_x = extents[irank+1][1][end] - extents[irank+1][1][1] + 1
    ext_y = extents[irank+1][2][end] - extents[irank+1][2][1] + 1
    ext_z = extents[irank+1][3][end] - extents[irank+1][3][1] + 1
    reshape(data, (ext_x, ext_y, ext_z))
end

function fillArrays(restart_files,P,uf,vf,wf,VF,param,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack irank = par_env
   
    # Read VTK files
    cell_data_restart = PVTKFile(restart_files.cell_data)
    xFace_data_restart = PVTKFile(restart_files.xFace_data)
    yFace_data_restart = PVTKFile(restart_files.yFace_data)
    zFace_data_restart = PVTKFile(restart_files.zFace_data)
    pvd_file_restart = PVDFile(restart_files.pvd_data)

    # Create cell data objects and fill dict 
    cell_data = get_cell_data(cell_data_restart)
    xpt_data = get_cell_data(xFace_data_restart)
    ypt_data = get_cell_data(yFace_data_restart)
    zpt_data = get_cell_data(zFace_data_restart)
    pvtk_data = Dict()

    # Need keys from each cell data object
    pt_cell_keys = union(keys(xpt_data),keys(ypt_data),keys(zpt_data),keys(cell_data))

    for k in pt_cell_keys
        if k in keys(cell_data) 
            pvtk_data[k] = get_data(cell_data[k])
        elseif k in keys(xpt_data)
            pvtk_data[k] = get_data(xpt_data[k])
        elseif k in keys(ypt_data)
            pvtk_data[k] = get_data(ypt_data[k])
        elseif k in keys(zpt_data)
            pvtk_data[k] = get_data(zpt_data[k])
        end
    end

    # Grab extents and reshape VTK data arrays
    global_extents = ReadVTK.get_extents(cell_data.parent_xml)
    ext_x = global_extents[irank+1][1][end]-global_extents[irank+1][1][1]
    ext_y = global_extents[irank+1][2][end]-global_extents[irank+1][2][1]
    ext_z = global_extents[irank+1][3][end]-global_extents[irank+1][3][1]

    pvtk_data["Pressure"][irank+1] = reshape(pvtk_data["Pressure"][irank+1],(ext_x,ext_y,ext_z))
    pvtk_data["VF"][irank+1] = reshape(pvtk_data["VF"][irank+1],(ext_x,ext_y,ext_z))

    # # #Grab x-face extents and reshape VTK data array 
    # pvtk_data["X_F_Velocity"][irank+1] = reshape_pvtk_data(pvtk_data["X_F_Velocity"][irank+1],xpt_data,irank)

    # # #Grab x-face extents and reshape VTK data array
    # pvtk_data["Y_F_Velocity"][irank+1] = reshape_pvtk_data(pvtk_data["Y_F_Velocity"][irank+1],ypt_data,irank)

    # # #Grab x-face extents and reshape VTK data array
    # pvtk_data["Z_F_Velocity"][irank+1] = reshape_pvtk_data(pvtk_data["Z_F_Velocity"][irank+1],zpt_data,irank)
    #Grab x-face extents and reshape VTK data array 
    xface_ghost_ext = ReadVTK.get_extents(xpt_data.parent_xml)
    xghost_ext_x = xface_ghost_ext[irank+1][1][end]-xface_ghost_ext[irank+1][1][1]+1
    xghost_ext_y = xface_ghost_ext[irank+1][2][end]-xface_ghost_ext[irank+1][2][1]+1
    xghost_ext_z = xface_ghost_ext[irank+1][3][end]-xface_ghost_ext[irank+1][3][1]+1

    pvtk_data["X_F_Velocity"][irank+1] = reshape(pvtk_data["X_F_Velocity"][irank+1],(xghost_ext_x,xghost_ext_y,xghost_ext_z))

    #Grab x-face extents and reshape VTK data array
    yface_ghost_ext = ReadVTK.get_extents(ypt_data.parent_xml)
    yghost_ext_x = yface_ghost_ext[irank+1][1][end]-yface_ghost_ext[irank+1][1][1]+1
    yghost_ext_y = yface_ghost_ext[irank+1][2][end]-yface_ghost_ext[irank+1][2][1]+1
    yghost_ext_z = yface_ghost_ext[irank+1][3][end]-yface_ghost_ext[irank+1][3][1]+1

    pvtk_data["Y_F_Velocity"][irank+1] = reshape(pvtk_data["Y_F_Velocity"][irank+1],(yghost_ext_x,yghost_ext_y,yghost_ext_z))

    #Grab x-face extents and reshape VTK data array
    zface_ghost_ext = ReadVTK.get_extents(zpt_data.parent_xml)
    zghost_ext_x = zface_ghost_ext[irank+1][1][end]-zface_ghost_ext[irank+1][1][1]+1
    zghost_ext_y = zface_ghost_ext[irank+1][2][end]-zface_ghost_ext[irank+1][2][1]+1
    zghost_ext_z = zface_ghost_ext[irank+1][3][end]-zface_ghost_ext[irank+1][3][1]+1

    pvtk_data["Z_F_Velocity"][irank+1] = reshape(pvtk_data["Z_F_Velocity"][irank+1],(zghost_ext_x,zghost_ext_y,zghost_ext_z))

    # Fill work arrays
    for k=kmin_:kmax_,j=jmin_:jmax_,i=imin_:imax_
        P[i,j,k] = pvtk_data["Pressure"][irank+1][i-imin_+1,j-jmin_+1,k-kmin_+1]
        VF[i,j,k] = pvtk_data["VF"][irank+1][i-imin_+1,j-jmin_+1,k-kmin_+1]
    end

    for k=kmin_-1:kmax_+1,j=jmin_-1:jmax_+1,i=imin_-1:imax_+2
        uf[i,j,k] = pvtk_data["X_F_Velocity"][irank+1][i-imin_+2,j-jmin_+2,k-kmin_+2]
    end

    for k=kmin_-1:kmax_+1,j=jmin_-1:jmax_+2,i=imin_-1:imax_+1
        vf[i,j,k] = pvtk_data["Y_F_Velocity"][irank+1][i-imin_+2,j-jmin_+2,k-kmin_+2]
    end

    for k=kmin_-1:kmax_+2,j=jmin_-1:jmax_+1,i=imin_-1:imax_+1
        wf[i,j,k] = pvtk_data["Z_F_Velocity"][irank+1][i-imin_+2,j-jmin_+2,k-kmin_+2]
    end

    # Grab iteration and timestep 
    pvd_time_index = findfirst(x -> x == basename(cell_data_restart.filename),pvd_file_restart.vtk_filenames)
    t = pvd_file_restart.timesteps[pvd_time_index]

    n_step = pvd_time_index-1

    return t,n_step
end
 
function pvd_file_cleanup!(restart_files, t)
    # clean up XML file to remove iterations past restart timestep
    doc = readxml(restart_files.pvd_data)
    elements2remove = []
    for i in eachelement(doc.root)
        for j in eachelement(i)
            if parse(Float64,j["timestep"]) >= t
                push!(elements2remove,j)
            end
        end
    end

    for n in elements2remove
        unlink!(n)
    end

    write(restart_files.pvd_data,doc)
end