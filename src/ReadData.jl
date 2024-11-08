using ReadVTK

function fillArrays(P,uf,vf,wf,VF,param,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack irank = par_env
    @unpack VTK_dir,restart_itr = param

    # Format restart iteration number 
    restart_itr = format(restart_itr)
   
    # Read VTK files
    data_restart_pvtr = PVTKFile(joinpath(VTK_dir,"restart_" *restart_itr*".pvtr"),dir=VTK_dir)
    pvd_file_restart = PVDFile(joinpath(VTK_dir,"restart.pvd"))

    # Get data 
    data_restart = get_cell_data(data_restart_pvtr)

    # Create cell data objects and fill dict 
    pvtk_data = Dict()
    for k in keys(data_restart)
        pvtk_data[k] = get_data(data_restart[k])
    end
    
    # Get extents
    global_extents = ReadVTK.get_extents(data_restart.parent_xml)
    ext_x = global_extents[irank+1][1][end]-global_extents[irank+1][1][1]
    ext_y = global_extents[irank+1][2][end]-global_extents[irank+1][2][1]
    ext_z = global_extents[irank+1][3][end]-global_extents[irank+1][3][1]

    # Reshape data
    pvtk_data["VF"][irank+1] = reshape(pvtk_data["VF"][irank+1],(ext_x,ext_y,ext_z))
    pvtk_data["uf"][irank+1] = reshape(pvtk_data["uf"][irank+1],(ext_x,ext_y,ext_z))
    pvtk_data["vf"][irank+1] = reshape(pvtk_data["vf"][irank+1],(ext_x,ext_y,ext_z))
    pvtk_data["wf"][irank+1] = reshape(pvtk_data["wf"][irank+1],(ext_x,ext_y,ext_z))
    pvtk_data["P" ][irank+1] = reshape(pvtk_data["P" ][irank+1],(ext_x,ext_y,ext_z))

    # Fill work arrays
    for k=kmino_:kmaxo_,j=jmino_:jmaxo_,i=imino_:imaxo_
        VF[i,j,k] = pvtk_data["VF"][irank+1][i-imino_+1,j-jmino_+1,k-kmino_+1]
        uf[i,j,k] = pvtk_data["uf"][irank+1][i-imino_+1,j-jmino_+1,k-kmino_+1]
        vf[i,j,k] = pvtk_data["vf"][irank+1][i-imino_+1,j-jmino_+1,k-kmino_+1]
        wf[i,j,k] = pvtk_data["wf"][irank+1][i-imino_+1,j-jmino_+1,k-kmino_+1]
        P[i,j,k]  = pvtk_data["P" ][irank+1][i-imino_+1,j-jmino_+1,k-kmino_+1]
    end

    # Grab iteration and timestep 
    pvd_time_index = findfirst(x -> x == basename(data_restart_pvtr.filename),pvd_file_restart.vtk_filenames)
    t = pvd_file_restart.timesteps[pvd_time_index]

    n_step = pvd_time_index-1

    return t,n_step
end
 
function pvd_file_cleanup!(t,param)
    @unpack VTK_dir = param

    # clean up XML file to remove iterations past restart timestep
    for file in ["Solver.pvd","PLIC.pvd","restart.pvd"]
        doc = readxml(joinpath(VTK_dir,file))
        elements2remove = []
        for i in eachelement(doc.root)
            for j in eachelement(i)
                if parse(Float64,j["timestep"]) > t
                    push!(elements2remove,j)
                end
            end
        end
        for n in elements2remove
            unlink!(n)
        end
        write(joinpath(VTK_dir,file),doc)
    end
end