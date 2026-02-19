using WriteVTK
using Printf

function csv_init!(param,par_env)
    @unpack isroot = par_env
    @unpack VTK_dir,instability = param
    if isroot
        param_fields = [(key,getfield(param,key)) for key in fieldnames(typeof(param))]
        open(VTK_dir*".csv","w") do io
            println(io, join(first.(param_fields), ","))
            println(io, join(last.(param_fields), ","))
            header = "Iteration, Time, max(u), max(v), max(w), max(divg), sum(mass_err), Piters"
            println(io,header)
        end
        if instability == "kelvin-helmholtz"
            open(VTK_dir*"KH_height.csv","w") do io
                println(io, "Iteration, Time, Heights")
            end
        end
    end
    return nothing
end

function std_out(h_last,t_last,nstep,t,P,VF,u,v,w,divg,VF_init,iter,param,mesh,par_env)
    @unpack std_out_period,VTK_dir,instability = param
    @unpack isroot = par_env
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mesh

    max_u    = parallel_max(abs.(u[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),   par_env)
    max_v    = parallel_max(abs.(v[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),   par_env)
    max_w    = parallel_max(abs.(w[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),   par_env)
    max_divg = parallel_max(abs.(divg[imin_:imax_,jmin_:jmax_,kmin_:kmax_]),par_env)
    sum_VF = parallel_sum(VF[imin_:imax_,jmin_:jmax_,kmin_:kmax_]*dx*dy*dz,par_env)
    

    if isroot
        if (now = time()) - t_last[1] > std_out_period
            t_last[1] = now
            h_last[1] += 1
            # Write header
            if h_last[1] >= 10
                h_last[1] = 0
                @printf(" Iteration      Time    max(u)    max(v)    max(w)  max(divg)  sum(mass_err)   Piters\n")
            end
            # Write values
            @printf(" %9i  %8.3f  %8.3g  %8.3g  %8.3g   %8.3g      %9.3g      %8.3g \n",nstep,t,max_u,max_v,max_w,max_divg,VF_init-sum_VF,iter)
            
            # Write values to CSV
            open(VTK_dir*".csv","a") do io
                # if instability == "kelvin-helmholtz"
                #     println(io,("$nstep,$t,$max_u,$max_v,$max_w,$max_divg,$(VF_init-sum_VF),$iter,$A,$ϕ"))
                # else    
                    println(io,("$nstep,$t,$max_u,$max_v,$max_w,$max_divg,$(VF_init-sum_VF),$iter"))
                # end
            end
            if instability == "kelvin-helmholtz"
                heights = get_VF_heights(VF,mesh,param,par_env)
                open(VTK_dir*"KH_height.csv","a") do io
                    println(io,("$nstep,$t,"*join(heights,",")))
                end
            end
        end
    end
    return nothing
end

function VTK_init(param,par_env)
    @unpack isroot = par_env 
    @unpack VTK_dir,restart = param
    # Create PVD file to hold timestep info
    dir=joinpath(pwd(),VTK_dir)
    if isroot
        if restart
            # If restarting, ensure the directory exists
            if !isdir(dir)
                error("Restart requested, but VTK directory does not exist: $dir")
            end
        else
            isdir(dir) && rm(dir, recursive=true)
            mkdir(dir)
        end
    end
    MPI.Barrier(par_env.comm)
    pvd         = paraview_collection(joinpath(dir,"Solver"),append=restart)
    pvd_restart = paraview_collection(joinpath(dir,"restart"),append=restart)
    pvd_PLIC    = paraview_collection(joinpath(dir,"PLIC"),append=restart)
    return pvd,pvd_restart,pvd_PLIC
end
   
function VTK_finalize(pvd,pvd_xface,pvd_yface,pvd_zface,pvd_PLIC)
    vtk_save(pvd)
    vtk_save(pvd_restart)
    vtk_save(pvd_PLIC)
    return nothing
end

function format(iter)
    return @sprintf("%05i",iter)
end

function VTK(iter,time,P,u,v,w,uf,vf,wf,VF,nx,ny,nz,D,band,divg,Curve,tmp,param,mesh,par_env,pvd,pvd_restart,pvd_PLIC,sfx,sfy,sfz,denx,deny,denz,verts,tets)
    @unpack VTK_dir,restart = param
    @unpack irank = par_env

    # Check if should write output
    if rem(iter,param.out_period)!==0
        return nothing
    end

    @unpack x,y,z,xm,ym,zm,
            imin_,imax_,jmin_,jmax_,kmin_,kmax_,
            imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_,
            Gimin_,Gimax_,Gjmin_,Gjmax_,Gkmin_,Gkmax_,
            Gimino_,Gimaxo_,Gjmino_,Gjmaxo_,Gkmino_,Gkmaxo_ = mesh
    @unpack irank,nproc = par_env
    # Build extents array
    p=1; extents=[(Gimin_[p]:Gimax_[p]+1,Gjmin_[p]:Gjmax_[p]+1,Gkmin_[p]:Gkmax_[p]+1), ]
    for p = 2:nproc
       push!(extents,(Gimin_[p]:Gimax_[p]+1,Gjmin_[p]:Gjmax_[p]+1,Gkmin_[p]:Gkmax_[p]+1))
    end

    # Write data to VTK
    pvtk_grid(
        joinpath(pwd(),VTK_dir,"Solver_"*format(iter)), 
        x[imin_:imax_+1], 
        y[jmin_:jmax_+1],
        z[kmin_:kmax_+1],
        part = irank+1,
        nparts = nproc,
        extents = extents,
        ) do pvtk
            pvtk["Pressure"] = @views P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvtk["Velocity"] = @views (
                u[imin_:imax_,jmin_:jmax_,kmin_:kmax_],
                v[imin_:imax_,jmin_:jmax_,kmin_:kmax_],
                w[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
            pvtk["VF"] = @views VF[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvtk["band"] = @views band[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvtk["Divergence"] = @views divg[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvtk["Normal"] = @views (
                nx[imin_:imax_,jmin_:jmax_,kmin_:kmax_],
                ny[imin_:imax_,jmin_:jmax_,kmin_:kmax_],
                nz[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
            pvtk["Curve"] = @views Curve[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvtk["SFx"] = @views sfx[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvtk["SFy"] = @views sfy[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvtk["SFz"] = @views sfz[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvtk["rho_x"] = @views 0.5*(denx[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+ denx[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_])
            pvtk["rho_y"] = @views 0.5*(deny[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+ deny[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_])
            pvtk["rho_z"] = @views 0.5*(denz[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+ denz[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1])
            
            # Indices for debugging
            for i=imin_:imax_; tmp[i,:,:] .= i; end
            pvtk["i_index"] = @views tmp[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            for j=jmin_:jmax_; tmp[:,j,:] .= j; end
            pvtk["j_index"] = @views tmp[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            for k=kmin_:kmax_; tmp[:,:,k] .= k; end
            pvtk["k_index"] = @views tmp[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvd[time] = pvtk
        end
    
    ## Restart data (note not stored at correct locations)
    p=1; extents_restart=[(Gimino_[p]:Gimaxo_[p]+1,Gjmino_[p]:Gjmaxo_[p]+1,Gkmino_[p]:Gkmaxo_[p]+1), ]
    for p = 2:nproc
       push!(extents_restart,(Gimino_[p]:Gimaxo_[p]+1,Gjmino_[p]:Gjmaxo_[p]+1,Gkmino_[p]:Gkmaxo_[p]+1))
    end

    # Write data to VTK
    pvtk_grid(
        joinpath(pwd(),VTK_dir,"restart_"*format(iter)), 
        x[imino_:imaxo_+1], 
        y[jmino_:jmaxo_+1],
        z[kmino_:kmaxo_+1],
        part = irank+1,
        nparts = nproc,
        extents = extents_restart,
        ) do pvtk
        pvtk["VF"] = @views VF[imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_]
        pvtk["uf"] = @views uf[imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_]
        pvtk["vf"] = @views vf[imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_]
        pvtk["wf"] = @views wf[imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_]
        pvtk["P" ] = @views  P[imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_]
        pvd_restart[time] = pvtk
    end

    # Write PLIC as unstructured mesh 
    pts, tris = PLIC2Mesh(nx,ny,nz,D,VF,verts,tets,param,mesh)

    # Put pts into VTK format 
    npts = length(pts)
    points = Matrix{Float64}(undef,3,npts)
    for n = eachindex(pts)
        points[:,n]=pts[n].pt
    end

    # Put tris into VTK format
    ncells = length(tris)
    cells  = Vector{WriteVTK.MeshCell{WriteVTK.VTKCellTypes.VTKCellType, Vector{Int64}}}(undef,ncells)
    ID  = Vector{Float64}(undef,ncells)
    for n = eachindex(tris)
        cells[n]=MeshCell(VTKCellTypes.VTK_TRIANGLE, tris[n].tri)
        ID[n] = n
    end
        
    # Write data
    if ncells > 0    
        pvtk_grid(
            joinpath(pwd(),VTK_dir,"PLIC_"*format(iter)), 
            points, 
            cells,
            part = irank+1,
            nparts = nproc,
            extents = [1,length(tris)],
            #append = restart,
            ) do pvtk
                pvtk["Tri ID"] = ID
                pvd_PLIC[time] = pvtk
        end
    end

    # Write pvd file to read even if simulation stops (or is stoped)
    if isopen(pvd)
        # if pvd.appended
        #     WriteVTK.save_with_appended_data(pvd)
        # else
            WriteVTK.save_file(pvd.xdoc, pvd.path)
        # end
    end

    if isopen(pvd_restart)
        # if pvd.appended
        #     WriteVTK.save_with_appended_data(pvd)
        # else
            WriteVTK.save_file(pvd_restart.xdoc, pvd_restart.path)
        # end
    end
    
    if isopen(pvd_PLIC)
        # if pvd_PLIC.appended
        #     WriteVTK.save_with_appended_data(pvd_PLIC)
        # else
            WriteVTK.save_file(pvd_PLIC.xdoc, pvd_PLIC.path)
        # end
    end

    return nothing
end

"""
Writes an array of tets[coord,vert,tet]
where coord = [1:3] and corresponds to x,y,z of each vert 
      vert = [1:4] and corresponds to each vertex in tet 
      tet = [1:n] where n is the number of tets 
"""
function tets2VTK(tets,filename)
    # Convert tets to points & tris 
    nTet = size(tets,3)
    pts  = Vector{Point}(undef,4*nTet)
    tris = Vector{  Tri}(undef,4*nTet)
    for n=1:nTet 
        # Compute index of 4 points on this tet (also index of tris)
        ind = 4(n-1).+(1:4)
        for nn=1:4
            pts[ind[nn]] = Point(tets[:,nn,n])
        end
        tris[ind[1]] = Tri([ind[1],ind[3],ind[2]])
        tris[ind[2]] = Tri([ind[1],ind[2],ind[4]])
        tris[ind[3]] = Tri([ind[1],ind[4],ind[3]])
        tris[ind[4]] = Tri([ind[2],ind[3],ind[4]])
    end

    # Put pts into VTK format 
    npts = length(pts)
    points = Matrix{Float64}(undef,3,npts)
    for n = eachindex(pts)
        points[:,n]=pts[n].pt
    end

    # Put tris into VTK format
    ncells = length(tris)
    cells  = Vector{WriteVTK.MeshCell{WriteVTK.VTKCellTypes.VTKCellType, Vector{Int64}}}(undef,ncells)
    ID  = Vector{Float64}(undef,ncells)
    for n = eachindex(tris)
        cells[n]=MeshCell(VTKCellTypes.VTK_TRIANGLE, tris[n].tri)
        ID[n] = Int(floor((n-1)/4)+1)
    end

    # Write data
    if ncells > 0    
        vtk_grid(
            filename, 
            points, 
            cells,
            ) do vtk
                vtk["Tri ID"] = ID
        end
    end

end


"""
Writes analytic pre-image mesh object to unstructured VTK file using poly-lines.
"""
function a_preimage2VTK(ap, filename)
    # Number of edges and points per edge
    n_edges = length(ap.edge_pts)
    sample_freq = length(ap.edge_pts[1])  # assume all edges sampled equally
    npts = n_edges * sample_freq

    # Flatten all edge points into a 3xN matrix
    points = Matrix{Float64}(undef, 3, npts)
    pt_counter = 1
    for e in 1:n_edges
        for s in 1:sample_freq
            points[:, pt_counter] .= ap.edge_pts[e][s]
            pt_counter += 1
        end
    end

    # Build polyline connectivity (VTK_POLY_LINE)
    ncells = n_edges
    cells = Vector{WriteVTK.MeshCell{WriteVTK.VTKCellTypes.VTKCellType, Vector{Int64}}}(undef, ncells)
    pt_offset = 0

    for e in 1:n_edges
        indices = pt_offset .+ (1:sample_freq)  # all points along this edge
        cells[e] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_POLY_LINE, collect(indices))
        pt_offset += sample_freq
    end

    # Optional: attach per-cell data (e.g., edge_cell ID)
    cell_data = Dict(
        "edge_cell" => ap.edge_cell
    )

    # Write to VTU
    vtk_grid(filename, points, cells) do vtk
        for (name, data) in cell_data
            vtk["cell_data/$name"] = data
        end
    end
end

function pVTK_init(param,par_env)
    @unpack isroot = par_env
    dir = joinpath(pwd(),"Piter_fields")

    if isroot
        isdir(dir) && rm(dir, recursive=true)
        mkdir(dir)
    end

    pvd_pressure = paraview_collection(joinpath(dir,"P_Solver"),append=true)
    return pvd_pressure,dir
end

function pressure_VTK(iter,P,AP,sfx,sfy,sfz,dir,pvd_pressure,param,mesh,par_env)
    @unpack x,y,z,xm,ym,zm,
    imin_,imax_,jmin_,jmax_,kmin_,kmax_,
    imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_,
    Gimin_,Gimax_,Gjmin_,Gjmax_,Gkmin_,Gkmax_,
    Gimino_,Gimaxo_,Gjmino_,Gjmaxo_,Gkmino_,Gkmaxo_ = mesh
    @unpack irank,nproc = par_env


    # Build extents array
    p=1; extents=[(Gimin_[p]:Gimax_[p]+1,Gjmin_[p]:Gjmax_[p]+1,Gkmin_[p]:Gkmax_[p]+1), ]
    for p = 2:nproc
        push!(extents,(Gimin_[p]:Gimax_[p]+1,Gjmin_[p]:Gjmax_[p]+1,Gkmin_[p]:Gkmax_[p]+1))
    end

    pvtk_grid(
        joinpath(pwd(),dir,"Piter_field_$iter"),
        x[imin_:imax_+1], 
        y[jmin_:jmax_+1],
        z[kmin_:kmax_+1],
        part = irank+1,
        nparts = nproc,
        extents = extents,
    ) do pvtk 
        pvtk["Pressure"] = @views P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        pvtk["AP"] = @views AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        pvtk["SFx"] = @views sfx[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        pvtk["SFy"] = @views sfy[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        pvtk["SFz"] = @views sfz[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
        pvd_pressure[iter] = pvtk
    end

    if isopen(pvd_pressure)
        WriteVTK.save_file(pvd_pressure.xdoc, pvd_pressure.path)
    end

    return nothing
end

function metrics2CSV(file_path,param,iter;kwargs...)
    @unpack restart,restart_itr = param

    if !isfile(file_path)
        open(file_path,"w") do file 
            println(file, "iter," *join(keys(kwargs),","))
        end
    end

    if iter == 1 || iter == restart_iter+1
        if isfile(file_path) && restart
            lines = readlines(file_path)
            filtered_lines = [lines[1]]
            
            for line in lines[2:end]
                row_iter = parse(Int,split(line,",")[1])
                if row_iter <= restart_itr
                    push!(filtered_lines,line)
                end
            end

            open(file_path,"w") do file
                foreach(l->println(file,l),filtered_lines)
                println(file,"$iter,"*join(values(kwargs),","))
            end
        end
    end

    open(file_path,"a") do file 
        println(file,"$iter ,"*join(values(kwargs),","))
    end
end


