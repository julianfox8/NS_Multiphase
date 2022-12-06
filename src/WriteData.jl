using WriteVTK
using Printf

function std_out(nstep,t,P,u,v,w,divg,iter,par_env)
    @unpack isroot = par_env

    max_u    = parallel_max(abs.(u),   par_env)
    max_v    = parallel_max(abs.(v),   par_env)
    max_w    = parallel_max(abs.(w),   par_env)
    max_divg = parallel_max(abs.(divg),par_env)
    
    if isroot 
        rem(nstep,10)==0 && @printf(" Iteration      Time    max(u)    max(v)    max(w) max(divg)    Piters\n")
        @printf(" %9i  %8.3f  %8.3g  %8.3g  %8.3g  %8.3g  %8.3g \n",nstep,t,max_u,max_v,max_w,max_divg,iter)
    end

    return nothing
end

function VTK_init(param,par_env)
    @unpack isroot = par_env 
    @unpack VTK_dir = param
    # Create PVD file to hold timestep info
    dir=joinpath(pwd(),VTK_dir)
    if isroot 
        isdir(dir) && rm(dir, recursive=true)
        mkdir(dir)
    end
    pvd      = paraview_collection(joinpath(dir,"Solver"))
    pvd_PLIC = paraview_collection(joinpath(dir,"PLIC"))
    return pvd,pvd_PLIC
end
   
function VTK_finalize(pvd,pvd_PLIC)
    vtk_save(pvd)
    vtk_save(pvd_PLIC)
    return nothing
end

function format(iter)
    return @sprintf("%05i",iter)
end

function VTK(iter,time,P,u,v,w,VF,nx,ny,nz,D,divg,tmp,param,mesh,par_env,pvd,pvd_PLIC)
    @unpack VTK_dir = param
    
    # Check if should write output
    if rem(iter,param.out_period)!==0
        return nothing
    end

    @unpack x,y,z,xm,ym,zm,
            imin_,imax_,jmin_,jmax_,kmin_,kmax_,
            Gimin_,Gimax_,Gjmin_,Gjmax_,Gkmin_,Gkmax_ = mesh
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
            pvtk["Divergence"] = @views divg[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvtk["Normal"] = @views (
                nx[imin_:imax_,jmin_:jmax_,kmin_:kmax_],
                ny[imin_:imax_,jmin_:jmax_,kmin_:kmax_],
                nz[imin_:imax_,jmin_:jmax_,kmin_:kmax_] )
            # Indices for debugging
            for i=imin_:imax_; tmp[i,:,:] .= i; end
            pvtk["i_index"] = @views tmp[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            for j=jmin_:jmax_; tmp[:,j,:] .= j; end
            pvtk["j_index"] = @views tmp[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            for k=kmin_:kmax_; tmp[:,:,k] .= k; end
            pvtk["k_index"] = @views tmp[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvd[time] = pvtk
        end

    # Write PLIC as unstructured mesh 
    pts, tris = PLIC2Mesh(nx,ny,nz,D,VF,param,mesh)

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
            ) do pvtk
                pvtk["Tri ID"] = ID
                pvd_PLIC[time] = pvtk
        end
    end

    # Write pvd file to read even if simulation stops (or is stoped)
    if isopen(pvd)
        # if pvd.appended
        #     save_with_appended_data(pvd)
        # else
        WriteVTK.save_file(pvd.xdoc, pvd.path)
        # end
    end
    if isopen(pvd_PLIC)
        WriteVTK.save_file(pvd_PLIC.xdoc, pvd_PLIC.path)
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
