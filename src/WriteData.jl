using WriteVTK
using Printf

function std_out(nstep,t,P,u,v,w,divg,iter,par_env)
    @unpack isroot = par_env

    max_u    = parallel_max(abs.(u),   par_env)
    max_v    = parallel_max(abs.(v),   par_env)
    max_w    = parallel_max(abs.(w),   par_env)
    max_divg = parallel_max(abs.(divg),par_env)
    
    if isroot 
        rem(nstep,10)==1 && @printf(" Iteration      Time    max(u)    max(v)    max(w) max(divg)    Piters\n")
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
    if rem(iter-1,param.out_period)!==0
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
    points = [
        0.0 0.0 0.0
        1.0 0.0 0.0
        1.0 1.0 0.0
        2.0 1.0 0.0
        ]'
    cells = [
        MeshCell(VTKCellTypes.VTK_TRIANGLE, [1, 2, 3]),
        MeshCell(VTKCellTypes.VTK_TRIANGLE, [2, 3, 4]),
            ]
    pvtk_grid(
        joinpath(pwd(),VTK_dir,"PLIC_"*format(iter)), 
        points, 
        cells,
        part = irank+1,
        nparts = nproc,
        ) do pvtk
            pvtk["testData"] = [1.234, 5.678]
            pvd_PLIC[time] = pvtk
    end

    # Write pvd file to read even if simulation stops (or is stoped)
    if isopen(pvd)
        # if pvd.appended
        #     save_with_appended_data(pvd)
        # else
        println("pvd.path=",pvd.path)
        WriteVTK.save_file(pvd.xdoc, pvd.path)
        # end
    end
    if isopen(pvd_PLIC)
        WriteVTK.save_file(pvd_PLIC.xdoc, pvd_PLIC.path)
    end

    return nothing
end
