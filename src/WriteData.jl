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

function VTK_init()
    # Create PVD file to hold timestep info
    pvd = paraview_collection("VTK")
    return pvd
end
   
function VTK_finalize(pvd)
    vtk_save(pvd)
    return nothing
end

function format(iter)
    return @sprintf("%05i",iter)
end

function VTK(iter,time,P,u,v,w,divg,param,mesh,par_env,pvd)
    
    # Check if should write output
    if rem(iter-1,param.out_freq)!==0
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
        "VTK"*format(iter), 
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
            pvtk["Divergence"] = @views divg[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvd[time] = pvtk
        end

    # Write pvd file to read even if simulation stops (or is stoped)
    if isopen(pvd)
        # if pvd.appended
        #     save_with_appended_data(pvd)
        # else
            WriteVTK.save_file(pvd.xdoc, pvd.path)
        # end
    end

    return nothing
end
