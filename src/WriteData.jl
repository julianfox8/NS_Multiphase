using WriteVTK
using Printf

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
   


function VTK(iter,time,P,mesh,par_env,pvd)
    @unpack x,y,z,xm,ym,zm,
            imin_,imax_,jmin_,jmax_,kmin_,kmax_,
            Gimin_,Gimax_,Gjmin_,Gjmax_,Gkmin_,Gkmax_ = mesh
    @unpack irank,nproc = par_env
    # Build extents array
    p=1; extents=[(Gimin_[p]:Gimax_[p]+1,Gjmin_[p]:Gjmax_[p]+1,Gkmin_[p]:Gkmax_[p]+1), ]
    for p in 2:nproc
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
            pvtk["pressure"] = P[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
            pvd[time] = pvtk
        end

end
