# __precompile__

""" 
Apply BC's of kind type to A
"""
function BC_apply!(A,type,mesh_par,par_env)

    @unpack imin_,imax_,jmin_,jmax_ = mesh_par 
    @unpack nprocx,nprocy,irankx,iranky = par_env

    if type == "Neuman"
        irankx == 0        ? A[imin_-1,:]=A[imin_,:] : nothing # Left 
        irankx == nprocx-1 ? A[imax_+1,:]=A[imax_,:] : nothing # Right
        iranky == 0        ? A[:,jmin_-1]=A[:,jmin_] : nothing # Bottom
        iranky == nprocy-1 ? A[:,jmax_+1]=A[:,jmax_] : nothing # Top
    else
        error("Unknown boundary condition type $type")
    end
    
    return nothing
end
