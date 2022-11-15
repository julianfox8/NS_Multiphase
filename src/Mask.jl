struct mask_object
    xmin; xmax; ymin; ymax; zmin; zmax;
end

function mask_create(obj,mesh)
    @unpack xm,ym,zm,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    # Initialize mask array to 0 everywhere
    mask = OffsetArray{Bool}(undef, imin_:imax_+1,jmin_:jmax_+1,kmin_:kmax_+1)

    # Check if obj is defined
    if obj === nothing 
        fill!(mask,false)
        return mask 
    end

    # Loop over the domain
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # Check if within solid
        if  xm[i] > obj.xmin && xm[i] < obj.xmax &&
            ym[j] > obj.ymin && ym[j] < obj.ymax &&
            zm[j] > obj.zmin && zm[j] < obj.zmax
            # Set mask=true in solid
            mask[i,j,k]=true;
        else
            mask[i,j,k]=false;
        end
    end

    return mask
end
