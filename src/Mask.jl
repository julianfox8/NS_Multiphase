struct mask_object
    xmin; xmax; ymin; ymax
end

function mask_create(obj::mask_object,mesh::mesh_struct,mesh_par::mesh_struct_par)
    #CREATE_MASK tags cells as either 0 in fluid or 1 in solid object
    #   Inputs
    #     obj : Structure containing xmin,xmax,ymin,ymax of object
    #     xm,ym :  Arrays of cell centers
    #     imin,imax,jmin,jmax : index extents
    #   Outputs
    #     mask  : array with 0 in fluid cells and 1 in solid cells

    @unpack xm,ym = mesh
    @unpack imin_,imax_,jmin_,jmax_ = mesh_par

    # Initialize mask array to 0 everywhere
    mask = OffsetArray{Float64}(undef, imin_:imax_+1,jmin_:jmax_+1)

    # Loop over the domain
    for j=jmin_:jmax_
        for i=imin_:imax_
            # Check if within solid
            if  xm[i] > obj.xmin && xm[i] < obj.xmax &&
                ym[j] > obj.ymin && ym[j] < obj.ymax
                # Set mask=1 in solid
                mask[i,j]=1;
            else
                mask[i,j]=0;
            end
        end
    end

    return mask
end