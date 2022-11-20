struct mask_object
    xmin; xmax; ymin; ymax; zmin; zmax;
end

function mask_create(obj,mesh)
    @unpack xm,ym,zm,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Initialize mask array to 0 everywhere
    mask = OffsetArray{Bool}(undef, imino_:imaxo_+1,jmino_:jmaxo_+1,kmino_:kmaxo_+1)

    # Check if obj is defined
    if obj === nothing 
        fill!(mask,false)
        return mask 
    end

    # Loop over the domain
    for k=kmino_:kmaxo_, j=jmino_:jmaxo_, i=imino_:imaxo_
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
