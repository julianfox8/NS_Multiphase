""" 
Look-up tables for cutting triangles
"""

# Number of new vertices on cut plane
const cutTri_nvert = [0, 2, 2, 2, 2, 2, 2, 0]

# Number of resulting tris on positive side
const cutTri_np = [0, 1, 1, 2, 1, 2, 2, 1]
 
# Number of resulting tris on negative side
const cutTri_nn  = [1, 2, 2, 1, 2, 1, 1, 0]

# First point on intersection
const cutTri_v1 = reshape([
 -1 -1
  1  1
  2  2
  3  3
  3  3
  2  2
  1  1
 -1 -1
]',(2,8))

# Second point on intersection
const cutTri_v2 = reshape([
  -1 -1
   2  3
   1  3
   1  2
   1  2
   1  3
   2  3
  -1 -1
]',(2,8))

# Vertices in each tri
const cutTri_v = reshape([
  1  2  3  -1 -1 -1  -1 -1 -1
  1  4  5   2  5  4   2  3  5
  2  5  4   1  4  5   1  5  3
  1  2  5   1  5  4   3  4  5
  3  4  5   1  2  5   1  5  4
  1  4  5   1  5  3   2  5  4
  2  3  5   2  5  4   1  4  5
  1  2  3  -1 -1 -1  -1 -1 -1
]',(3,3,8))

""" 
Look-up tables for cutting tetrahedra
"""
  
# Number of new vertices on cut plane
const cut_nvert = [ 0, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 0]

# Number of resulting tets
const cut_ntets = [ 1, 4, 4, 6, 4, 6, 6, 4, 4, 6, 6, 4, 6, 4, 4, 1]

# Number of tets on negative side of the plane
# (Index of first positive tet = # tets - # negative tets + 1)
const cut_nntet = [ 1, 2, 2, 4, 2, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2]

# First point on intersection: cut_v1[4,16]
const cut_v1 = reshape([
  -1 -1 -1 -1
    1  1  1 -1
    2  2  2 -1
    1  2  1  2
    3  3  3 -1
    1  3  1  3
    2  3  2  3
    4  4  4 -1
    4  4  4 -1
    1  4  1  4
    2  4  2  4
    3  3  3 -1
    3  4  3  4
    2  2  2 -1
    1  1  1 -1
  -1 -1 -1 -1
]',(4,16))

# Second point on intersection: cut_v2[4,16]
const cut_v2 = reshape([
  -1 -1 -1 -1
    2  3  4 -1
    3  4  1 -1
    4  4  3  3
    4  1  2 -1
    4  4  2  2
    4  4  1  1
    1  2  3 -1
    1  2  3 -1
    3  3  2  2
    3  3  1  1
    4  1  2 -1
    2  2  1  1
    3  4  1 -1
    2  3  4 -1
  -1 -1 -1 -1
]',(4,16))

# Vertices in each tet: cut_vtet[4,6,16]
const cut_vtet = reshape([
  1  2  3  4   -1 -1 -1 -1   -1 -1 -1 -1   -1 -1 -1 -1   -1 -1 -1 -1   -1 -1 -1 -1
  5  7  6  1    6  2  3  4    4  2  5  6    5  6  7  4   -1 -1 -1 -1   -1 -1 -1 -1
  7  5  6  2    1  3  4  6    1  5  3  6    5  7  6  1   -1 -1 -1 -1   -1 -1 -1 -1
  5  8  6  2    5  7  8  1    5  1  8  2    5  6  8  4    5  8  7  3    5  8  3  4
  6  5  7  3    2  1  4  6    6  5  4  2    6  7  5  2   -1 -1 -1 -1   -1 -1 -1 -1
  5  6  8  3    5  8  7  1    5  8  1  3    5  8  6  4    5  7  8  2    5  8  4  2
  8  6  5  3    5  7  8  2    8  5  2  3    8  5  6  4    5  8  7  1    5  8  1  4
  1  2  3  7    1  2  7  6    5  7  6  1    5  6  7  4   -1 -1 -1 -1   -1 -1 -1 -1
  5  6  7  4    1  2  3  6    5  1  3  6    5  7  6  3   -1 -1 -1 -1   -1 -1 -1 -1
  5  8  6  4    5  7  8  1    5  8  4  1    5  6  8  3    5  8  7  2    5  8  2  3
  8  5  6  4    5  8  7  2    8  2  5  4    8  6  5  3    5  7  8  1    5  8  3  1
  1  4  2  7    4  1  6  7    6  7  5  4    6  5  7  3   -1 -1 -1 -1   -1 -1 -1 -1
  8  6  5  4    5  7  8  3    8  4  5  3    8  5  6  2    5  8  7  1    5  8  1  2
  3  4  1  7    7  6  3  4    7  6  5  3    7  5  6  2   -1 -1 -1 -1   -1 -1 -1 -1
  7  4  2  3    2  3  6  7    5  6  7  2    5  7  6  1   -1 -1 -1 -1   -1 -1 -1 -1
  1  2  3  4   -1 -1 -1 -1   -1 -1 -1 -1   -1 -1 -1 -1   -1 -1 -1 -1   -1 -1 -1 -1
]',(4,6,16))

# Number of triangles on cut plate
const cut_ntris = [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0]

# Vertices in each tri on cut plane 
cut_vtri = reshape([
  -1 -1 -1   -1 -1 -1 
  5  7  6   -1 -1 -1 
  5  6  7   -1 -1 -1 
  5  8  6    5  7  8 
  5  7  6   -1 -1 -1 
  5  6  8    5  8  7 
  5  8  6    5  7  8 
  5  7  6   -1 -1 -1 
  5  6  7   -1 -1 -1 
  5  8  6    5  7  8 
  5  6  8    5  8  7 
  5  6  7   -1 -1 -1 
  5  8  6    5  7  8 
  5  7  6   -1 -1 -1 
  5  6  7   -1 -1 -1 
  -1 -1 -1   -1 -1 -1
]',(3,2,16)) 

# Side of cut plane (used to update i,j,k)
const cut_side = reshape([
  1 -1 -1 -1 -1 -1 
  2  1  1  1 -1 -1 
  2  1  1  1 -1 -1 
  2  2  2  1  1  1 
  2  1  1  1 -1 -1 
  2  2  2  1  1  1 
  2  2  2  1  1  1 
  2  2  2  1 -1 -1 
  2  1  1  1 -1 -1 
  2  2  2  1  1  1 
  2  2  2  1  1  1 
  2  2  2  1 -1 -1 
  2  2  2  1  1  1 
  2  2  2  1 -1 -1 
  2  2  2  1 -1 -1 
  2 -1 -1 -1 -1 -1 
]',(6,  16))

function cell2tets(i,j,k,mesh)
  @unpack x,y,z = mesh
    # Cell vertices 
    p1=[x[i  ],y[j  ],z[k  ]]
    p2=[x[i+1],y[j  ],z[k  ]]
    p3=[x[i  ],y[j+1],z[k  ]]
    p4=[x[i+1],y[j+1],z[k  ]]
    p5=[x[i  ],y[j  ],z[k+1]]
    p6=[x[i+1],y[j  ],z[k+1]]
    p7=[x[i  ],y[j+1],z[k+1]]
    p8=[x[i+1],y[j+1],z[k+1]]
    # Make five tets 
    tets = Array{Float64}(undef,3,4,5)
    tets[:,1,1]=p1; tets[:,2,1]=p2; tets[:,3,1]=p4; tets[:,4,1]=p6
    tets[:,1,2]=p1; tets[:,2,2]=p4; tets[:,3,2]=p3; tets[:,4,2]=p7
    tets[:,1,3]=p1; tets[:,2,3]=p5; tets[:,3,3]=p6; tets[:,4,3]=p7
    tets[:,1,4]=p4; tets[:,2,4]=p7; tets[:,3,4]=p6; tets[:,4,4]=p8
    tets[:,1,5]=p1; tets[:,2,5]=p4; tets[:,3,5]=p7; tets[:,4,5]=p6
    return tets 
end


function cell2tets_withProject(i,j,k,u,v,w,dt,mesh)
  @unpack x,y,z = mesh
    # Cell vertices 
    p=Matrix{Float64}(undef,(3,8))
    p[:,1]=[x[i  ],y[j  ],z[k  ]]
    p[:,2]=[x[i+1],y[j  ],z[k  ]]
    p[:,3]=[x[i  ],y[j+1],z[k  ]]
    p[:,4]=[x[i+1],y[j+1],z[k  ]]
    p[:,5]=[x[i  ],y[j  ],z[k+1]]
    p[:,6]=[x[i+1],y[j  ],z[k+1]]
    p[:,7]=[x[i  ],y[j+1],z[k+1]]
    p[:,8]=[x[i+1],y[j+1],z[k+1]]
    # For each vertex
    I = Matrix{Int64}(undef,(3,8))
    for n=1:8
      # Perform semi-Lagrangian projection
      p[:,n] = project(p[:,n],i,j,k,u,v,w,dt,mesh)
      # Get cell index of projected vertex
      I[:,n] = pt2index(p[:,n],i,j,k,mesh)  
    end
    # Make five tets 
    tets = Array{Float64}(undef,3,4,5)
    tets[:,1,1]=p[:,1]; tets[:,2,1]=p[:,2]; tets[:,3,1]=p[:,4]; tets[:,4,1]=p[:,6]
    tets[:,1,2]=p[:,1]; tets[:,2,2]=p[:,4]; tets[:,3,2]=p[:,3]; tets[:,4,2]=p[:,7]
    tets[:,1,3]=p[:,1]; tets[:,2,3]=p[:,5]; tets[:,3,3]=p[:,6]; tets[:,4,3]=p[:,7]
    tets[:,1,4]=p[:,4]; tets[:,2,4]=p[:,7]; tets[:,3,4]=p[:,6]; tets[:,4,4]=p[:,8]
    tets[:,1,5]=p[:,1]; tets[:,2,5]=p[:,4]; tets[:,3,5]=p[:,7]; tets[:,4,5]=p[:,6]

    # Make five tets 
    inds = Array{Int64}(undef,3,4,5)
    inds[:,1,1]=I[:,1]; inds[:,2,1]=I[:,2]; inds[:,3,1]=I[:,4]; inds[:,4,1]=I[:,6]
    inds[:,1,2]=I[:,1]; inds[:,2,2]=I[:,4]; inds[:,3,2]=I[:,3]; inds[:,4,2]=I[:,7]
    inds[:,1,3]=I[:,1]; inds[:,2,3]=I[:,5]; inds[:,3,3]=I[:,6]; inds[:,4,3]=I[:,7]
    inds[:,1,4]=I[:,4]; inds[:,2,4]=I[:,7]; inds[:,3,4]=I[:,6]; inds[:,4,4]=I[:,8]
    inds[:,1,5]=I[:,1]; inds[:,2,5]=I[:,4]; inds[:,3,5]=I[:,7]; inds[:,4,5]=I[:,6]

    return tets, inds
end


"""
Computes volume of tet !
"""
function tet_vol(verts) 
  f1_6=0.16666666666666667
  a = Vector{Float64}(undef,3)
  b = Vector{Float64}(undef,3)
  c = Vector{Float64}(undef,3)
  for p=1:3
      a[p]=verts[p,1]-verts[p,4]
      b[p]=verts[p,2]-verts[p,4]
      c[p]=verts[p,3]-verts[p,4]
  end
  tet_vol = -f1_6 * 
       ( a[1] * ( b[2]*c[3] - c[2]*b[3] ) 
       - a[2] * ( b[1]*c[3] - c[1]*b[3] ) 
       + a[3] * ( b[1]*c[2] - c[1]*b[2] ) )
  return tet_vol
end


""" 
Cut tet by mesh then PLIC and return VF
"""
function cutTet(tet,ind,nx,ny,nz,D,mesh)
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack x,y,z = mesh

    # Determine max/min of indices
    maxi,maxj,maxk = maximum(ind,dims=2)
    mini,minj,mink = minimum(ind,dims=2)

    # Allocate work arrays 
    vert     = Array{Float64}(undef,3,8)
    vert_ind = Array{Int64}(undef,3,8,2)
    d = Array{Float64}(undef,4)    
    newtet = Array{Float64}(undef,3,4)

    # Cut by x-planes
    if maxi > mini
        dir=1
        cut_ind=maxi 
        for n=1:4 
            d[n]=tet[1,n] - x[cut_ind]
        end

    # Cut by y-planes
    elseif maxj > minj
        dir=2
        cut_ind=maxj
        for n=1:4 
            d[n]=tet[2,n] - y[cut_ind]
        end

    # Cut by z-planes
    elseif maxk > mink
        dir=3
        cut_ind=maxk
        for n=1:4 
            d[n]=tet[3,n] - z[cut_ind]
        end

    # Cut by PLIC and compute output
    else
        vol =0.0
        vLiq=0.0
        # Copy vertices
        for n=1:4
            for p=1:3
                vert[p,n]=tet[p,n]
            end
        end
        # Get index
        i=ind[1,1];  j=ind[2,1];  k=ind[3,1];
        # Calculate distance from each vertex to cut plane
        for n=1:4
            d[n]=nx[i,j,k]*vert[1,n]+ny[i,j,k]*vert[2,n]+nz[i,j,k]*vert[3,n]-D[i,j,k]
        end
        # Handle zero distances
        npos = count(d.>0.0)
        nneg = count(d.<0.0)
        d[d.==0] .= eps()*( npos > nneg ? 1.0 : -1.0 )
        # Determine case
        case=(
        1+Int(0.5+0.5*sign(d[1]))+
        2*Int(0.5+0.5*sign(d[2]))+
        4*Int(0.5+0.5*sign(d[3]))+
        8*Int(0.5+0.5*sign(d[4])))

        # Create interpolated vertices on cut plane
        for n=1:cut_nvert[case]
            v1 = cut_v1[n,case]; v2 = cut_v2[n,case]
            mu=min(1.0,max(0.0, -d[v1] /̂ (d[v2]-d[v1])))
            vert[:,4+n]=(1.0-mu)*vert[:,v1]+mu*vert[:,v2]
        end
        # Create new tets on liquid side
        for n=cut_ntets[case]:-1:cut_nntet[case]
            # Form tet
            for t = 1:4
              vt = cut_vtet[t,n,case]
              for p = 1:3
                  newtet[p,t] = vert[p,vt]
              end
            end
            # Compute volume
            tetVol = tet_vol(newtet)
            # Update volumes in this cell
            vLiq += tetVol
            vol  += tetVol
        end
        # Create new tets on gas side
        for n=1:cut_nntet[case]-1
            # Form tet
            for t = 1:4
                vt = cut_vtet[t,n,case]
                for p = 1:3
                    newtet[p,t] = vert[p,vt]
                end
            end
            # Compute volume
            tetVol = tet_vol(newtet)
            # Update volumes in this cell
            vol += tetVol
        end

        return vol, vLiq
    end
    
    # Cut by plane
    # -------------
    # Handle zero distances
    npos = count(d.>0.0)
    nneg = count(d.<0.0)
    d[d.==0] .= eps()*( npos > nneg ? 1.0 : -1.0 )
    # Determine case
    case=(
    1+Int(0.5+0.5*sign(d[1]))+
    2*Int(0.5+0.5*sign(d[2]))+
    4*Int(0.5+0.5*sign(d[3]))+
    8*Int(0.5+0.5*sign(d[4])))
    # Get vertices and indices of tet
    for n=1:4
        vert[      :,n  ]=tet[:,n]
        vert_ind[  :,n,1]=ind[:,n]
        vert_ind[  :,n,2]=ind[:,n]
        vert_ind[dir,n,1]=min(vert_ind[dir,n,1],cut_ind-1)
        vert_ind[dir,n,2]=max(vert_ind[dir,n,1],cut_ind  )
    end
    # Create interpolated vertices on cut plane
    for n=1:cut_nvert[case]
        v1 = cut_v1[n,case]; v2 = cut_v2[n,case]
        mu=min(1.0,max(0.0, -d[v1] /̂ (d[v2]-d[v1])))
        vert[:,4+n]=(1.0-mu)*vert[:,v1]+mu*vert[:,v2]
        # Get index for interpolated vertex
        i=vert_ind[1,n,1]
        j=vert_ind[2,n,1]
        k=vert_ind[3,n,1]
        vert_ind[:,4+n,1] = pt2index(vert[:,4+n],i,j,k,mesh)
        # Enforce boundedness
        vert_ind[:,4+n,1]=max(vert_ind[:,4+n,1],min(vert_ind[:,v1,1],vert_ind[:,v2,1]))
        vert_ind[:,4+n,1]=min(vert_ind[:,4+n,1],max(vert_ind[:,v1,1],vert_ind[:,v2,1]))
        # Set +/- indices in cut direction
        vert_ind[:,4+n,2]=vert_ind[:,4+n,1]
        vert_ind[dir,4+n,1]=cut_ind-1
        vert_ind[dir,4+n,2]=cut_ind
    end
    # Create new tets
    vol  = 0.0
    vLiq = 0.0
    for n=1:cut_ntets[case]
       # Form new tet
       for nn=1:4
          tet[:,nn]=vert[:,cut_vtet[nn,n,case]]
          ind[:,nn]=vert_ind[:,cut_vtet[nn,n,case],cut_side[n,case]]
       end
       # Cut new tet by next plnae
       tetVol, tetVLiq = cutTet(tet,ind,nx,ny,nz,D,mesh)
       # Accumulate quantities
       vol += tetVol
       vLiq += tetVLiq
    end

    return vol,vLiq
end 