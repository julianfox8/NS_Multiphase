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