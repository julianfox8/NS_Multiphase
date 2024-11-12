""" 
Look-up tables for cutting triangles
"""

# Number of new vertices on cut plane
const cutTri_nvert = SVector{8}([0, 2, 2, 2, 2, 2, 2, 0])

# Number of resulting tris on positive side
const cutTri_np = SVector{8}([0, 1, 1, 2, 1, 2, 2, 1])

# Number of resulting tris on negative side
const cutTri_nn = [1, 2, 2, 1, 2, 1, 1, 0]

# First point on intersection
const cutTri_v1 = SMatrix{2,8}([
        -1 -1
        1 1
        2 2
        3 3
        3 3
        2 2
        1 1
        -1 -1
    ]')

# Second point on intersection
const cutTri_v2 = SMatrix{2,8}([
        -1 -1
        2 3
        1 3
        1 2
        1 2
        1 3
        2 3
        -1 -1
    ]')

# Vertices in each tri
const cutTri_v = SArray{Tuple{3,3,8}}([
        1 2 3 -1 -1 -1 -1 -1 -1
        1 4 5 2 5 4 2 3 5
        2 5 4 1 4 5 1 5 3
        1 2 5 1 5 4 3 4 5
        3 4 5 1 2 5 1 5 4
        1 4 5 1 5 3 2 5 4
        2 3 5 2 5 4 1 4 5
        1 2 3 -1 -1 -1 -1 -1 -1
    ]')

""" 
Look-up tables for cutting tetrahedra
"""

# Number of new vertices on cut plane
const cut_nvert = SVector{16}([0, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 0])

# Number of resulting tets
const cut_ntets = SVector{16}([1, 4, 4, 6, 4, 6, 6, 4, 4, 6, 6, 4, 6, 4, 4, 1])

# Number of tets on negative side of the plane
# (Index of first positive tet = # tets - # negative tets + 1)
const cut_nntet = SVector{16}([1, 2, 2, 4, 2, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2])

# First point on intersection: cut_v1[4,16]
const cut_v1 = SArray{Tuple{4,16}}([
        -1 -1 -1 -1
        1 1 1 -1
        2 2 2 -1
        1 2 1 2
        3 3 3 -1
        1 3 1 3
        2 3 2 3
        4 4 4 -1
        4 4 4 -1
        1 4 1 4
        2 4 2 4
        3 3 3 -1
        3 4 3 4
        2 2 2 -1
        1 1 1 -1
        -1 -1 -1 -1
    ]')

# Second point on intersection: cut_v2[4,16]
const cut_v2 = SArray{Tuple{4,16}}([
        -1 -1 -1 -1
        2 3 4 -1
        3 4 1 -1
        4 4 3 3
        4 1 2 -1
        4 4 2 2
        4 4 1 1
        1 2 3 -1
        1 2 3 -1
        3 3 2 2
        3 3 1 1
        4 1 2 -1
        2 2 1 1
        3 4 1 -1
        2 3 4 -1
        -1 -1 -1 -1
    ]')

# Vertices in each tet: cut_vtet[4,6,16]
const cut_vtet = SArray{Tuple{4,6,16}}([
        1 2 3 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
        5 7 6 1 6 2 3 4 4 2 5 6 5 6 7 4 -1 -1 -1 -1 -1 -1 -1 -1
        7 5 6 2 1 3 4 6 1 5 3 6 5 7 6 1 -1 -1 -1 -1 -1 -1 -1 -1
        5 8 6 2 5 7 8 1 5 1 8 2 5 6 8 4 5 8 7 3 5 8 3 4
        6 5 7 3 2 1 4 6 6 5 4 2 6 7 5 2 -1 -1 -1 -1 -1 -1 -1 -1
        5 6 8 3 5 8 7 1 5 8 1 3 5 8 6 4 5 7 8 2 5 8 4 2
        8 6 5 3 5 7 8 2 8 5 2 3 8 5 6 4 5 8 7 1 5 8 1 4
        1 2 3 7 1 2 7 6 5 7 6 1 5 6 7 4 -1 -1 -1 -1 -1 -1 -1 -1
        5 6 7 4 1 2 3 6 5 1 3 6 5 7 6 3 -1 -1 -1 -1 -1 -1 -1 -1
        5 8 6 4 5 7 8 1 5 8 4 1 5 6 8 3 5 8 7 2 5 8 2 3
        8 5 6 4 5 8 7 2 8 2 5 4 8 6 5 3 5 7 8 1 5 8 3 1
        1 4 2 7 4 1 6 7 6 7 5 4 6 5 7 3 -1 -1 -1 -1 -1 -1 -1 -1
        8 6 5 4 5 7 8 3 8 4 5 3 8 5 6 2 5 8 7 1 5 8 1 2
        3 4 1 7 7 6 3 4 7 6 5 3 7 5 6 2 -1 -1 -1 -1 -1 -1 -1 -1
        7 4 2 3 2 3 6 7 5 6 7 2 5 7 6 1 -1 -1 -1 -1 -1 -1 -1 -1
        1 2 3 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    ]')

# Number of triangles on cut plate
const cut_ntris = SVector{16}([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0])

# Vertices in each tri on cut plane 
cut_vtri = SArray{Tuple{3,2,16}}([
        -1 -1 -1 -1 -1 -1
        5 7 6 -1 -1 -1
        5 6 7 -1 -1 -1
        5 8 6 5 7 8
        5 7 6 -1 -1 -1
        5 6 8 5 8 7
        5 8 6 5 7 8
        5 7 6 -1 -1 -1
        5 6 7 -1 -1 -1
        5 8 6 5 7 8
        5 6 8 5 8 7
        5 6 7 -1 -1 -1
        5 8 6 5 7 8
        5 7 6 -1 -1 -1
        5 6 7 -1 -1 -1
        -1 -1 -1 -1 -1 -1
    ]')

# Side of cut plane (used to update i,j,k)
const cut_side = SArray{Tuple{6,16}}([
        1 -1 -1 -1 -1 -1
        2 1 1 1 -1 -1
        2 1 1 1 -1 -1
        2 2 2 1 1 1
        2 1 1 1 -1 -1
        2 2 2 1 1 1
        2 2 2 1 1 1
        2 2 2 1 -1 -1
        2 1 1 1 -1 -1
        2 2 2 1 1 1
        2 2 2 1 1 1
        2 2 2 1 -1 -1
        2 2 2 1 1 1
        2 2 2 1 -1 -1
        2 2 2 1 -1 -1
        2 -1 -1 -1 -1 -1
    ]')

"""
Classify cell as either even (true) or odd (false) based on index
- used to set ordering of vertices to make discretization of faces consistent
"""
function oddEvenCell(i,j,k)
    return (i+j+k) % 2 == 0
end

"""
Determine verts for a cell in ordering that triangulates faces consistently
    - verts is a 3x8 array and that is filled by this function
"""
function cell2verts!(verts,i,j,k,mesh)
    @unpack x, y, z = mesh

    # Determine verts for odd or even cell
    if oddEvenCell(i,j,k)
        tetsign = +1.0
        verts[:,1] = [x[i  ], y[j  ], z[k  ]]
        verts[:,2] = [x[i+1], y[j  ], z[k  ]]
        verts[:,3] = [x[i  ], y[j+1], z[k  ]]
        verts[:,4] = [x[i+1], y[j+1], z[k  ]]
        verts[:,5] = [x[i  ], y[j  ], z[k+1]]
        verts[:,6] = [x[i+1], y[j  ], z[k+1]]
        verts[:,7] = [x[i  ], y[j+1], z[k+1]]
        verts[:,8] = [x[i+1], y[j+1], z[k+1]]
    else
        tetsign = -1.0
        verts[:,1] = [x[i+1], y[j  ], z[k  ]]  
        verts[:,2] = [x[i  ], y[j  ], z[k  ]]  
        verts[:,3] = [x[i+1], y[j+1], z[k  ]]  
        verts[:,4] = [x[i  ], y[j+1], z[k  ]]  
        verts[:,5] = [x[i+1], y[j  ], z[k+1]] 
        verts[:,6] = [x[i  ], y[j  ], z[k+1]]
        verts[:,7] = [x[i+1], y[j+1], z[k+1]]
        verts[:,8] = [x[i  ], y[j+1], z[k+1]]
    end
    return tetsign
end  

""" 
Make 5 tets out of a polyhedron represented by 8 vertices
    - tets is is a 3x4x5 array that is filled by this function
"""
function verts2tets!(tets,p)
    # Make five tets 
    tets[:, 1, 1] = p[:, 1]
    tets[:, 2, 1] = p[:, 2]
    tets[:, 3, 1] = p[:, 4]
    tets[:, 4, 1] = p[:, 6]
    tets[:, 1, 2] = p[:, 1]
    tets[:, 2, 2] = p[:, 4]
    tets[:, 3, 2] = p[:, 3]
    tets[:, 4, 2] = p[:, 7]
    tets[:, 1, 3] = p[:, 1]
    tets[:, 2, 3] = p[:, 5]
    tets[:, 3, 3] = p[:, 6]
    tets[:, 4, 3] = p[:, 7]
    tets[:, 1, 4] = p[:, 4]
    tets[:, 2, 4] = p[:, 7]
    tets[:, 3, 4] = p[:, 6]
    tets[:, 4, 4] = p[:, 8]
    tets[:, 1, 5] = p[:, 1]
    tets[:, 2, 5] = p[:, 4]
    tets[:, 3, 5] = p[:, 7]
    tets[:, 4, 5] = p[:, 6]
    return nothing
end

        
"""
Create 5 tets to represent computational cell
    - verts is a 3x8   array that is filled by this function
    - tets  is a 3x4x5 array that is filled by this function
    - options:
        - project: if true then verts are projected - uf,vf,wf, and dt are required
        - indices: if true then indices of projected verts are computed 
            - ind (3x4x5) array is filled by this function
            - I (3x8 work array) is required
"""
function cell2tets!(verts, tets, i, j, k, param, mesh; 
    project_verts=false, uf=nothing, vf=nothing, wf=nothing, dt=nothing, 
    compute_indices=false, inds=nothing, vInds=nothing
    )
    @unpack x, y, z = mesh
    
    # Cell vertices 
    tetsign = cell2verts!(verts,i,j,k,mesh)

    # Project if requested 
    if project_verts
        for n=1:8
            project!(@view(verts[:,n]),i,j,k,uf,vf,wf,dt,param,mesh)
        end
    end

    # Make five tets 
    verts2tets!(tets,verts)

    # Compute indices if requested
    if compute_indices
        for n=1:8
            pt2index!(@view(vInds[:,n]),@view(verts[:,n]),i,j,k,mesh)
        end 
        verts2tets!(inds,vInds)
    end
    
    return tetsign
end

""" 
Determine volume of semi-Lagrangian flux on face
- consistent with cell2tets! discretization  
- verts is a 3x8 work array 
- tets is a 3x4x5 work array 
"""
function SLfluxVol(dir,i,j,k,verts,tets,uf,vf,wf,dt,param,mesh)
    @unpack x,y,z = mesh

    # Form verts on flux volume 
    if dir == 1 # x
        if oddEvenCell(i,j,k)
            tetsign = +1.0
            verts[:,1] = [x[i  ], y[j  ], z[k  ]]; project!(@view(verts[:,1]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,2] = [x[i  ], y[j  ], z[k  ]];
            verts[:,3] = [x[i  ], y[j+1], z[k  ]]; project!(@view(verts[:,3]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,4] = [x[i  ], y[j+1], z[k  ]];
            verts[:,5] = [x[i  ], y[j  ], z[k+1]]; project!(@view(verts[:,5]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,6] = [x[i  ], y[j  ], z[k+1]];
            verts[:,7] = [x[i  ], y[j+1], z[k+1]]; project!(@view(verts[:,7]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,8] = [x[i  ], y[j+1], z[k+1]];
        else
            tetsign = -1.0
            verts[:,1] = [x[i  ], y[j  ], z[k  ]];
            verts[:,2] = [x[i  ], y[j  ], z[k  ]]; project!(@view(verts[:,2]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,3] = [x[i  ], y[j+1], z[k  ]];
            verts[:,4] = [x[i  ], y[j+1], z[k  ]]; project!(@view(verts[:,4]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,5] = [x[i  ], y[j  ], z[k+1]];
            verts[:,6] = [x[i  ], y[j  ], z[k+1]]; project!(@view(verts[:,6]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,7] = [x[i  ], y[j+1], z[k+1]];
            verts[:,8] = [x[i  ], y[j+1], z[k+1]]; project!(@view(verts[:,8]),i,j,k,uf,vf,wf,dt,param,mesh)
        end
    elseif dir == 2 # y
        if oddEvenCell(i,j,k)
            tetsign = +1.0
            verts[:,1] = [x[i  ], y[j  ], z[k  ]]; project!(@view(verts[:,1]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,2] = [x[i+1], y[j  ], z[k  ]]; project!(@view(verts[:,2]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,3] = [x[i  ], y[j  ], z[k  ]]; 
            verts[:,4] = [x[i+1], y[j  ], z[k  ]];
            verts[:,5] = [x[i  ], y[j  ], z[k+1]]; project!(@view(verts[:,5]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,6] = [x[i+1], y[j  ], z[k+1]]; project!(@view(verts[:,6]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,7] = [x[i  ], y[j  ], z[k+1]]; 
            verts[:,8] = [x[i+1], y[j  ], z[k+1]];
        else
            tetsign = -1.0
            verts[:,1] = [x[i+1], y[j  ], z[k  ]]; project!(@view(verts[:,1]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,2] = [x[i  ], y[j  ], z[k  ]]; project!(@view(verts[:,2]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,3] = [x[i+1], y[j  ], z[k  ]]; 
            verts[:,4] = [x[i  ], y[j  ], z[k  ]];
            verts[:,5] = [x[i+1], y[j  ], z[k+1]]; project!(@view(verts[:,5]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,6] = [x[i  ], y[j  ], z[k+1]]; project!(@view(verts[:,6]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,7] = [x[i+1], y[j  ], z[k+1]]; 
            verts[:,8] = [x[i  ], y[j  ], z[k+1]];
        end
    elseif dir == 3 # z
        if oddEvenCell(i,j,k)
            tetsign = +1.0
            verts[:,1] = [x[i  ], y[j  ], z[k  ]]; project!(@view(verts[:,1]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,2] = [x[i+1], y[j  ], z[k  ]]; project!(@view(verts[:,2]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,3] = [x[i  ], y[j+1], z[k  ]]; project!(@view(verts[:,3]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,4] = [x[i+1], y[j+1], z[k  ]]; project!(@view(verts[:,4]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,5] = [x[i  ], y[j  ], z[k  ]]; 
            verts[:,6] = [x[i+1], y[j  ], z[k  ]]; 
            verts[:,7] = [x[i  ], y[j+1], z[k  ]]; 
            verts[:,8] = [x[i+1], y[j+1], z[k  ]];
        else
            tetsign = -1.0
            verts[:,1] = [x[i+1], y[j  ], z[k  ]]; project!(@view(verts[:,1]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,2] = [x[i  ], y[j  ], z[k  ]]; project!(@view(verts[:,2]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,3] = [x[i+1], y[j+1], z[k  ]]; project!(@view(verts[:,3]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,4] = [x[i  ], y[j+1], z[k  ]]; project!(@view(verts[:,4]),i,j,k,uf,vf,wf,dt,param,mesh)
            verts[:,5] = [x[i+1], y[j  ], z[k  ]]; 
            verts[:,6] = [x[i  ], y[j  ], z[k  ]]; 
            verts[:,7] = [x[i+1], y[j+1], z[k  ]]; 
            verts[:,8] = [x[i  ], y[j+1], z[k  ]];
        end
    else
        error("Unknown direction in SLfluxVol")
    end
    
    # Make five tets 
    verts2tets!(tets,verts)

     # Compute flux volume 
     flux_vol = tets_vol(tets) * tetsign
    
     return flux_vol
end

"""
Add correction tets on to semi-Lagrangian cell
"""
function add_correction_tets(ntets,verts,tets, inds, i, j, k, uf, vf, wf, dt, param, mesh)
    @unpack dx,dy,dz = mesh

    """ 
    Add correction tets on faces 
        - verts is a 3x8 work array 
    """
    function add_correction_tets_x(tets,verts,i,j,k)
        # Create seperate tets work array to not impact exisiting tets 
        tets_work  = Array{Float64}(undef, 3, 4, 5)
        # Constrct SL flux volume - compute volume and set verts 
        flux_vol = SLfluxVol(1,i,j,k,verts,tets_work,uf,vf,wf,dt,param,mesh)
        # Compute required volume 
        divg_vol = dt*dy*dz*uf[i,j,k]
        # Create array with 2 additional tets appended 
        ntets = size(tets,3)
        newtets = Array{eltype(tets)}(undef, 3, 4, ntets+2)
        newtets[:,:,1:ntets] = tets
        # Construct correction tets
        if oddEvenCell(i,j,k)
            newtets[:,:,ntets+1:ntets+2] = correction_tets_x(verts[:,5],verts[:,7],verts[:,3],verts[:,1],+(divg_vol - flux_vol))
        else
            newtets[:,:,ntets+1:ntets+2] = correction_tets_x(verts[:,8],verts[:,6],verts[:,2],verts[:,4],-(divg_vol - flux_vol))
        end
        return newtets
    end
    function add_correction_tets_y(tets,verts,i,j,k)
        # Create seperate tets work array to not impact exisiting tets 
        tets_work  = Array{Float64}(undef, 3, 4, 5)
        # Constrct SL flux volume - compute volume and set verts 
        flux_vol = SLfluxVol(2,i,j,k,verts,tets_work,uf,vf,wf,dt,param,mesh)
        # Compute required volume 
        divg_vol = dt*dx*dz*vf[i,j,k]
        # Create array with 2 additional tets appended 
        ntets = size(tets,3)
        newtets = Array{eltype(tets)}(undef, 3, 4, ntets+2)
        newtets[:,:,1:ntets] = tets
        # Construct correction tets
        if oddEvenCell(i,j,k)
            newtets[:,:,ntets+1:ntets+2] = correction_tets_y(verts[:,2],verts[:,6],verts[:,5],verts[:,1],+(divg_vol - flux_vol))
        else
            newtets[:,:,ntets+1:ntets+2] = correction_tets_y(verts[:,2],verts[:,6],verts[:,5],verts[:,1],-(divg_vol - flux_vol))
        end
        return newtets
    end
    function add_correction_tets_z(tets,verts,i,j,k)
        # Create seperate tets work array to not impact exisiting tets 
        tets_work  = Array{Float64}(undef, 3, 4, 5)
        # Constrct SL flux volume - compute volume and set verts 
        flux_vol = SLfluxVol(3,i,j,k,verts,tets_work,uf,vf,wf,dt,param,mesh)
        # Compute required volume 
        divg_vol = dt*dx*dy*wf[i,j,k]
        # Create array with 2 additional tets appended 
        ntets = size(tets,3)
        newtets = Array{eltype(tets)}(undef, 3, 4, ntets+2)
        newtets[:,:,1:ntets] = tets
        # Construct correction tets
        if oddEvenCell(i,j,k)
            newtets[:,:,ntets+1:ntets+2] = correction_tets_z(verts[:,2],verts[:,1],verts[:,3],verts[:,4],+(divg_vol - flux_vol))
        else
            newtets[:,:,ntets+1:ntets+2] = correction_tets_z(verts[:,2],verts[:,1],verts[:,3],verts[:,4],-(divg_vol - flux_vol))
        end
        return newtets
    end

    function correction_tets_x(a,b,c,d,vol)
        tets = Array{eltype(tets)}(undef, 3, 4, 2)
        e = 0.25*(a+b+c+d)
        e[1] = ( (6.0*vol + a[1]*b[2]*d[3] - a[1]*b[3]*d[2] - a[2]*b[1]*d[3] + a[2]*b[3]*d[1] + a[3]*b[1]*d[2]
               - a[3]*b[2]*d[1] - a[1]*b[2]*e[3] + a[1]*b[3]*e[2] + a[2]*b[1]*e[3] - a[3]*b[1]*e[2] + b[1]*c[2]*d[3] 
               - b[1]*c[3]*d[2] - b[2]*c[1]*d[3] + b[2]*c[3]*d[1] + b[3]*c[1]*d[2] - b[3]*c[2]*d[1] + a[1]*d[2]*e[3] 
               - a[1]*d[3]*e[2] - a[2]*d[1]*e[3] + a[3]*d[1]*e[2] - b[1]*c[2]*e[3] + b[1]*c[3]*e[2] + b[2]*c[1]*e[3] 
               - b[3]*c[1]*e[2] - c[1]*d[2]*e[3] + c[1]*d[3]*e[2] + c[2]*d[1]*e[3] - c[3]*d[1]*e[2]) 
               / (a[2]*b[3] - a[3]*b[2] - a[2]*d[3] + a[3]*d[2] + b[2]*c[3] - b[3]*c[2] + c[2]*d[3] - c[3]*d[2]) )
        tets[:,1,1]=a; tets[:,2,1]=b; tets[:,3,1]=d; tets[:,4,1]=e;
        tets[:,1,2]=b; tets[:,2,2]=c; tets[:,3,2]=d; tets[:,4,2]=e;
        return tets
    end
    function correction_tets_y(a,b,c,d,vol)
        tets = Array{eltype(tets)}(undef, 3, 4, 2)
        e=0.25*(a+b+c+d)
        e[2] = ( -(6.0*vol + a[1]*b[2]*d[3] - a[1]*b[3]*d[2] - a[2]*b[1]*d[3] + a[2]*b[3]*d[1] + a[3]*b[1]*d[2] 
               - a[3]*b[2]*d[1] - a[1]*b[2]*e[3] + a[2]*b[1]*e[3] - a[2]*b[3]*e[1] + a[3]*b[2]*e[1] + b[1]*c[2]*d[3] 
               - b[1]*c[3]*d[2] - b[2]*c[1]*d[3] + b[2]*c[3]*d[1] + b[3]*c[1]*d[2] - b[3]*c[2]*d[1] + a[1]*d[2]*e[3] 
               - a[2]*d[1]*e[3] + a[2]*d[3]*e[1] - a[3]*d[2]*e[1] - b[1]*c[2]*e[3] + b[2]*c[1]*e[3] - b[2]*c[3]*e[1] 
               + b[3]*c[2]*e[1] - c[1]*d[2]*e[3] + c[2]*d[1]*e[3] - c[2]*d[3]*e[1] + c[3]*d[2]*e[1]) 
               / (a[1]*b[3] - a[3]*b[1] - a[1]*d[3] + a[3]*d[1] + b[1]*c[3] - b[3]*c[1] + c[1]*d[3] - c[3]*d[1]) )
        tets[:,1,1]=a; tets[:,2,1]=b; tets[:,3,1]=d; tets[:,4,1]=e;
        tets[:,1,2]=b; tets[:,2,2]=c; tets[:,3,2]=d; tets[:,4,2]=e;
        return tets
    end
    function correction_tets_z(a,b,c,d,vol)
        tets = Array{eltype(tets)}(undef, 3, 4, 2)
        e=0.25*(a+b+c+d)
        e[3] = ( (6.0*vol + a[1]*b[2]*d[3] - a[1]*b[3]*d[2] - a[2]*b[1]*d[3] + a[2]*b[3]*d[1] + a[3]*b[1]*d[2] 
               - a[3]*b[2]*d[1] + a[1]*b[3]*e[2] - a[2]*b[3]*e[1] - a[3]*b[1]*e[2] + a[3]*b[2]*e[1] + b[1]*c[2]*d[3] 
               - b[1]*c[3]*d[2] - b[2]*c[1]*d[3] + b[2]*c[3]*d[1] + b[3]*c[1]*d[2] - b[3]*c[2]*d[1] - a[1]*d[3]*e[2] 
               + a[2]*d[3]*e[1] + a[3]*d[1]*e[2] - a[3]*d[2]*e[1] + b[1]*c[3]*e[2] - b[2]*c[3]*e[1] - b[3]*c[1]*e[2] 
               + b[3]*c[2]*e[1] + c[1]*d[3]*e[2] - c[2]*d[3]*e[1] - c[3]*d[1]*e[2] + c[3]*d[2]*e[1]) 
               / (a[1]*b[2] - a[2]*b[1] - a[1]*d[2] + a[2]*d[1] + b[1]*c[2] - b[2]*c[1] + c[1]*d[2] - c[2]*d[1]) )
        tets[:,1,1]=a; tets[:,2,1]=b; tets[:,3,1]=d; tets[:,4,1]=e;
        tets[:,1,2]=b; tets[:,2,2]=c; tets[:,3,2]=d; tets[:,4,2]=e;
        return tets
    end

    # Add 2 tets to correct flux on each face 
    new_tets = tets[:,:,1:5]
    new_tets = add_correction_tets_x(new_tets,verts,i  ,j,k); ntets += 2
    new_tets = add_correction_tets_x(new_tets,verts,i+1,j,k); ntets += 2
    new_tets = add_correction_tets_y(new_tets,verts,i,j  ,k); ntets += 2
    new_tets = add_correction_tets_y(new_tets,verts,i,j+1,k); ntets += 2
    new_tets = add_correction_tets_z(new_tets,verts,i,j,k  ); ntets += 2
    new_tets = add_correction_tets_z(new_tets,verts,i,j,k+1); ntets += 2
    # Compute indices of new tets 
    new_inds = Array{eltype(inds)}(undef, size(new_tets))
    new_inds[:,:,1:5] = inds[:,:,1:5] # transfer original indices 
    for n = 6:size(new_tets,3)
        for v = 1:4
            pt2index!(@view(new_inds[:,v,n]),@view(new_tets[:,v,n]),i,j,k,mesh)
        end
    end
    return new_tets, new_inds, ntets
end

"""
Computes volume of tet
"""
function tet_vol(verts)
    a(p) = verts[p, 1] - verts[p, 4]
    b(p) = verts[p, 2] - verts[p, 4]
    c(p) = verts[p, 3] - verts[p, 4]
    tet_vol = -1.0/6.0 *
        ( a(1) * (b(2) * c(3) - c(2) * b(3))
        - a(2) * (b(1) * c(3) - c(1) * b(3))
        + a(3) * (b(1) * c(2) - c(1) * b(2)))
    if tet_vol == isnan
        error("tet_vol is nan")
    end
    return tet_vol
end

function tets_vol(tets)
    vol= 0.0
    for n=1:size(tets,3)
        vol += tet_vol(tets[:,:,n])
    end
    return vol
end
""" 
Determine cut case based on sign of d's
"""
function d2case(d)
    # Handle zero distances
    npos = count(>(0),d)
    nneg = count(<(0),d)
    @inbounds for n=eachindex(d)
        if abs(d[n])<eps()
            d[n] = npos > nneg ? eps() : -eps()
        end
    end
    # Determine case
    case = 1
    @inbounds for n=eachindex(d)
        case += 2^(n-1) * Int(0.5 + 0.5 * sign(d[n]))
    end
    return case
end

""" 
Cut tet by mesh then PLIC and return VF
"""
function cutTet(tet, ind, u, v, w, xdone, ydone, zdone, nx, ny, nz, D, mesh,lvl,vert,vert_ind,d,newtet)
    @unpack imino_, imaxo_, jmino_, jmaxo_, kmino_, kmaxo_ = mesh
    @unpack x, y, z = mesh

    id = Threads.threadid()

    # Cut by x-planes
    if !xdone
        if (maxi = maximum(ind[1, :])) > minimum(ind[1, :])
            dir = 1
            cut_ind = maxi
            for n = 1:4
                d[n,id] = tet[1, n] - x[cut_ind]
            end
        else
            xdone = true
            return tetVol, tetvLiq, tetvU, tetvV, tetvW, maxlvl = cutTet(tet, ind, u, v, w, xdone, ydone, zdone, nx, ny, nz, D, mesh,lvl,vert,vert_ind,d,newtet)
        end
        # Cut by y-planes
    elseif !ydone
        if (maxj = maximum(ind[2, :])) > minimum(ind[2, :])
            dir = 2
            cut_ind = maxj
            for n = 1:4
                d[n,id] = tet[2, n] - y[cut_ind]
            end
        else
            ydone = true
            return tetVol, tetvLiq, tetvU, tetvV, tetvW, maxlvl = cutTet(tet, ind, u, v, w, xdone, ydone, zdone, nx, ny, nz, D, mesh,lvl,vert,vert_ind,d,newtet)
        end
        # Cut by z-planes
    elseif !zdone
        if (maxk = maximum(ind[3, :])) > minimum(ind[3, :])
            dir = 3
            cut_ind = maxk
            for n = 1:4
                d[n,id] = tet[3, n] - z[cut_ind]
            end
        else
            zdone = true
            return tetVol, tetvLiq, tetvU, tetvV, tetvW, maxlvl = cutTet(tet, ind, u, v, w, xdone, ydone, zdone, nx, ny, nz, D, mesh,lvl,vert,vert_ind,d,newtet)
        end
        # Cut by PLIC and compute output
    else
        vol = 0.0
        vLiq = 0.0
        vU = 0.0
        vV = 0.0
        vW = 0.0
        # Copy vertices
        for n = 1:4
            for p = 1:3
                vert[p, n,lvl,id] = tet[p, n]
            end
        end
        # Get index
        i = ind[1, 1]
        j = ind[2, 1]
        k = ind[3, 1]
        # Calculate distance from each vertex to cut plane
        for n = 1:4
            d[n,id] = (
                nx[i, j, k] * vert[1, n,lvl,id] + 
                ny[i, j, k] * vert[2, n,lvl,id] + 
                nz[i, j, k] * vert[3, n,lvl,id] - D[i, j, k] )
        end
        # Compute cut case 
        case = d2case(d[:,id])
        # Create interpolated vertices on cut plane
        for n = 1:cut_nvert[case]
            v1 = cut_v1[n, case]
            v2 = cut_v2[n, case]
            mu = min(1.0, max(0.0, -d[v1,id] /̂ (d[v2,id] - d[v1,id])))
            vert[:, 4+n,lvl,id] = (1.0 - mu) * vert[:, v1,lvl,id] + mu * vert[:, v2,lvl,id]
        end
        # Create new tets on liquid side
        for n = cut_ntets[case]:-1:cut_nntet[case]
            # Form tet
            for t = 1:4
                vt = cut_vtet[t, n, case]
                for p = 1:3
                    newtet[p, t,id] = vert[p, vt,lvl,id]
                end
            end
            # Compute volume
            tetVol = tet_vol(newtet[:,:,id])
            # Update volumes in this cell
            vol += tetVol
            vLiq += tetVol
            vU += tetVol*u[i,j,k]
            vV += tetVol*v[i,j,k]
            vW += tetVol*w[i,j,k]
            
        end
        # Create new tets on gas side
        for n = 1:cut_nntet[case]-1
            # Form tet
            for t = 1:4
                vt = cut_vtet[t, n, case]
                for p = 1:3
                    newtet[p, t,id] = vert[p, vt,lvl,id]
                end
            end
            # Compute volume
            tetVol = tet_vol(newtet[:,:,id])
            # Update volumes in this cell
            vol += tetVol
            vU += tetVol*u[i,j,k]
            vV += tetVol*v[i,j,k]
            vW += tetVol*w[i,j,k]
        end

        return vol, vLiq, vU, vV, vW, lvl
    end

    # Cut by plane
    # -------------
    case = d2case(d[:,id])
    # Get vertices and indices of tet
    for n = 1:4
        for p = 1:3
            vert[p, n,lvl,id] = tet[p, n]
            vert_ind[p, n, 1,lvl,id] = ind[p, n]
            vert_ind[p, n, 2,lvl,id] = ind[p, n]
        end
        vert_ind[dir, n, 1,lvl,id] = min(vert_ind[dir, n, 1,lvl,id], cut_ind - 1)
        vert_ind[dir, n, 2,lvl,id] = max(vert_ind[dir, n, 1,lvl,id], cut_ind)
    end
    # Create interpolated vertices on cut plane
    for n = 1:cut_nvert[case]
        v1 = cut_v1[n, case]
        v2 = cut_v2[n, case]
        mu = min(1.0, max(0.0, -d[v1,id] /̂ (d[v2,id] - d[v1,id])))
        for p = 1:3
            vert[p, 4+n,lvl,id] = (1.0 - mu) * vert[p, v1,lvl,id] + mu * vert[p, v2,lvl,id]
        end
        # Get index for interpolated vertex
        i = vert_ind[1, n, 1,lvl,id]
        j = vert_ind[2, n, 1,lvl,id]
        k = vert_ind[3, n, 1,lvl,id]
        pt2index!(@view(vert_ind[:, 4+n, 1,lvl,id]), @view(vert[:, 4+n,lvl,id]), i, j, k, mesh)
        # Enforce boundedness
        for p=1:3
            vert_ind[p, 4+n, 1,lvl,id] = max(vert_ind[p, 4+n, 1,lvl,id], min(vert_ind[p, v1, 1,lvl,id], vert_ind[p, v2, 1,lvl,id]))
            vert_ind[p, 4+n, 1,lvl,id] = min(vert_ind[p, 4+n, 1,lvl,id], max(vert_ind[p, v1, 1,lvl,id], vert_ind[p, v2, 1,lvl,id]))
        end
        # Set +/- indices in cut direction
        for p=1:3
            vert_ind[p, 4+n, 2,lvl,id] = vert_ind[p, 4+n, 1,lvl,id]
        end
        vert_ind[dir, 4+n, 1,lvl,id] = cut_ind - 1
        vert_ind[dir, 4+n, 2,lvl,id] = cut_ind
    end
    # Create new tets
    vol = 0.0
    vLiq = 0.0
    vU = 0.0
    vV = 0.0
    vW = 0.0
    for n = 1:cut_ntets[case]
        # Form new tet
        for nn = 1:4
            for p = 1:3
                tet[p, nn] = vert[p, cut_vtet[nn, n, case],lvl,id]
                ind[p, nn] = vert_ind[p, cut_vtet[nn, n, case], cut_side[n, case],lvl,id]
            end
        end
        # Cut new tet by next plnae
        tetVol, tetvLiq, tetvU, tetvV, tetvW, maxlvl = cutTet(tet, ind, u, v, w, xdone, ydone, zdone, nx, ny, nz, D, mesh,lvl+1,vert,vert_ind,d,newtet)

        # Accumulate quantities
        vol += tetVol
        vLiq += tetvLiq
        vU += tetvU
        vV += tetvV
        vW += tetvW
    end

    return vol, vLiq, vU, vV, vW, maxlvl
end