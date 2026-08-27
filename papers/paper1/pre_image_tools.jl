using CairoMakie

"""
need to keep in mind the orientation of the odd and even cells in mind
    # even cell    
    face 1: verts 5,7,3,1
    face 2: verts 8,6,2,4
    face 3: verts 2,6,5,1
    face 4: verts 8,4,3,7
    face 5: verts 2,1,3,4
    face 6: verts 6,8,7,5

    odd cell
    face 1: verts 8,6,2,4
    face 2: verts 5,7,3,1
    face 3: verts 2,6,5,1
    face 4: verts 7,8,4,3
    face 5: verts 2,1,3,4
    face 6: verts 6,5,7,8
"""

const FACES_EVEN = (
    (5,7,3,1),
    (8,6,2,4),
    (2,6,5,1),
    (8,4,3,7),
    (2,1,3,4),
    (6,8,7,5),
)

const FACES_ODD = (
    (8,6,2,4),
    (5,7,3,1),
    (2,6,5,1),
    (7,8,4,3),
    (2,1,3,4),
    (6,5,7,8),
)

"""
Richardson extrapolation for "analytic" semi-Lagrangian projection
    - I is a Romberg-style table 2D array of size (nsteps, nsteps) to store
    - extrapolation is considered converged based upon the tol
    - for use in the pre-image error test case
"""
function richardson_extrapolation_n!(I,pt, i, j, k, uf, vf, wf, dt, nsteps, param, mesh; tol = 1e-14)
    @unpack projection_method = param

    for idx in eachindex(I)
        I[idx] .= 0.0
    end
    for n in 1:nsteps
        I[n,1] .= pt
    end
 
    # Determine the base order of the method
    # p = projection_method == "Euler" ? 1 : projection_method == "Midpoint" || projection_method == "Heun" ? 2 : projection_method == "RK4" ? 4 : error("Unknown method")
    p = 1 # set to 1 as only Euler is used for the pre-image test case
    # Step 1: compute the time-integrated solutions
    for n in 1:nsteps
        dt_step = dt / 2^(n-1)
       
        # Apply the base integrator 2^(n-1) times
        for _ in 1:2^(n-1)
            # If the projection method is specified, use it; otherwise, default to Euler
            # NS.project!(I[n,1], i, j, k, uf, vf, wf, dt_step, param, mesh;)
            NS.project!(I[n,1], i, j, k, uf, vf, wf, dt_step, param, mesh;proj="Euler")
        end

        # move to next step if n == 1, since we need at least two levels to extrapolate
        if n == 1
            continue
        else
            for m in 1:n-1
                for l in m+1:n
                    I[n,l] .= (2^(p+(m-1)) * I[n,l-1] .- I[n-1,l-1]) / (2^(p+(m-1)) - 1)

                    if n > 2 && l == n
                        # Check convergence: compare last two refined levels
                        diff = maximum(abs.(I[n,l] .- I[n,l-1]))
                        if diff < tol
                            pt .= I[n,l]
                            return true  # early exit, converged
                        end
                    end
                end
            end
        end
    end
    
    # Return the most refined extrapolation
    println("Warning: Richardson extrapolation did not converge within the specified tolerance. Returning the most refined estimate.")
    pt .= I[end,end]

    return nothing
end

"""
Map the sample points on the pre-image based on barycentric coordinates
    - sample_pts is a 3xN array of points sampled along the cell faces
    - tri_verts is a 3x14 array of the 8 cell vertices followed by the 6 face midpoints used for triangulation
    - tri_ids is a length N vector that contains the triangle ID for each sample point
    - lambdas is a 3xN array that contains the barycentric coordinates for each sample point
    - tetsign is used to determine the correct triangulation of the faces based on cell orientation
"""
function map_sample_points_to_preimage!(sample_pts, tri_verts, tri_ids, lambdas, tetsign, ns)
    # Map each point using barycentric coordinates
    nfp = ns^2
    
    # loop over triangle id's
    for pt_id in eachindex(tri_ids)
    
        # grab point and face information based on tetsign
        tri = tri_ids[pt_id]
        face_id = div(pt_id-1, nfp) + 1

        if tetsign > 0
            face = FACES_EVEN[face_id]
        else
            face = FACES_ODD[face_id]
        end
        
        v1 = @view tri_verts[:,face[tri]]
        v2 = @view tri_verts[:,face[mod(tri,4)+1]]
        v3 = @view tri_verts[:,9+(face_id-1)]

        λ = lambdas[:, pt_id]
        sample_pts[:, pt_id] = λ[1]*v1 + λ[2]*v2 + λ[3]*v3
    end
end

"""
Sample points along the triangulated faces
    - pts is a 3xN array that will be filled with the sampled points
    - verts is a 3x8 array of the cell vertices
    - tetsign is used to determine the correct triangulation of the faces based on cell orientation
    - ns is the number of sample points along each edge of the face (total points per face = ns^2)
"""
function sample_cell_faces!(pts,verts,tetsign,ns)

    faces = tetsign > 0 ? FACES_EVEN : FACES_ODD

    # ξvals = range(0, 1, length=ns)
    ξvals = ((1:ns) .- 0.5) ./ ns
    ηvals = ξvals

    idx = 1
    for f in 1:6
        face = faces[f]

        v1 = @view verts[:,face[1]]
        v2 = @view verts[:,face[2]]
        v3 = @view verts[:,face[3]]
        v4 = @view verts[:,face[4]]

        for η in ηvals, ξ in ξvals
            @inbounds pts[:,idx] .=
                (1-ξ)*(1-η)*v1 +
                 ξ   *(1-η)*v2 +
                 ξ   * η   *v3 +
                (1-ξ)* η   *v4
            idx += 1
        end
    end
end

"""
Compute barycentric coordinates of sample points with respect to triangulated cell faces
    - tri_verts is a 3x14 array of the 8 cell vertices followed by the 6 face midpoints
    - sample_pts is a 3xN array of points sampled along the cell faces
    - tri_ids is a length N vector that will store the triangle ID for each sample point
    - lambdas is a 3xN array that will store the barycentric coordinates for each sample point

    algorithm:
    - determine the face the sample point lives on based on the number of samples points
    - using the FACES_EVEN and FACES_ODD arrays, determine the vertices and center point to loop over
    - loop over the 4 triangles that make up the face and compute barycentric coordinates 
    for each triangle to find the correct triangle and barycentric coordinates for the sample point
"""
function compute_barycentric!(tri_verts, sample_pts, tri_ids, lambdas,tetsign,ns; tol=1e-12)

    nfp = ns^2

    # loop over sample points
    for pt_id in axes(sample_pts, 2)
        # grab point and face information based on tetsign
        p = sample_pts[:, pt_id]
        found = false
        face_id = div(pt_id-1, nfp) + 1

        if tetsign > 0
            face = FACES_EVEN[face_id]
        else
            face = FACES_ODD[face_id]
        end

        for tri_id in 1:4
            v1 = @view tri_verts[:,face[tri_id]]
            v2 = @view tri_verts[:,face[mod(tri_id,4)+1]]
            v3 = @view tri_verts[:,9+(face_id-1)]
            
            # Vectors
            v0 = v2 - v1
            v1v = v3 - v1  
            v2v = p - v1

            # Dot products
            d00 = dot(v0, v0)
            d01 = dot(v0, v1v)
            d11 = dot(v1v, v1v)
            d20 = dot(v2v, v0)
            d21 = dot(v2v, v1v)

            denom = d00*d11 - d01*d01
            λ2 = (d11*d20 - d01*d21)/denom
            λ3 = (d00*d21 - d01*d20)/denom
            λ1 = 1 - λ2 - λ3

            # Allow small tolerance for points on edges/corners
            if λ1 >= -tol && λ2 >= -tol && λ3 >= -tol
                # Clamp slightly negative values to zero for stability
                λ1 = max(0.0, λ1)
                λ2 = max(0.0, λ2)
                λ3 = max(0.0, λ3)

                tri_ids[pt_id] = tri_id
                lambdas[:, pt_id] .= (λ1, λ2, λ3)
                found = true
                break
            end

        end

        if !found
            error("Point $pt_id is not inside any triangle!")
        end
    end
end

"""
Determine triangulated vertices from the tets 
    - tets is a 4xN array of the vertex indices for each tetrahedron in the cell (independent of cell orientation)
    - tri_verts is a 3x14 array of the 8 cell vertices followed by the 6 face midpoints
"""
function triangulate_face_wtets!(tetsign,verts,tets)
    if tetsign > 0
        verts[:,9] = tets[:,4,6] # face 1: verts 5,7,3,1
        verts[:,10] = tets[:,4,8] # face 2: verts 8,6,2,4
        verts[:,11] = tets[:,4,10] # face 3: verts 2,6,5,1
        verts[:,12] = tets[:,4,12] # face 4: verts 8,4,3,7
        verts[:,13] = tets[:,4,14] # face 5: verts 2,1,3,4
        verts[:,14] = tets[:,4,16] # face 6: verts 6,8,7,5
    else
        verts[:,9] = tets[:,4,6] # face 1: verts 8,6,2,4
        verts[:,10] = tets[:,4,8] # face 2: verts 5,7,3,1
        verts[:,11] = tets[:,4,10] # face 3: verts 2,6,5,1
        verts[:,12] = tets[:,4,12] # face 4: verts 7,8,4,3
        verts[:,13] = tets[:,4,14] # face 5: verts 2,1,3,4
        verts[:,14] = tets[:,4,16] # face 6: verts 6,5,7,8
    end
    return nothing
end

"""
Compute the barycentric triangulation of the cells consistent with tet sign
    - verts is a 3xn array and that is filled by this function
    - assumes square cells and that the first 8 verts are ordered as in cell2verts!
    - uses midpoint between two diagonal cells for barycentric triangulation
"""
function triangulate_face!(sign,verts)
    if sign > 0
        # even cell
        # face 1: verts 5,7,3,1
        verts[:,9] = 0.5*(verts[:,1] + verts[:,7]) # diagonal midpoint
        # face 2: verts 8,6,2,4
        verts[:,10] = 0.5*(verts[:,6] + verts[:,4]) # diagonal midpoint
        # face 3: verts 2,6,5,1
        verts[:,11] = 0.5*(verts[:,1] + verts[:,6]) # diagonal midpoint
        # face 4: verts 8,4,3,7
        verts[:,12] = 0.5*(verts[:,7] + verts[:,4]) # diagonal midpoint
        # face 5: verts 2,1,3,4
        verts[:,13] = 0.5*(verts[:,1] + verts[:,4]) # diagonal midpoint
        # face 6: verts 6,8,7,5
        verts[:,14] = 0.5*(verts[:,6] + verts[:,7]) # diagonal midpoint
    else
        # odd cell
        # face 1: verts 8,6,2,4
        verts[:,9] = 0.5*(verts[:,6] + verts[:,4]) # diagonal midpoint
        # face 2: verts 5,7,3,1
        verts[:,10] = 0.5*(verts[:,1] + verts[:,7]) # diagonal midpoint
        # face 3: verts 2,6,5,1
        verts[:,11] = 0.5*(verts[:,1] + verts[:,6]) # diagonal midpoint
        # face 4: verts 7,8,3,4
        verts[:,12] = 0.5*(verts[:,7] + verts[:,4]) # diagonal midpoint
        # face 5: verts 2,1,3,4
        verts[:,13] = 0.5*(verts[:,1] + verts[:,4]) # diagonal midpoint
        # face 6: verts 6,5,7,8
        verts[:,14] = 0.5*(verts[:,6] + verts[:,7]) # diagonal midpoint
    end
    return nothing
end  

"""
    plot_points_on_faces(tri_verts, tri_id, lambdas; tetsign=nothing)

Plot points mapped back onto the triangulated cell faces using barycentric coordinates.

Arguments:
- `tri_verts` : 3×3×Ntri array of triangle vertices (x,y,z)
- `tri_id`    : vector of length Npts with triangle indices for each point
- `lambdas`   : 3×Npts barycentric coordinates
- `tetsign`  : optional, scalar or vector to color points by cell orientation
"""
function plot_points_on_faces(tri_verts, tri_ids, lambdas, tetsign, ns)
    
    points_mapped = zeros(3, length(tri_ids))

    # Map each point using barycentric coordinates
    nfp = ns^2

    # loop over triangle id's
    for pt_id in eachindex(tri_ids)
        # grab point and face information based on tetsign
        tri = tri_ids[pt_id]
        face_id = div(pt_id-1, nfp) + 1

        if tetsign > 0
            face = FACES_EVEN[face_id]
        else
            face = FACES_ODD[face_id]
        end

        
        v1 = @view tri_verts[:,face[tri]]
        v2 = @view tri_verts[:,face[mod(tri,4)+1]]
        v3 = @view tri_verts[:,9+(face_id-1)]

        λ = lambdas[:, pt_id]
        points_mapped[:, pt_id] = λ[1]*v1 + λ[2]*v2 + λ[3]*v3
    end

    # Make a 3D scatter plot
    fig = Figure(size=(600,600))
    ax = Axis3(fig[1,1],
        xlabel = "X",
        ylabel = "Y",
        zlabel = "Z",
        title = "Mapped points on triangulated cell faces",
        aspect = :data)
    
    # Optional coloring by tetsign
    colors = tetsign === nothing ? :blue : tetsign

    scatter!(ax, points_mapped[1,:], points_mapped[2,:], points_mapped[3,:], markersize=8, color=colors)
    nfaces = 6
    nfp = ns^2
    offset = 0

    for f in 1:nfaces

        face_pts = @view points_mapped[:, offset+1 : offset+nfp]

        X = reshape(face_pts[1,:], ns, ns)
        Y = reshape(face_pts[2,:], ns, ns)
        Z = reshape(face_pts[3,:], ns, ns)

        # # ξ-direction lines
        # for j in 1:ns
        #     lines!(ax, X[:,j], Y[:,j], Z[:,j])
        # end

        # # η-direction lines
        # for i in 1:ns
        #     lines!(ax, X[i,:], Y[i,:], Z[i,:])
        # end
        surface!(ax,X,Y,Z,color = :lightblue, shading=true,transparency = true)
        offset += nfp
    end

    display(fig)
end

"""
Plot the sampled cell at different stages within the test
"""
function plot_sampled_cell(sample_pts, verts, ns; original_verts = nothing, title_str = "test cell")
    fig = Figure(size = (700, 700))
    ax = Axis3(fig[1,1],
        # xlabel = "X",
        # ylabel = "Y",
        # zlabel = "Z",
        protrusions = 75,
        # title = title_str,
        titlesize = 30,
        xlabelsize = 24,
        ylabelsize = 24,
        zlabelsize = 24,
        xticklabelsize = 18,
        yticklabelsize = 18,
        zticklabelsize = 18,
        aspect = :data
    )
    hidedecorations!(ax,grid=false)
    # Plot original 8 vertices for reference
    if !isnothing(original_verts)
        scatter!(ax,
            original_verts[1, 1:8],
            original_verts[2, 1:8],
            original_verts[3, 1:8],
            markersize = 9,
            color = :red
        )
        for f in FACES_EVEN
            i1,i2,i3,i4 = f

            xs = original_verts[1,[i1,i2,i3,i4,i1]]
            ys = original_verts[2,[i1,i2,i3,i4,i1]]
            zs = original_verts[3,[i1,i2,i3,i4,i1]]

            lines!(ax, xs, ys, zs, color = :red, linewidth = 2)
        end
    end

    
    nfaces = 6
    nfp = ns^2
    offset = 0
    alpha_list = [0.7, 0.4, 0.7, 0.1, 0.2, 0.7]
    face_colors = [:lightblue, :blue, :yellow, :orange, :green, :darkgreen]  # one color per face

    for f in 1:nfaces
        face_pts = @view sample_pts[:, offset+1 : offset+nfp]

        X = reshape(face_pts[1,:], ns, ns)
        Y = reshape(face_pts[2,:], ns, ns)
        Z = reshape(face_pts[3,:], ns, ns)

        # Plot surface of the face
        # surface!(ax, X, Y, Z, color=face_colors[f], alpha=alpha_list[f])
        # surface!(ax, X, Y, Z, color = :lightblue, alpha=0.4,transparency=true)
        surface!(ax, X, Y, Z, alpha=0.4,transparency=true,colormap = :magma)

        # Scatter the points of this face
        # scatter!(ax, face_pts[1,:], face_pts[2,:], face_pts[3,:], 
        #          color=face_colors[f], markersize=6)#, label="Face $f points")
        scatter!(ax, face_pts[1,:], face_pts[2,:], face_pts[3,:], 
                 color = :black, markersize=6)
        offset += nfp
    end

    # axislegend(ax, position = :rb)

    display(fig)
end
