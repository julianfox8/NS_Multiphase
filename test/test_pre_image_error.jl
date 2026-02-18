using NavierStokes_Parallel 
# using CairoMakie
using GLMakie
using LinearAlgebra
NS = NavierStokes_Parallel

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


function plot_sampled_cell(sample_pts, verts, ns; original_verts = nothing, title_str = "test cell")
    fig = Figure(size = (700, 700))
    ax = Axis3(fig[1,1],
        xlabel = "X",
        ylabel = "Y",
        zlabel = "Z",
        protrusions = 75,
        title = title_str,
        #azimuth = π/6,
        aspect = :data
    )
    
    # Plot original 8 vertices for reference
    if !isnothing(original_verts)
        scatter!(ax,
            verts[1, 1:8],
            verts[2, 1:8],
            verts[3, 1:8],
            markersize = 9,
            color = :red
        )
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
        surface!(ax, X, Y, Z, alpha=0.4,transparency=true)

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


function sample_cell_faces!(pts,verts,tetsign,ns)

    faces = tetsign > 0 ? FACES_EVEN : FACES_ODD

    ξvals = range(0, 1, length=ns)
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
    -using the FACES_EVEN and FACES_ODD arrays, determine the vertices and center point to loop over
    -loop over the 4 triangles that make up the face and compute barycentric coordinates 
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
            # if p == [0.0 , 0.25, 0.25]
            #     println("Checking point ", p, " against triangle with vertices ", v1, ", ", v2, ", ", v3)
            #     println(8+(face_id-1))
            # end
            # println("Checking point ", p, " against triangle with vertices ", v1, ", ", v2, ", ", v3)
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
function triangulate_face_wtets(tetsign,verts,tets)
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
Determine verts for a cell in ordering that triangulates faces consistently
    - verts is a 3xn array and that is filled by this function
    - assumes square cells and that the first 8 verts are ordered as in cell2verts!
    -uses midpoint between two diagonal cells
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
Richardson extrapolation wrapper for project!

Arguments:
- pt         : point to be updated (mutated in-place)
- i,j,k      : mesh indices
- uf,vf,wf   : velocity fields
- dt         : coarse time step
- param,mesh: passed through untouched
- p          : order of the base method (Euler=1, Midpoint=2, RK4=4)
"""
function richardson_extrapolation_n!(pt, i, j, k, uf, vf, wf, dt, nsteps, param, mesh; tol = 1e-14)
    @unpack projection_method = param
    # Base solutions at different step refinements
    I = [copy(pt) for _ in 1:nsteps]
    
    # Step 1: compute the time-integrated solutions
    for n in 1:nsteps
        dt_step = dt / 2^(n-1)
        # Apply the base integrator 2^(n-1) times
        for _ in 1:2^(n-1)
            NS.project!(I[n], i, j, k, uf, vf, wf, dt_step, param, mesh)
        end
    end
    
    # Determine the base order of the method
    p = projection_method == "Euler" ? 1 : projection_method == "Midpoint" ? 2 : projection_method == "RK4" ? 4 : error("Unknown method")
    
    # Build Richardson table
    R = [copy(I[n]) for n in 1:nsteps]  # first column = base solutions
    
    # Extrapolate: Romberg-style triangular table
    for k in 1:nsteps-1
        for i in nsteps:-1:k+1
            # multiplier for error reduction: 2^p for first level, 2^(2p) for second, etc.
            R[i] .= (2^(p+(k-1)) * R[i] .- R[i-1]) / (2^(p+(k-1)) - 1)
        end

        # Check convergence: compare last two refined levels
        # diff =  norm(R[end] .- R[end-1])
        diff = maximum(abs.(R[end] .- R[end-1]))
        if diff < tol
            # println("Converged at level $k with error = $diff")
            pt .= R[end]
            return true  # early exit, converged
        end
    end
    
    # Return the most refined extrapolation
    pt .= R[end]

    return nothing
end



function pre_image_err(dts)
    # Define parameters 
    param = parameters(
        # Constants
        mu_liq=0.0,            # Dynamic viscosity
        mu_gas = 0.0,
        rho_liq=1.0,           # Density
        rho_gas = 1.0,
        sigma = 0.0, # surface tension coefficient (N/m)
        grav_x = 0.0,
        grav_y = 0.0,
        grav_z = 0.0, # Gravity (m/s^2)
        Lx=1.0,            # Domain size
        Ly=1.0,
        Lz=1.0,#1/50,
        tFinal=3.0,      # Simulation time
        
        # Discretization inputs
        Nx=32,           # Number of grid cells
        Ny=32,
        Nz=32,
        stepMax=1,   # Maximum number of timesteps
        max_dt = 1e-1,
        CFL=3.0,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
        std_out_period = 0.0,
        out_period=1,     # Number of steps between when plots are updated
        tol = 1e-7,

        # Processors 
        nprocx = 1,
        nprocy = 1,
        nprocz = 1,

        # Periodicity
        xper = false,
        yper = false,
        zper = false,

        # Turn off NS solver
        solveNS = false,
        # VFVelocity = "Deformation",
        VFVelocity = "Deformation3D",

        pressure_scheme = "finite-difference",
        # pressure_scheme = "semi-lagrangian",
        # pressureSolver = "hypreSecant",
        # pressureSolver = "res_iteration",

        hypreSolver = "GMRES-AMG",
        # hypreSolver = "BiCGSTAB",
        projection_method = "Euler",
        # projection_method = "RK4",
        # projection_method = "Midpoint",

        # Iteration method used in @loop macro
        iter_type = "standard",
        #iter_type = "floop",
        test_case = "test_error_pre_image",
    )

    """
    Initial conditions for pressure and velocity
    """
    function IC!(P,u,v,w,VF,mesh)
        @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
        @unpack x,y,z = mesh
        @unpack xm,ym,zm = mesh

        # Velocity
        t=0.0
        # u_fun(x,y,z,t) = -2(sin(π*x))^2*sin(π*y)*cos(π*y)*cos(π*t/3.0)
        # v_fun(x,y,z,t) = +2(sin(π*y))^2*sin(π*x)*cos(π*x)*cos(π*t/3.0)
        # w_fun(x,y,z,t) = 0.0

        u_fun = (x,y,z,t) -> 2(sin(π*x))^2*sin(2π*y)*sin(2π*z)*cos(π*t/3.0)
        v_fun = (x,y,z,t) -> -(sin(π*y))^2*sin(2π*x)*sin(2π*z)*cos(π*t/3.0)
        w_fun = (x,y,z,t) -> -(sin(π*z))^2*sin(2π*x)*sin(2π*y)*cos(π*t/3.0)
        # Set velocities (including ghost cells)
        for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
            u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
            v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
            w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
        end

        # Volume Fraction
        # rad=0.15
        # xo=0.5
        # yo=0.75

        # for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        #     VF[i,j,k]=VFcircle(x[i],x[i+1],y[j],y[j+1],rad,xo,yo)
        # end


        # # Volume Fraction 3D
        rad=0.15
        xo=0.5
        yo=0.75
        zo=0.5

        for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
            VF[i,j,k]=VFbubble3d(x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],rad,xo,yo,zo)
        end
        return nothing    
    end

    """
    Boundary conditions for velocity
    """
    function BC!(u,v,w,t,mesh,par_env)
        # Not needed when solveNS=false
        return nothing
    end

    # Setup par_env
    par_env = NS.parallel_init(param)

    # Setup mesh
    mg_mesh = NS.init_mg_mesh(param,par_env)
    mesh = mg_mesh.mesh_lvls[1]
    # Initialize arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,divg,mask,tets,verts,inds,vInds = NS.initArrays(mesh)

    @unpack x,y,z,dx,dy,dz,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    p_min,p_max = NS.prepare_indices(tmp5,par_env,mesh)
    mg_arrays = NS.mg_initArrays(mg_mesh,param,p_min,p_max,par_env)


    if param.pressure_scheme == "finite-difference"
        if param.tesselation == "6_tets"
            num_tets = 6
        elseif param.tesselation == "5_tets"
            num_tets = 5
        elseif param.tesselation == "24_tets"
            num_tets = 24
        end
    end

    # Create initial condition
    t = 0.0 :: Float64
    IC!(P,u,v,w,VF,mesh)


    # # Check divergence
    # dt = NS.compute_dt(u,v,w,param,mesh,par_env)
    @unpack dx,dy,Nx,Ny,Nz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh


    # compute density and viscosity at intial conditions
    NS.compute_props!(denx,deny,denz,viscx,viscy,viscz,VF,param,mesh)

    #initialize sample points along cell face
    ns = 5
    nfaces = 6
    nfp = ns^2
    nverts = 8
    ntotal = nfaces * nfp
    sample_pts = Matrix{Float64}(undef, 3, ntotal)
    preimage_sample_pts = Matrix{Float64}(undef, 3, ntotal)
    
    # objects for barycentric coordinate computation
    tri_ids = Vector{Int}(undef, ntotal)
    lambdas = Matrix{Float64}(undef, 3, ntotal)


    pre_tri_verts = Array{eltype(tets)}(undef,3,nverts+6) ;fill!(pre_tri_verts, 0.0)
    tri_verts = Array{eltype(tets)}(undef,3,nverts+6) ;fill!(tri_verts, 0.0)


    # #initialize error
    # errs = zeros(length(test_dts))
    #! store copies of face velocity field 
    error_dt = zeros(length(dts))

    for (idx, dt) in enumerate(dts)
        # Set velocity for iteration using deformation field
        NS.defineVelocity!(t,u,v,w,uf,vf,wf,param,mesh)

        uf_old = copy(uf)
        vf_old = copy(vf)
        wf_old = copy(wf)
        if param.pressure_scheme == "semi-lagrangian" #need to determine corrected field
            # Determine pressure correction
            iter = NS.pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mg_mesh,par_env,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,gradx,grady,gradz,verts,tets,mg_arrays,BC!;)
            println("Pressure solver converged in $iter iterations for dt = $dt")
            # Corrector face velocities
            NS.corrector!(uf,vf,wf,P,dt,denx,deny,denz,mesh)
        end

        errors = tmp9; fill!(errors,0.0)
        # loop over domain and project vertices to test numerical integration 
        for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_
            
            # Get cell vertices with triangulation ordering
            tetsign = NS.cell2verts!(verts,i,j,k,param,mesh)
            
            tri_verts[:,1:8] = verts
            triangulate_face!(tetsign,tri_verts)
            
            # Sample face of cell with some number of points
            sample_cell_faces!(sample_pts, verts, tetsign, ns)
            
            # Visualize sampled points on faces before projection
            # if k == 1 && j == 11 && i == 11
            #     plot_sampled_cell(sample_pts, verts, ns;title_str = "cell sampled with $nfp points per face")
            # end
            # Compute barycentric coordinates 
            compute_barycentric!(tri_verts, sample_pts, tri_ids, lambdas, tetsign, ns)
            
            # Check that barycentric coordinates map back to the correct point on the face
            # if k == 3 && j == 3 && i == 3
            #     plot_points_on_faces(tri_verts, tri_ids, lambdas, tetsign, ns)
            # end

            # Project sampled points to get analytic pre-image
            for pt_id in axes(sample_pts, 2)
                nsteps = 10
                # pt = sample_pts[:, pt_id]
                # richardson_extrapolation_n!(pt, i, j, k, uf, vf, wf, dt, nsteps, param, mesh; tol = 1e-12)
                richardson_extrapolation_n!(@view(sample_pts[:,pt_id]), i, j, k, uf_old, vf_old, wf_old, dt, nsteps, param, mesh; tol = 1e-14)
            end

            # if k == 1 && j ==11 && i == 11
            #     plot_sampled_cell(sample_pts, verts, ns; title_str = "reference projected cell sampled with $nfp points per face")
            # end
            

            #! need to store a copy of the velocity field for use with the pressure corrected field
            #! will just use corrected field and cell to tets 

            # Compute numerical pre-image (either flux-corrected or pressure-corrected)
            tetsign = NS.cell2tets!(verts,tets,i,j,k,param,mesh; 
            project_verts=true,uf=uf,vf=vf,wf=wf,dt=dt,
            compute_indices=true,inds=inds,vInds=vInds,)

            pre_tri_verts[:,1:8] = verts

            if param.pressure_scheme == "finite-difference"
                # Add correction tets 
                tets,inds,ntets = NS.add_correction_tets(num_tets,verts,tets,inds,i,j,k,uf,vf,wf,dt,param,mesh;)  
            end
            
            # Triangulate pre-image along faces
            if param.pressure_scheme == "semi-lagrangian"
                triangulate_face!(tetsign,pre_tri_verts)
            elseif param.pressure_scheme == "finite-difference"
                triangulate_face_wtets(tetsign,pre_tri_verts,tets)
            end

            # Map sample points to pre-image using barycentric coordinates and triangulation with tets
            map_sample_points_to_preimage!(preimage_sample_pts, pre_tri_verts, tri_ids, lambdas, tetsign, ns)

            # Check that barycentric coordinates map back to the correct point on the face
            if k == 11 && j == 11 && i == 11
                plot_sampled_cell(preimage_sample_pts, pre_tri_verts, ns;title_str = "flux-corrected pre-image of sampled points with $nfp points per face")
                error("stop")
            end

            # Compute error in sample points for i,j,k cell
            # errors[i,j,k] = maximum(sqrt.(sum((preimage_sample_pts .- sample_pts).^2)))
            errors[i,j,k] =  sum((preimage_sample_pts .- sample_pts).^2)
            # error("stop")
        end
        error_dt[idx] = sqrt(sum(errors)/(ntotal*Nx*Ny*Nz))
        # error("stop")
    end
    # errs[n] = err

    # println("Max error in vertex position after projection is: ", maximum(errors))
    return error_dt
    
end


dts = [0.1,0.075,0.05,0.025,0.01,0.0075,0.005,0.0025,0.001] 
# dts = [0.075,0.05,0.025,0.01,0.0075,0.005,0.0025,0.001]
errors = pre_image_err(dts)

println("Errors for each dt: ", errors)
f = Figure(size = (700, 500))
ax = Axis(
    f[1,1],
    xscale = log10,
    yscale = log10,
    xlabel = "Δt",
    ylabel = "L2 error",
    title  = "Flux-corrected pre-image 3D deformation flow"
)

# line
lines!(ax, dts, errors, label = "Observed error")

# markers
scatter!(ax, dts, errors, markersize = 10)

# reference slopes (anchored at first point)
err0 = errors[1]
dt0  = dts[1]

ref2 = err0 .* (dts ./ dt0).^2
ref3 = err0 .* (dts ./ dt0).^3
ref4 = err0 .* (dts ./ dt0).^4

lines!(ax, dts, ref2, linestyle = :dot,  label = "O(Δt²)")
lines!(ax, dts, ref3, linestyle = :dashdot, label = "O(Δt³)")
lines!(ax, dts, ref4, linestyle = :dash, label = "O(Δt⁴)")
axislegend(ax, position = :rb)

f