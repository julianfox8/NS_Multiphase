"""
Calculate analytic pre-image for deformation test case
"""

using NavierStokes_Parallel
# using ProfileView
# using Profile

using StaticArrays
using LinearAlgebra

NS = NavierStokes_Parallel

struct analyticPreimage
    # Pre-image  geometry
    edge_pts   :: Vector{Vector{NTuple{3,Float64}}} # 3D coordinate location of each point along cell edges
    edge_s     :: Vector{Vector{Float64}} # parametric variable for subdivision along edge
    edge_verts :: Vector{NTuple{2,Int}}   # vertex connectivity for edge (v_start, v_end)
    edge_cell  :: Vector{Int} # cell id for each edge
end

"""
Determine verts for a cell in ordering that triangulates faces consistently
    - verts is a 3xn array and that is filled by this function
"""
function cell2verts!(verts,i,j,k,param,mesh)
    @unpack x, y, z = mesh

    # Determine vertices for cell
    verts[:,1] = [x[i  ], y[j  ], z[k  ]]
    verts[:,2] = [x[i+1], y[j  ], z[k  ]]
    verts[:,3] = [x[i  ], y[j+1], z[k  ]]
    verts[:,4] = [x[i+1], y[j+1], z[k  ]]
    verts[:,5] = [x[i  ], y[j  ], z[k+1]]
    verts[:,6] = [x[i+1], y[j  ], z[k+1]]
    verts[:,7] = [x[i  ], y[j+1], z[k+1]]
    verts[:,8] = [x[i+1], y[j+1], z[k+1]]

    return nothing
end  

function analytic_projection()

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
        Lz=1/50,
        tFinal=1.0,      # Simulation time
        
        # Discretization inputs
        Nx=48,           # Number of grid cells
        Ny=48,
        Nz=2,
        stepMax=10000,   # Maximum number of timesteps
        max_dt =6e-2,#2.5e-3,
        CFL=3.0,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
        std_out_period = 0.0,
        out_period=1,     # Number of steps between when plots are updated
        tol = 1e-8,

        # Processors
        nprocx = 1,
        nprocy = 1,
        nprocz = 1,

        # Periodicity
        xper = false,
        yper = false,
        zper = true,

        # Turn off NS solver
        solveNS = false,
        VFVelocity = "Deformation",

        pressure_scheme = "finite-difference",
        # pressure_scheme = "semi-lagrangian",
        # pressureSolver = "hypreSecant",
        # pressureSolver = "res_iteration",
        # hypreSolver = "LGMRES",

        # projection_method = "Euler",
        projection_method = "RK4",
        # projection_method = "Midpoint",
        
        # pressurePrecond = "nl_jacobi",

        # hypreSolver = "GMRES-AMG",
        hypreSolver = "BiCGSTAB",
        mg_lvl = 1,
        # Iteration method used in @loop macro
        iter_type = "standard",
        #iter_type = "floop",
        test_case = "analytic_deformation",
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
    u_fun(x,y,z,t) = -2(sin(π*x))^2*sin(π*y)*cos(π*y)*cos(π*t/8.0)
    v_fun(x,y,z,t) = +2(sin(π*y))^2*sin(π*x)*cos(π*x)*cos(π*t/8.0)
    w_fun(x,y,z,t) = 0.0
    # Set velocities (including ghost cells)
    for k = kmino_:kmaxo_, j = jmino_:jmaxo_, i = imino_:imaxo_ 
        u[i,j,k]  = u_fun(xm[i],ym[j],zm[k],t)
        v[i,j,k]  = v_fun(xm[i],ym[j],zm[k],t)
        w[i,j,k]  = w_fun(xm[i],ym[j],zm[k],t)
    end


        fill!(VF,0.0)

        return nothing    
    end 

    # Setup par_env
    par_env = NS.parallel_init(param)
    # Setup mesh
    mg_mesh = NS.init_mg_mesh(param,par_env)
    # Initialize work arrays for finest level along with subset of arrays for coarser levels
    mesh = mg_mesh.mesh_lvls[1]
    # Initialize arrays
    P,u,v,w,VF,nx,ny,nz,D,band,us,vs,ws,uf,vf,wf,RHS,tmp2,exact_sol,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmplrg,Curve,sfx,sfy,sfz,denx,deny,denz,viscx,viscy,viscz,gradx,grady,gradz,divg,mask,tets,verts,inds,vInds = NS.initArrays(mesh)

    # init time
    t=0.0

    # Set velocity for iteration using deformation field
    NS.defineVelocity!(t,u,v,w,uf,vf,wf,param,mesh)
    @unpack dx,dy,Nx,Ny,Nz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh


    # Check timestep 
    dt = NS.compute_dt(u,v,w,param,mesh,par_env)
    println("CFL number of :", dt/max(dx/maximum(abs.(u)),dy/maximum(abs.(v))))

    n_cells = Nx*Ny*Nz
    n_edges_total = n_cells * 12
    sample_freq = 10
    edges = [(1,2), (2,4), (1,3), (3,4),(1,5), (2,6), (4,8), (3,7), (5,6), (6,8), (5,7), (7,8)]

    analytic = analyticPreimage(
        [Vector{NTuple{3,Float64}}(undef, sample_freq) for _ in 1:n_edges_total],
        [Vector{Float64}(undef, sample_freq) for _ in 1:n_edges_total],
        Vector{NTuple{2,Int}}(undef, n_edges_total),
        Vector{Int}(undef, n_edges_total)
    )

    edge_counter = 1
    cell_counter = 1
   # loop over domain and sample point along cell boundary for projection
    for k = kmin_:kmax_, j = jmin_:jmax_, i = imin_:imax_

        # determine verts from cell
        cell2verts!(verts,i,j,k,param,mesh)
        
        # sample along cell boundaries (need a cell 2 verts dependent upon sample rate)'
        
        for n in edges
            
            #identify vertices based on edges array
            vert1 = verts[:,n[1]]
            vert2 = verts[:,n[2]]
            #find index of the different coordinate 
            idx = findfirst(vert1 .!= vert2)
            #calculate the subdivision step size
            step = (vert2[idx] - vert1[idx])/(sample_freq-1)
            #loop over subdivisions
            for s in 0:(sample_freq-1)
                #copy pt to apply subdivision
                subd_pt = copy(vert1)
                subd_pt[idx] += step*s
                # project point along cell edge
                NS.project!(subd_pt,i,j,k,uf,vf,wf,dt,param,mesh)
                # store projected cell edges into edge_pts 
                analytic.edge_pts[edge_counter][s+1] = (subd_pt[1],subd_pt[2],subd_pt[3])
                # store paramteric varibale for edge subdivisions
                analytic.edge_s[edge_counter][s+1]   = s / (sample_freq-1)
                analytic.edge_cell[edge_counter]  = cell_counter
            end
            
            # convert local vertices to global
            global_n1 = (cell_counter - 1) * 8 + n[1]
            global_n2 = (cell_counter - 1) * 8 + n[2]
            analytic.edge_verts[edge_counter] = (global_n1,global_n2)
            edge_counter += 1

        end

        cell_counter += 1

    end
    NS.a_preimage2VTK(analytic,"analytic_preimage")
#    println("Max projection error: ", error1)
end

@time analytic_projection()
# @profview_allocs test_project()
# Profile.print(format=:flat, sortedby=:count)
