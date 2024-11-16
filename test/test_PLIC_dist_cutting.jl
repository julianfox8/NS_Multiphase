    function test_PLIC_dist_cutting(nx,ny,nz,VF)
        # Setup test parameters
        param = parameters(
            # Constants
            mu_liq=0.1,       # Dynamic viscosity liquid
            mu_gas = 1e-4,
            rho_liq=1.0,
            rho_gas=1.0,           # Density
            sigma = 1.0,
            gravity = 1.0,
            Lx=3.0,            # Domain size
            Ly=1.0,
            Lz=1.0,
            tFinal=100.0,      # Simulation time

            # Discretization inputs
            Nx=10,           # Number of grid cells
            Ny=10,
            Nz=1,
            stepMax=20,   # Maximum number of timesteps
            CFL=0.1,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
            out_period=10,     # Number of steps between when plots are updated
            tol = 1e-3,

            # Processors 
            nprocx = 1,
            nprocy = 1,
            nprocz = 1,

            # Periodicity
            xper = false,
            yper = false,
            zper = false,
        )

        # Setup par_env
        par_env = NS.parallel_init(param)

        # Setup mesh
        mesh = NS.create_mesh(param,par_env)

        # Form PLIC distance computation 
        i=mesh.imin_; j=mesh.jmin_; k=mesh.kmin_
        dist = NS.computeDist(i,j,k,nx,ny,nz,VF,param,mesh)

        # Now compute VOF from PLIC to compare with specified value
        VF_calc = NS.computePLIC2VF(i,j,k,nx,ny,nz,dist,param,mesh)
        
        tol = 1e-12
        return VF_calc,VF,tol
    end