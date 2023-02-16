
using NavierStokes_Parallel
using Printf
using OffsetArrays
using Plots

NS = NavierStokes_Parallel

function test_SLdivergence()
    # Setup test parameters
    param = parameters(
        # Constants
        mu=1e-5,       # Dynamic viscosity
        rho=1.0,           # Density
        Lx=1.0,            # Domain size
        Ly=1.0,
        Lz=1.0,
        tFinal=100.0,      # Simulation time

        # Discretization inputs
        Nx=3,           # Number of grid cells
        Ny=3,
        Nz=1,
        stepMax=20,   # Maximum number of timesteps
        CFL=0.5,         # Courant-Friedrichs-Lewy (CFL) condition for timestep
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

    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack x,y,z,xm,ym,zm = mesh
    @unpack dx,dy,dz = mesh

    # Create work arrays
    # zero velocity
    v0 = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(v0,0.0)
    # velocity field a
    ufa = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(ufa,0.0)
    vfa = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(vfa,0.0)
    wfa = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(wfa,0.0)
    # velocity field b
    ufb = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(ufb,0.0)
    vfb = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(vfb,0.0)
    wfb = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(wfb,0.0)
    # divergence
    diva  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(diva ,0.0)
    divb  = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(divb ,0.0)
    divab = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_); fill!(divab,0.0)
    
    
    #####################################
    # Play with different velocity fields
    #####################################
    # Set velocity fields 
    for k=kmino_:kmaxo_, j=jmino_:jmaxo_, i=imino_:imaxo_
        # # Field a
        # ufa[i,j,k] = ym[j] + x[i]
        # vfa[i,j,k] = 0.0
        # wfa[i,j,k] = 0.0
        # # Field b
        # ufb[i,j,k] = 0.0
        # vfb[i,j,k] = xm[i] + y[j]
        # wfb[i,j,k] = 0.0

        # Field a
        ufa[i,j,k] = sin(π*x[i]) + x[i]
        vfa[i,j,k] = xm[i]+0.5
        wfa[i,j,k] = 0.0
        # Field b
        ufb[i,j,k] = ym[j]+0.5
        vfb[i,j,k] = sin(π*y[j]) + y[j]
        wfb[i,j,k] = 0.0
    end

    # # Print velocity fields
    # NS.printArray("ufa",ufa[imin_:imax_+1,jmin_:jmax_  ,kmin_],par_env)
    # NS.printArray("vfa",vfa[imin_:imax_  ,jmin_:jmax_+1,kmin_],par_env)

    # NS.printArray("ufb",ufb[imin_:imax_+1,jmin_:jmax_  ,kmin_],par_env)
    # NS.printArray("vfb",vfb[imin_:imax_  ,jmin_:jmax_+1,kmin_],par_env)

    # Set timestep 
    dta = NS.compute_dt(ufa,vfa,wfa,param,mesh,par_env)
    dtb = NS.compute_dt(ufb,vfb,wfb,param,mesh,par_env)
    dt = minimum([dta,dtb])

    # Compute divergences
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # Field a 
        tets,ind = NS.cell2tets_withProject_uvwf(i,j,k,ufa,vfa,wfa,dt,mesh)
        vol1 = dx*dy*dz
        vol2 = NS.tets_vol(tets)
        diva[i,j,k] = vol1-vol2

        # Field b
        tets,ind = NS.cell2tets_withProject_uvwf(i,j,k,ufb,vfb,wfb,dt,mesh)
        vol1 = dx*dy*dz
        vol2 = NS.tets_vol(tets)
        divb[i,j,k] = vol1-vol2

        # Field a + b
        tets,ind = NS.cell2tets_withProject_uvwf(i,j,k,ufa.+ufb,vfa.+vfb,wfa.+wfb,dt,mesh)
        vol1 = dx*dy*dz
        vol2 = NS.tets_vol(tets)
        divab[i,j,k] = vol1-vol2
    end

    # Output 
    @printf(" %3s %3s %3s  %10s  %10s  %10s  %10s  %9s \n",
        "i","j","k","∇⋅A","∇⋅B","∇⋅A + ∇⋅B","∇⋅(A+B)", "Error")
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        error = abs((diva[i,j,k]+divb[i,j,k]) - divab[i,j,k])
        @printf(" %3i %3i %3i  %+6.3e  %+6.3e  %+6.3e  %+6.3e  %6.3e \n",
        i,j,k,diva[i,j,k],divb[i,j,k],diva[i,j,k]+divb[i,j,k],divab[i,j,k],error)
    end


    # Plots
    fig1 = plot(legend=false)
    fig2 = plot(legend=false)
    fig3 = plot(legend=false)
    fig4 = plot(legend=false)
    plotGrid(fig1, (ufa    ,), (vfa    ,), (wfa    ,),     diva,  dt, mesh, title="Velocity A"               ,color=:blue  )
    plotGrid(fig2, (    ufb,), (    vfb,), (    wfb,),     divb,  dt, mesh, title="Velocity B"               ,color=:blue  )
    plotGrid(fig3, (ufa+ufb,), (vfa+vfb,), (wfa+wfb,),     divab, dt, mesh, title="Velocity (A + B)"         ,color=:blue  )
    plotGrid(fig4, (ufa,ufb,), (vfa,vfb,), (wfa,wfb,), diva+divb, dt, mesh, title="Velocity A + Velocity B"  ,color=:blue  )
    myplt=plot(fig1,fig2,fig3,fig4,
                layout=(2,2),
                size=(1000,900),
    )
    display(myplt)
    savefig("SLdivergence1.pdf")

    # Plot of Velocity (A+B) and Velocity A + Velocity B on same figure
    fig1 = plot(legend=false,size=(1000,900))
    plotGrid(fig1, (ufa+ufb,), (vfa+vfb,), (wfa+wfb,),     divab, dt, mesh, title=""         ,color=:blue  )
    plotGrid(fig1, (ufa,ufb,), (vfa,vfb,), (wfa,wfb,), diva+divb, dt, mesh, title="Velocity (A + B) & Velocity A + Velocity B"  ,color=:blue  )
    display(fig1)
    savefig("SLdivergence2.pdf")

    
end

function plotGrid(fig,uf,vf,wf,div,dt,mesh; title="", color=:black)
    @unpack x,y,z = mesh 
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    
    # Only works in 2D!
    k = kmin_
    for j=jmin_:jmax_, i=imin_:imax_

        # Cell corners 
        pt1 = [x[i  ],y[j  ],z[k  ]]
        pt2 = [x[i+1],y[j  ],z[k  ]]
        pt3 = [x[i  ],y[j+1],z[k  ]]
        pt4 = [x[i+1],y[j+1],z[k  ]]

        # Plot cell 
        plot_cell(fig,pt1,pt2,pt3,pt4,title,(:grey,:dash,1,0.5))

        # Lines for porjections 
        lines=[:black,:green]
        for n in eachindex(uf)
            
            # Project 
            pt1_new = NS.project_uvwf(pt1,i,j,k,uf[n],vf[n],wf[n],dt,mesh)
            pt2_new = NS.project_uvwf(pt2,i,j,k,uf[n],vf[n],wf[n],dt,mesh)
            pt3_new = NS.project_uvwf(pt3,i,j,k,uf[n],vf[n],wf[n],dt,mesh)
            pt4_new = NS.project_uvwf(pt4,i,j,k,uf[n],vf[n],wf[n],dt,mesh)

            # Plot projection 
            GR.setarrowsize(0.5)
            plot!([pt1[1],pt1_new[1]],[pt1[2],pt1_new[2]],arrow=(:closed), arrowsize=5, line=lines[n], label="")
            plot!([pt2[1],pt2_new[1]],[pt2[2],pt2_new[2]],arrow=(:closed), arrowsize=5, line=lines[n], label="")
            plot!([pt3[1],pt3_new[1]],[pt3[2],pt3_new[2]],arrow=(:closed), arrowsize=5, line=lines[n], label="")
            plot!([pt4[1],pt4_new[1]],[pt4[2],pt4_new[2]],arrow=(:closed), arrowsize=5, line=lines[n], label="")

            # Update points 
            pt1 = pt1_new
            pt2 = pt2_new
            pt3 = pt3_new
            pt4 = pt4_new

        end

        # Plot projected cell 
        divrange=range(-5e-2, 5e-2, length=100)
        colors=range(colorant"blue", stop=colorant"red", length=100)
        value,index = findmin(abs.(divrange .- div[i,j,k]))
        fillcolor=colors[index]
        plot_cell(fig,pt1,pt2,pt3,pt4,title,color; fill=true, fillcolor=fillcolor)

    end
    return fig
end

function plot_cell(fig,pt1,pt2,pt3,pt4,title,linestyle; fill=false, fillcolor=nothing)
    # Plot this cell 
    fig = plot!(fig,title=title, aspect_ratio=:equal,
            xlim=(-0.1,1.3),
            ylim=(-0.1,1.25),
            #axis=nothing,
            #border=:none,
            grid=false,
        )
    fig = plot!(fig,[pt1[1],pt2[1]],[pt1[2],pt2[2]],line=linestyle) # Bottom
    fig = plot!(fig,[pt3[1],pt4[1]],[pt3[2],pt4[2]],line=linestyle) # Top
    fig = plot!(fig,[pt1[1],pt3[1]],[pt1[2],pt3[2]],line=linestyle) # Left
    fig = plot!(fig,[pt2[1],pt4[1]],[pt2[2],pt4[2]],line=linestyle) # Right
    if fill 
        fig = plot!(fig, Shape(
            [pt1[1],pt2[1],pt4[1],pt3[1]],
            [pt1[2],pt2[2],pt4[2],pt3[2]]),
            color=fillcolor,
            opacity=.25,
            )
    end

    return fig
end

# Run test
test_SLdivergence()