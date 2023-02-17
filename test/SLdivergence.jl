
using NavierStokes_Parallel
using Printf
using OffsetArrays
using Plots
using Measures

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
        Nx=1,           # Number of grid cells
        Ny=1,
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
        ufa[i,j,k] = -(x[i]-0.5) #1.0
        vfa[i,j,k] = -(y[j] - 0.5)
        wfa[i,j,k] = 0.0
        # Field b
        ufb[i,j,k] = x[i]-0.5
        vfb[i,j,k] = (y[j] - 0.5)
        wfb[i,j,k] = 0.0
    end

    # Set timestep 
    dta = NS.compute_dt(ufa,vfa,wfa,param,mesh,par_env)
    dtb = NS.compute_dt(ufb,vfb,wfb,param,mesh,par_env)
    dt = minimum([dta,dtb])
    println(dt)

    # Output header 
    @printf("%10s  %10s  %10s  %10s  %10s  %9s \n",
            "Δt","∇⋅A","∇⋅B","∇⋅A + ∇⋅B","∇⋅(A+B)", "Error")
        

    dts=0.01:0.05:2dt
    diva  = similar(dts)
    divb  = similar(dts)
    divab = similar(dts)
    diva_fd  = similar(dts)
    divb_fd  = similar(dts)
    divab_fd = similar(dts)
    anim = @animate for n in eachindex(dts)

        # Only work with 1 cell 
        i=imin_;j=jmin_;k=kmin_

        # Field a 
        tets,ind = NS.cell2tets_withProject_uvwf(i,j,k,ufa,vfa,wfa,dts[n],mesh)
        vol1 = dx*dy*dz
        vol2 = NS.tets_vol(tets)
        diva[n] = (vol1-vol2)/dts[n]
        diva_fd[n] = ( (ufa[i+1,j,k] - ufa[i,j,k])/dx
                     + (vfa[i,j+1,k] - vfa[i,j,k])/dy
                     + (wfa[i,j,k+1] - wfa[i,j,k])/dz )

        # Field b
        tets,ind = NS.cell2tets_withProject_uvwf(i,j,k,ufb,vfb,wfb,dts[n],mesh)
        vol1 = dx*dy*dz
        vol2 = NS.tets_vol(tets)
        divb[n] = (vol1-vol2)/dts[n]
        divb_fd[n] = ( (ufb[i+1,j,k] - ufb[i,j,k])/dx
                     + (vfb[i,j+1,k] - vfb[i,j,k])/dy
                     + (wfb[i,j,k+1] - wfb[i,j,k])/dz )

        # Field a + b
        tets,ind = NS.cell2tets_withProject_uvwf(i,j,k,ufa.+ufb,vfa.+vfb,wfa.+wfb,dts[n],mesh)
        vol1 = dx*dy*dz
        vol2 = NS.tets_vol(tets)
        divab[n] = (vol1-vol2)/dts[n]
        divab_fd[n] = ( ( (ufa[i+1,j,k] + ufb[i+1,j,k]) - (ufa[i,j,k] + ufb[i,j,k]) )/dx
                      + ( (vfa[i,j+1,k] + vfb[i,j+1,k]) - (vfa[i,j,k] + vfb[i,j,k]) )/dy
                      + ( (wfa[i,j,k+1] + wfb[i,j,k+1]) - (wfa[i,j,k] + wfb[i,j,k]) )/dz )
    
        # Output 
        error = abs((diva[n]+divb[n]) - divab[n])
        @printf(" %+6.3e  %+6.3e  %+6.3e  %+6.3e  %+6.3e  %6.3e \n",
            dts[n],diva[n],divb[n],diva[n]+divb[n],divab[n],error)

        # Plot 
        error = abs((diva[n]+divb[n]) - divab[n])
        fig1 = plot(legend=false)
        fig2 = plot(legend=false)
        fig3 = plot(legend=false)
        fig4 = plot()
        fig5 = plot()
        fig6 = plot()
        fig7 = plot()
        plotGrid(fig1, (ufa    ,), (vfa    ,), (wfa    ,),     diva,  dts[n], mesh, title=@sprintf("Velocity A: Divg = %4.3f",diva[n]) ,color=:blue  )
        plotGrid(fig2, (    ufb,), (    vfb,), (    wfb,),     divb,  dts[n], mesh, title=@sprintf("Velocity B: Divg = %4.3f",divb[n])       ,color=:blue  )
        plotGrid(fig3, (ufa+ufb,), (vfa+vfb,), (wfa+wfb,),     divab, dts[n], mesh, title=@sprintf("Velocity A+B: Divg = %4.3f",divab[n]),color=:blue  )
        plot!(fig4,dts[1:n],diva[1:n],label="∇⋅A",                         xlim=[0,maximum(dts)]) #,ylim=[-2,1])
        plot!(fig4,dts[1:n],diva_fd[1:n],label="∇⋅A_fd",                   xlim=[0,maximum(dts)],ylim=[-2.7,-1.9])
        plot!(fig5,dts[1:n],divb[1:n],label="∇⋅B",                         xlim=[0,maximum(dts)]) #,ylim=[-2,1])
        plot!(fig5,dts[1:n],divb_fd[1:n],label="∇⋅B_fd",                   xlim=[0,maximum(dts)],ylim=[1.3,2.1])
        plot!(fig6,dts[1:n],divab[1:n],label="∇⋅(A+B)",                    xlim=[0,maximum(dts)]) #,ylim=[-2,1])
        plot!(fig6,dts[1:n],divab_fd[1:n],label="∇⋅(A+B)_fd",              xlim=[0,maximum(dts)]) #,ylim=[-2,1])
        plot!(fig6,dts[1:n],diva[1:n].+divb[1:n],label="∇⋅A + ∇⋅B",l=:dash,xlim=[0,maximum(dts)],ylim=[-1.3,0.1])
        if n>=2
            fig7 = plot(fig7,legend=:bottomright)
            plot!(fig7,dts[2:n],((diva[2:n].-diva[1:n-1])./(dts[2:n].-dts[1:n-1])),label="d(∇⋅A)/(dΔt)",xlim=[0,maximum(dts)]) #,ylim=[-2,1])
            plot!(fig7,dts[2:n],((divb[2:n].-divb[1:n-1])./(dts[2:n].-dts[1:n-1])),label="d(∇⋅B)/(dΔt)",xlim=[0,maximum(dts)]) #,ylim=[-2,1])
        end
        myplt=plot(fig1,fig2,fig3,fig4,fig5,fig6,fig7,
                    layout=(3,3),
                    size=(1000,800),
                    #plot_title=@sprintf("Div(A) + Divg(B) = %4.3f,  Div(A+B) = %4.3f, Error = %4.3f",diva[n]+divb[n],divab[n],error),
                    top_margin=10mm, 
        )
        display(myplt)
    end

    gif(anim, "animation.gif", fps = 15)

    # Plot volumes vs time 
    fig = plot() 
    fig = plot!(fig,dts,diva ,label="∇⋅A",line = :dash)
    fig = plot!(fig,dts,divb ,label="∇⋅B",line = :dash)
    fig = plot!(fig,dts,diva .+ divb ,label="∇⋅A + ∇⋅B")
    fig = plot!(fig,dts,divab,label="∇⋅(A+B)")
    fig = plot!(legend=:bottomleft,
                size=(400,300),
                xlabel="Δt",
                ylabel="Divergence",
                )
    savefig("divVsTime.pdf")

    # # Plots
    # fig1 = plot(legend=false)
    # fig2 = plot(legend=false)
    # fig3 = plot(legend=false)
    # fig4 = plot(legend=false)
    # plotGrid(fig1, (ufa    ,), (vfa    ,), (wfa    ,),     diva,  dt, mesh, title="Velocity A"               ,color=:blue  )
    # plotGrid(fig2, (    ufb,), (    vfb,), (    wfb,),     divb,  dt, mesh, title="Velocity B"               ,color=:blue  )
    # plotGrid(fig3, (ufa+ufb,), (vfa+vfb,), (wfa+wfb,),     divab, dt, mesh, title="Velocity (A + B)"         ,color=:blue  )
    # plotGrid(fig4, (ufa,ufb,), (vfa,vfb,), (wfa,wfb,), diva+divb, dt, mesh, title="Velocity A + Velocity B"  ,color=:blue  )
    # myplt=plot(fig1,fig2,fig3,fig4,
    #             layout=(2,2),
    #             size=(1000,900),
    # )
    # display(myplt)
    # savefig("SLdivergence1.pdf")

    # # Plot of Velocity (A+B) and Velocity A + Velocity B on same figure
    # fig1 = plot(legend=false,size=(1000,900))
    # plotGrid(fig1, (ufa+ufb,), (vfa+vfb,), (wfa+wfb,),     divab, dt, mesh, title=""         ,color=:blue  )
    # plotGrid(fig1, (ufa,ufb,), (vfa,vfb,), (wfa,wfb,), diva+divb, dt, mesh, title="Velocity (A + B) & Velocity A + Velocity B"  ,color=:blue  )
    # display(fig1)
    # savefig("SLdivergence2.pdf")


    # fig1 = plot(legend=false)
    # plotGrid(fig1, (ufa    ,), (vfa    ,), (wfa    ,),     diva,  dt, mesh, title="Velocity A"               ,color=:blue  )
    # plotGrid(fig1, (    ufb,), (    vfb,), (    wfb,),     divb,  dt, mesh, title="Velocity B"               ,color=:red  )
    # plotGrid(fig1, (ufa+ufb,), (vfa+vfb,), (wfa+wfb,),     divab, dt, mesh, title="Velocity (A + B)"         ,color=:green  )
    # myplt=plot(fig1,
    #             size=(1000,900),
    # )
    # display(myplt)
    # savefig("SLdivergence3.pdf")

    
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
        plot_cell(fig,pt1,pt2,pt3,pt4,title,(:red,:dash,2,0.5))

        # projected points 
        pt1_p = pt1
        pt2_p = pt2
        pt3_p = pt3
        pt4_p = pt4

        # Lines for porjections 
        lines=[:black,:green]
        for n in eachindex(uf)
            
            # Project 
            pt1_pnew = pt1_p + NS.project_uvwf(pt1,i,j,k,uf[n],vf[n],wf[n],-dt,mesh)-pt1
            pt2_pnew = pt2_p + NS.project_uvwf(pt2,i,j,k,uf[n],vf[n],wf[n],-dt,mesh)-pt2
            pt3_pnew = pt3_p + NS.project_uvwf(pt3,i,j,k,uf[n],vf[n],wf[n],-dt,mesh)-pt3
            pt4_pnew = pt4_p + NS.project_uvwf(pt4,i,j,k,uf[n],vf[n],wf[n],-dt,mesh)-pt4

            # Plot projection 
            GR.setarrowsize(0.5)
            plot!([pt1_p[1],pt1_pnew[1]],[pt1_p[2],pt1_pnew[2]],arrow=(:closed), arrowsize=5, line=lines[n], label="")
            plot!([pt2_p[1],pt2_pnew[1]],[pt2_p[2],pt2_pnew[2]],arrow=(:closed), arrowsize=5, line=lines[n], label="")
            plot!([pt3_p[1],pt3_pnew[1]],[pt3_p[2],pt3_pnew[2]],arrow=(:closed), arrowsize=5, line=lines[n], label="")
            plot!([pt4_p[1],pt4_pnew[1]],[pt4_p[2],pt4_pnew[2]],arrow=(:closed), arrowsize=5, line=lines[n], label="")

            # Update points 
            pt1_p = pt1_pnew
            pt2_p = pt2_pnew
            pt3_p = pt3_pnew
            pt4_p = pt4_pnew

        end

        # Plot projected cell 
        # Color by divergence
        # divrange=range(-5e-2, 5e-2, length=100)
        # colors=range(colorant"blue", stop=colorant"red", length=100)
        # value,index = findmin(abs.(divrange .- div[n]))
        # fillcolor=colors[index]
        # plot_cell(fig,pt1_p,pt2_p,pt3_p,pt4_p,title,color; fill=true, fillcolor=fillcolor)
        plot_cell(fig,pt1_p,pt2_p,pt3_p,pt4_p,title,color; fill=true, fillcolor=color)

    end
    return fig
end

function plot_cell(fig,pt1,pt2,pt3,pt4,title,linestyle; fill=false, fillcolor=nothing)
    # Plot this cell 
    fig = plot!(fig,
            #title=title, 
            aspect_ratio=:equal,
            xlim=(-0.5,1.5),
            ylim=(-0.5,1.5),
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