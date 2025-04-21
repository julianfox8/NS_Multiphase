
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
        mu_liq=1e-5,       # Dynamic viscosity
        mu_gas=1e-5,       # Dynamic viscosity
        rho_liq=1.0,           # Density
        rho_gas=1.0,           # Density
        sigma = 1e-2,
        gravity = 1,
        Lx=3.0,            # Domain size
        Ly=3.0,
        Lz=3.0,
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
        test_case = "DIv_test"
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
    # for case in ["C"]
    for case in ["A","B","C","D"]    
        # Set velocity fields 
        for k=kmino_:kmaxo_, j=jmino_:jmaxo_, i=imino_:imaxo_

            if case == "A"
                # This velocity field works because 1 of the area dimensions remains constant
                # Therefore the change in volume is linear w.r.t. time. 

                # Field a
                ufa[i,j,k] = ym[j]
                vfa[i,j,k] = 0.0
                wfa[i,j,k] = 0.0


                # Field b
                ufb[i,j,k] = 0.0
                vfb[i,j,k] = xm[i]
                wfb[i,j,k] = 0.0

            elseif case == "B"
                # This velocity field does not work because the area changes 
                # non-linearly w.r.t. time. 

                # Field a
                ufa[i,j,k] = y[j]
                vfa[i,j,k] = 0.0
                wfa[i,j,k] = 0.0
                # Field b
                ufb[i,j,k] = 0.0
                vfb[i,j,k] = x[i]
                wfb[i,j,k] = 0.0

            elseif case == "C"
                # This velocity field does not work because the area changes 
                # non-linearly w.r.t. time. 

                # Field a
                ufa[i,j,k] = -(ym[j]-1.5)
                vfa[i,j,k] =  (xm[i]-1.5)
                wfa[i,j,k] = 0.0
                # Field b
                ufb[i,j,k] =  2*(ym[j]-1.5) #x[i]-0.25
                vfb[i,j,k] = 2*(xm[i]-1.5) #y[j]-0.25
                wfb[i,j,k] = 0.0

            elseif case == "D"
                # This has a different answer from finite difference, but
                # ∇⋅(A+B) = ∇⋅A + ∇⋅B 

                # Field a
                ufa[i,j,k] = -(y[j]-1.5)
                vfa[i,j,k] =  (x[i]-1.5)
                wfa[i,j,k] = 0.0
                # Field b
                ufb[i,j,k] = x[i]-0.25
                vfb[i,j,k] = y[j]-0.25
                wfb[i,j,k] = 0.0
            else 
                error("Unknown velocity field specified")
            end
        end

        # Set timestep 
        dta = NS.compute_dt(ufa,vfa,wfa,param,mesh,par_env)
        dtb = NS.compute_dt(ufb,vfb,wfb,param,mesh,par_env)
        dt = minimum([dta,dtb])

        # Output header 
        @printf("%10s  %10s  %10s  %10s  %10s  %10s  %9s \n",
                "Δt","∇⋅A","∇⋅A_fd","∇⋅B","∇⋅A + ∇⋅B","∇⋅(A+B)", "Error")
            

        dts=0.01:0.05:2dt
        diva  = similar(dts)
        divb  = similar(dts)
        divab = similar(dts)
        diva_fd  = similar(dts)
        divb_fd  = similar(dts)
        divab_fd = similar(dts)
        vol2a  = similar(dts)
        vol2b  = similar(dts)
        vol2ab = similar(dts)
        for n in eachindex(dts)

            # Only work with 1 cell 
            i=imin_;j=jmin_;k=kmin_

            ##########################
            # Compute divergences 
            ##########################

            # Field a 
            tets,ind = NS.cell2tets_withProject_uvwf(i,j,k,ufa,vfa,wfa,dts[n],mesh)
            vol1 = dx*dy*dz
            vol2a[n] = NS.tets_vol(tets)
            diva[n] = (vol1-vol2a[n])/dts[n]/vol1
            diva_fd[n] = ( (ufa[i+1,j,k] - ufa[i,j,k])/dx
                         + (vfa[i,j+1,k] - vfa[i,j,k])/dy
                         + (wfa[i,j,k+1] - wfa[i,j,k])/dz )

            # Field b
            tets,ind = NS.cell2tets_withProject_uvwf(i,j,k,ufb,vfb,wfb,dts[n],mesh)
            vol1 = dx*dy*dz
            vol2b[n] = NS.tets_vol(tets)
            divb[n] = (vol1-vol2b[n])/dts[n]/vol1
            divb_fd[n] = ( (ufb[i+1,j,k] - ufb[i,j,k])/dx
                         + (vfb[i,j+1,k] - vfb[i,j,k])/dy
                         + (wfb[i,j,k+1] - wfb[i,j,k])/dz )

            # Field a + b
            tets,ind = NS.cell2tets_withProject_uvwf(i,j,k,ufa.+ufb,vfa.+vfb,wfa.+wfb,dts[n],mesh)
            vol1 = dx*dy*dz
            vol2ab[n] = NS.tets_vol(tets)
            divab[n] = (vol1-vol2ab[n])/dts[n]/vol1
            divab_fd[n] = ( ( (ufa[i+1,j,k] + ufb[i+1,j,k]) - (ufa[i,j,k] + ufb[i,j,k]) )/dx
                          + ( (vfa[i,j+1,k] + vfb[i,j+1,k]) - (vfa[i,j,k] + vfb[i,j,k]) )/dy
                          + ( (wfa[i,j,k+1] + wfb[i,j,k+1]) - (wfa[i,j,k] + wfb[i,j,k]) )/dz )

        end

        ####################
        # Output 
        ####################
        myplt = plot()
        anim = @animate for n in eachindex(dts)
            error = abs((diva[n]+divb[n]) - divab[n])
            @printf(" %+6.3e  %+6.3e  %+6.3e  %+6.3e  %+6.3e  %+6.3e  %6.3e \n",
                dts[n],diva[n],diva_fd[n],divb[n],diva[n]+divb[n],divab[n],error)

            # Plot of projected cell 
            xmin = x[imin_  ] - 2param.CFL*dx
            xmax = x[imax_+1] + 2param.CFL*dx
            ymin = y[jmin_  ] - 2param.CFL*dy
            ymax = y[jmax_+1] + 2param.CFL*dy
            fig1 = plot(legend=false, xlim=(xmin,xmax), ylim=(ymin,ymax),)
            fig2 = plot(legend=false, xlim=(xmin,xmax), ylim=(ymin,ymax),)
            fig3 = plot(legend=false, xlim=(xmin,xmax), ylim=(ymin,ymax),)
            plotGrid(fig1, ufa    , vfa    , wfa    , dts[n], mesh,color=:blue  )
            plotGrid(fig2,     ufb,     vfb,     wfb, dts[n], mesh,color=:blue  )
            plotGrid(fig3, ufa+ufb, vfa+vfb, wfa+wfb, dts[n], mesh,color=:blue  )

            # Plot of divergence vs time
            ymin = minimum([minimum(diva),minimum(divb),minimum(diva+divb),minimum(divab)])
            ymax = maximum([maximum(diva),maximum(divb),minimum(diva+divb),maximum(divab)])
            ymin = minimum([ymin,minimum(diva_fd),minimum(divb_fd),minimum(divab_fd)])
            ymax = maximum([ymax,maximum(diva_fd),maximum(divb_fd),maximum(divab_fd)])
            pady = 0.05*(ymax-ymin)
            ymin -= pady 
            ymax += pady
            fig4 = plot(legend=:outertop,foreground_color_legend = nothing,xlim=[0,maximum(dts)], ylim=[ymin,ymax])
            fig5 = plot(legend=:outertop,foreground_color_legend = nothing,xlim=[0,maximum(dts)], ylim=[ymin,ymax])
            fig6 = plot(legend=:outertop,foreground_color_legend = nothing,xlim=[0,maximum(dts)], ylim=[ymin,ymax])
            plot!(   fig4,dts[1:n],diva[1:n],label="∇⋅A (SL)",           lw=3              )
            scatter!(fig4,dts[1:n],diva_fd[1:n],label="∇⋅A (FD)",     marker=(:+,:black))
            plot!(   fig5,dts[1:n],divb[1:n],label="∇⋅B (SL)",           lw=3              )
            scatter!(fig5,dts[1:n],divb_fd[1:n],label="∇⋅B (FD)",     marker=(:+,:black))
            plot!(   fig6,dts[1:n],divab[1:n],label="∇⋅(A+B) (SL)",      lw=3              )
            plot!(   fig6,dts[1:n],diva[1:n].+divb[1:n],label="∇⋅A + ∇⋅B (SL)",l=:dash,lw=3)
            scatter!(fig6,dts[1:n],divab_fd[1:n],label="∇⋅(A+B) (FD)",marker=(:+,:black))
            
            # Plot of volumes vs time 
            # Needed volume is the projected volume to have the semi-Lagrangian divg match the finite diff divg
            vol2a_fd   = dx*dy*dz .* ( 1.0 .- dts.*diva_fd )
            vol2b_fd   = dx*dy*dz .* ( 1.0 .- dts.*divb_fd )
            vol2ab_fd  = dx*dy*dz .* ( 1.0 .- dts.*divab_fd )
            vol2apb_fd = dx*dy*dz .* ( 2.0 .- dts.*(diva_fd + divb_fd) )
            vol1 = dx*dy*dz
            ymin = minimum([minimum(vol2a),minimum(vol2b),minimum(vol2a+vol2b.-vol1),minimum(vol2ab)])
            ymax = maximum([maximum(vol2a),maximum(vol2b),maximum(vol2a+vol2b.-vol1),maximum(vol2ab)])
            ymin = minimum([ymin,minimum(vol2a_fd),minimum(vol2b_fd),minimum(vol2ab_fd)])
            ymax = maximum([ymax,maximum(vol2a_fd),maximum(vol2b_fd),maximum(vol2ab_fd)])
            pady = 0.05*(ymax-ymin)
            ymin -= pady 
            ymax += pady
            fig7 = plot(legend=:outertop,foreground_color_legend = nothing,xlim=[0,maximum(dts)], ylim=[ymin,ymax])
            fig8 = plot(legend=:outertop,foreground_color_legend = nothing,xlim=[0,maximum(dts)], ylim=[ymin,ymax])
            fig9 = plot(legend=:outertop,foreground_color_legend = nothing,xlim=[0,maximum(dts)], ylim=[ymin,ymax])
            plot!(   fig7,dts[1:n],vol2a[1:n]   ,lw=3,                  label="Vol(A)")
            scatter!(fig7,dts[1:n],vol2a_fd[1:n],marker=(:+,:black),    label="Vol(A)_fd")
            plot!(   fig8,dts[1:n],vol2b[1:n]   ,lw=3,                  label="Vol(B)")
            scatter!(fig8,dts[1:n],vol2b_fd[1:n],marker=(:+,:black),    label="Vol(B)_fd")
            plot!(   fig9,dts[1:n],vol2a[1:n]+vol2b[1:n].-vol1,lw=3,    label="(Vol(A) + Vol(B)) - vol1")
            plot!(   fig9,dts[1:n],vol2ab[1:n], l=:dash,lw=3,           label="Vol(A+B) ")
            scatter!(fig9,dts[1:n],vol2ab_fd[1:n],marker=(:+,:black),   label="Vol(A+B)_fd")
            #scatter!(fig9,dts[1:n],vol2apb_fd[1:n].-vol1,markershape=:circle,label="(Vol(A) + Vol(B))_fd - vol1")
            
            # Put all plots together
            myplt=plot(fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,
                        layout=(3,3),
                        size=(1000,1000),
                        #plot_title=@sprintf("Div(A) + Divg(B) = %4.3f,  Div(A+B) = %4.3f, Error = %4.3f",diva[n]+divb[n],divab[n],error),
                        top_margin=10mm, 
            )
            display(myplt)
        end
        # Save final figure
        savefig(myplt,"divVsTime_Velocity$case.pdf")

        # Save animation
        gif(anim, "divVsTime_Velocity$case.gif", fps = 15)

        
    end
end

function plotGrid(fig,uf,vf,wf,dt,mesh; color=:black)
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
        plot_cell(fig,pt1,pt2,pt3,pt4,(:red,:dash,2,0.5))
    
        # Project 
        pt1_p = pt1 + NS.project_uvwf(pt1,i,j,k,uf,vf,wf,-dt,mesh)-pt1
        pt2_p = pt2 + NS.project_uvwf(pt2,i,j,k,uf,vf,wf,-dt,mesh)-pt2
        pt3_p = pt3 + NS.project_uvwf(pt3,i,j,k,uf,vf,wf,-dt,mesh)-pt3
        pt4_p = pt4 + NS.project_uvwf(pt4,i,j,k,uf,vf,wf,-dt,mesh)-pt4

        # Plot projection 
        GR.setarrowsize(0.5)
        plot!([pt1[1],pt1_p[1]],[pt1[2],pt1_p[2]],arrow=(:closed), arrowsize=5, line=:black, label="")
        plot!([pt2[1],pt2_p[1]],[pt2[2],pt2_p[2]],arrow=(:closed), arrowsize=5, line=:black, label="")
        plot!([pt3[1],pt3_p[1]],[pt3[2],pt3_p[2]],arrow=(:closed), arrowsize=5, line=:black, label="")
        plot!([pt4[1],pt4_p[1]],[pt4[2],pt4_p[2]],arrow=(:closed), arrowsize=5, line=:black, label="")

        # Plot projected cell 
        plot_cell(fig,pt1_p,pt2_p,pt3_p,pt4_p,color; fill=true, fillcolor=color)

    end
    return fig
end

function plot_cell(fig,pt1,pt2,pt3,pt4,linestyle; fill=false, fillcolor=nothing)
    # Plot this cell 
    fig = plot!(fig,
            #title=title, 
            aspect_ratio=:equal,
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