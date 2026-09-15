
using NavierStokes_Parallel
using Printf
using OffsetArrays
using CairoMakie

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
        grav_x = 0.0,
        grav_y = 0.0,
        grav_z = 0.0, # Gravity (m/s^2)
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

        projection_method = "Heun",

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
    
    # pre-image arrays
    tetsa  = Array{Float64}(undef, 3, 4, 24); fill!(tetsa,0.0)
    indsa  = Array{Int32}(undef, 3, 4, 24); fill!(indsa,0.0)
    vertsa = Array{Float64}(undef, 3, 8)
    vIndsa = Array{Int32}(undef, 3, 8)

    tetsb  = Array{Float64}(undef, 3, 4, 24); fill!(tetsb,0.0)
    indsb  = Array{Int32}(undef, 3, 4, 24); fill!(indsb,0.0)
    vertsb = Array{Float64}(undef, 3, 8)
    vIndsb = Array{Int32}(undef, 3, 8)

    tetsab  = Array{Float64}(undef, 3, 4, 24); fill!(tetsab,0.0)
    indsab  = Array{Int32}(undef, 3, 4, 24); fill!(indsab,0.0)
    vertsab = Array{Float64}(undef, 3, 8)
    vIndsab = Array{Int32}(undef, 3, 8)
    #####################################
    # Play with different velocity fields
    #####################################
    for case in ["A"]
    # for case in ["A","B","C","D"]    
        # Set velocity fields 
        for k=kmino_:kmaxo_, j=jmino_:jmaxo_, i=imino_:imaxo_

            if case == "A"
                # This velocity field works because 1 of the area dimensions remains constant
                # Therefore the change in volume is linear w.r.t. time. 

                # Field a
                ufa[i,j,k] = -(ym[j]-1.5);  vfa[i,j,k] =  (xm[i]-1.5); wfa[i,j,k] = 0.0     # ω = +1
                # ufb[i,j,k] =  (ym[j]-1.5);  vfb[i,j,k] = -(xm[i]-1.5); wfb[i,j,k] = 0.0     # ω = −1
                ufb[i,j,k] = 0;            vfb[i,j,k] = (xm[i]-1.5); wfb[i,j,k] = 0.0
                # ufa[i,j,k] = ym[j]
                # vfa[i,j,k] = 0.0
                # wfa[i,j,k] = 0.0


                # # Field b
                # ufb[i,j,k] = 0.0
                # vfb[i,j,k] = xm[i]
                # wfb[i,j,k] = 0.0

            elseif case == "B"
                # This velocity field does not work because the area changes 
                # non-linearly w.r.t. time. 

                # Field a
                # ufa[i,j,k] = y[j]
                # vfa[i,j,k] = 0.0
                # wfa[i,j,k] = 0.0
                # # Field b
                # ufb[i,j,k] = 0.0
                # vfb[i,j,k] = x[i]
                # wfb[i,j,k] = 0.0

                ufa[i,j,k] = (ym[j]-1.5);  vfa[i,j,k] = 0; wfa[i,j,k] = 0.0
                ufb[i,j,k] = 0;            vfb[i,j,k] = (xm[i]-1.5); wfb[i,j,k] = 0.0

                                
                # ufa[i,j,k] = -(y[j]-1.5)
                # vfa[i,j,k] =  (x[i]-1.5)
                
                # # Field b
                # ufb[i,j,k] = x[i]-0.25
                # vfb[i,j,k] = y[j]-0.25
                
            elseif case == "C"
                # This velocity field does not work because the area changes 
                # non-linearly w.r.t. time. 

                ufa[i,j,k] = 1.0*(x[i]-1.5);  vfa[i,j,k] = 1.0*(y[j]-1.5); wfa[i,j,k] = 0.0     # d = 1
                ufb[i,j,k] = 0.5*(x[i]-1.5);  vfb[i,j,k] = 0.5*(y[j]-1.5); wfb[i,j,k] = 0.0     # d = 0.5

                # # Field a
                # ufa[i,j,k] = -(ym[j]-1.5)
                # vfa[i,j,k] =  (xm[i]-1.5)
                # wfa[i,j,k] = 0.0
                # # Field b
                # ufb[i,j,k] =  2*(ym[j]-1.5) #x[i]-0.25
                # vfb[i,j,k] = 2*(xm[i]-1.5) #y[j]-0.25
                # wfb[i,j,k] = 0.0

            elseif case == "D"
                # This has a different answer from finite difference, but
                # ∇⋅(A+B) = ∇⋅A + ∇⋅B 

                ufa[i,j,k] =  (x[i]-1.5);                        vfa[i,j,k] = -(y[j]-1.5); wfa[i,j,k] = 0.0
                ufb[i,j,k] =  0.5*(x[i]-1.5) + 0.866*(ym[j]-1.5); vfb[i,j,k] = 0.866*(xm[i]-1.5) - 0.5*(y[j]-1.5); wfb[i,j,k] = 0.0

                # # Field a
                # ufa[i,j,k] = -(y[j]-1.5)
                # vfa[i,j,k] =  (x[i]-1.5)
                # wfa[i,j,k] = 0.0
                # # Field b
                # ufb[i,j,k] = x[i]-0.25
                # vfb[i,j,k] = y[j]-0.25
                # wfb[i,j,k] = 0.0
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
            tetsigna = NS.cell2tets!(vertsa,tetsa,i,j,k,param,mesh; 
                project_verts=true,uf=ufa,vf=vfa,wf=wfa,dt=dts[n])
            vol1 = dx*dy*dz
            vol2a[n] = NS.tets_vol(tetsa)*tetsigna
            diva[n] = (vol1-vol2a[n])/dts[n]/vol1
            diva_fd[n] = ( (ufa[i+1,j,k] - ufa[i,j,k])/dx
                         + (vfa[i,j+1,k] - vfa[i,j,k])/dy
                         + (wfa[i,j,k+1] - wfa[i,j,k])/dz )

            # Field b
            tetsignb = NS.cell2tets!(vertsb,tetsb,i,j,k,param,mesh; 
                project_verts=true,uf=ufb,vf=vfb,wf=wfb,dt=dts[n])
            vol1 = dx*dy*dz
            vol2b[n] = NS.tets_vol(tetsb)*tetsignb
            divb[n] = (vol1-vol2b[n])/dts[n]/vol1
            divb_fd[n] = ( (ufb[i+1,j,k] - ufb[i,j,k])/dx
                         + (vfb[i,j+1,k] - vfb[i,j,k])/dy
                         + (wfb[i,j,k+1] - wfb[i,j,k])/dz )

            # Field a + b
            tetsignab = NS.cell2tets!(vertsab,tetsab,i,j,k,param,mesh; 
                project_verts=true,uf=ufa.+ufb,vf=vfa.+vfb,wf=wfa.+wfb,dt=dts[n])
            vol1 = dx*dy*dz
            vol2ab[n] = NS.tets_vol(tetsab)*tetsignab
            divab[n] = (vol1-vol2ab[n])/dts[n]/vol1
            divab_fd[n] = ( ( (ufa[i+1,j,k] + ufb[i+1,j,k]) - (ufa[i,j,k] + ufb[i,j,k]) )/dx
                          + ( (vfa[i,j+1,k] + vfb[i,j+1,k]) - (vfa[i,j,k] + vfb[i,j,k]) )/dy
                          + ( (wfa[i,j,k+1] + wfb[i,j,k+1]) - (wfa[i,j,k] + wfb[i,j,k]) )/dz )

        end

        ####################
        # Output 
        ####################
        # ---- Limits (fixed for the whole animation) ----
        xmin_c = x[imin_  ] - 2param.CFL*dx
        xmax_c = x[imax_+1] + 2param.CFL*dx
        ymin_c = y[jmin_  ] - 2param.CFL*dy
        ymax_c = y[jmax_+1] + 2param.CFL*dy

        # Convective CFL of the most restrictive field (A, B or A+B), so all
        # three columns share one abscissa and stay comparable
        u_max = maximum([maximum(abs.(ufa)),maximum(abs.(ufb)),maximum(abs.(ufa.+ufb))])
        v_max = maximum([maximum(abs.(vfa)),maximum(abs.(vfb)),maximum(abs.(vfa.+vfb))])
        w_max = maximum([maximum(abs.(wfa)),maximum(abs.(wfb)),maximum(abs.(wfa.+wfb))])
        cfls  = dts .* maximum([u_max/dx, v_max/dy, w_max/dz])

        # Divergence limits
        dmin = minimum([minimum(diva),minimum(divb),minimum(diva+divb),minimum(divab),
                        minimum(diva_fd),minimum(divb_fd),minimum(divab_fd)])
        dmax = maximum([maximum(diva),maximum(divb),maximum(diva+divb),maximum(divab),
                        maximum(diva_fd),maximum(divb_fd),maximum(divab_fd)])
        padd = 0.05*(dmax-dmin); dmin -= padd; dmax += padd

        # Volume change relative to the undeformed cell, ΔV/V₁ = V/V₁ - 1, so
        # the ticks carry the same exponent as the divergence panel.
        # The FD reference is the pre-image volume that would make the
        # semi-Lagrangian divergence match the finite-difference one, so its
        # change is just -dt*div_fd.
        vol1       = dx*dy*dz
        dvol_a     = vol2a  ./ vol1 .- 1.0
        dvol_b     = vol2b  ./ vol1 .- 1.0
        dvol_ab    = vol2ab ./ vol1 .- 1.0
        dvol_apb   = dvol_a .+ dvol_b
        dvol_ab_fd = -dts .* divab_fd
        vmin = minimum([minimum(dvol_apb),minimum(dvol_ab),minimum(dvol_ab_fd)])
        vmax = maximum([maximum(dvol_apb),maximum(dvol_ab),maximum(dvol_ab_fd)])
        padv = 0.05*(vmax-vmin); vmin -= padv; vmax += padv

        # ---- Series styles (matching papers/paper1/common.jl) ----
        # Data series: default Makie cycle colors, dotted line, distinct markers
        # Reference series: gray dashed, as in add_refslopes!
        pal     = Makie.wong_colors()
        sl_pts  = (color=pal[1], linewidth=2, linestyle=:dot,
                   marker=:circle,  markersize=8, strokewidth=0)
        sl_pts2 = (color=pal[2], linewidth=2, linestyle=:dot,
                   marker=:diamond, markersize=9, strokewidth=0)
        fd_ref  = (color=:gray, linewidth=2, linestyle=:dash)

        # ---- Figure built once, contents updated each frame ----
        fig = Figure(size=(1000,600), figure_padding=16)

        # Top row: projected cells (redrawn each frame), in their own layout so
        # the panels' y-decorations below cannot skew the spacing between cells
        gtop = GridLayout(fig[1,1:6])
        axcell = [Axis(gtop[1,c], aspect=DataAspect(), xgridvisible=false, ygridvisible=false,limits=(xmin_c,xmax_c,ymin_c,ymax_c),
                       yticklabelsvisible=false,xticklabelsvisible=false) for c in 1:3]
        # axcell = [Axis(fig[1,(2c-1):(2c)], aspect=DataAspect(), xgridvisible=false, ygridvisible=false,
        #                limits=(xmin_c,xmax_c,ymin_c,ymax_c)) for c in 1:3]

        # Data panels: axis with an outer-top legend, via a nested layout
        function panel(cols,ylims; ylabel="", ytickformat=Makie.automatic)
            ax = Axis(fig[2,cols]; xlabel="CFL", ylabel, ytickformat,
                      limits=((0.0,maximum(cfls)), ylims),ylabelsize = 18, xlabelsize = 18)
            return ax
        end
        # function panel(cols,ylims; ylabel="", ytickformat=Makie.automatic)
        #     gl = GridLayout(fig[2,cols])
        #     ax = Axis(gl[2,1]; xlabel="CFL", ylabel, ytickformat,
        #               limits=((0.0,maximum(cfls)), ylims))
        #     return gl, ax
        # end
        ax6 = panel(2:3,(dmin,dmax); ylabel="∇⋅u")
        # gl5,ax5 = panel(2,2,(dmin,dmax))
        # gl6,ax6 = panel(2,3,(dmin,dmax))
        ax9 = panel(4:5,(vmin,vmax); ylabel="ΔV/V₁")
        # gl8,ax8 = panel(3,2,(vmin,vmax);                ytickformat=voltick)
        # gl9,ax9 = panel(3,3,(vmin,vmax);                ytickformat=voltick)

        # Growing SL series (updated in the record loop)
        # sl_diva   = Observable(Point2f[])
        # sl_divb   = Observable(Point2f[])
        sl_divab  = Observable(Point2f[])
        sl_divapb = Observable(Point2f[])
        # sl_vola   = Observable(Point2f[])
        # sl_volb   = Observable(Point2f[])
        sl_volab  = Observable(Point2f[])
        sl_volapb = Observable(Point2f[])

        # Divergence vs time
        # scatterlines!(ax4, sl_diva   ; label="∇⋅A (SL)",       sl_pts... )
        # hlines!(      ax4, [diva_fd[1]]  ; label="∇⋅A (FD)",       fd_ref... )
        # scatterlines!(ax5, sl_divb   ; label="∇⋅B (SL)",       sl_pts... )
        # hlines!(      ax5, [divb_fd[1]]  ; label="∇⋅B (FD)",       fd_ref... )
        scatterlines!(ax6, sl_divab  ; label="∇⋅(F₁+F₂) (SL)",   sl_pts... )
        scatterlines!(ax6, sl_divapb ; label="∇⋅F₁ + ∇⋅F₂ (SL)", sl_pts2...)
        hlines!(      ax6, [divab_fd[1]] ; label="∇⋅(F₁+F₂) (FD)",   fd_ref... )

        # Volume change vs CFL
        # scatterlines!(ax7, sl_vola   ; label="V(A)/V₁ (SL)",               sl_pts... )
        # lines!(       ax7, cfls, vol2a_fd_n  ; label="V(A)/V₁ (FD)",           fd_ref... )
        # scatterlines!(ax8, sl_volb   ; label="V(B)/V₁ (SL)",               sl_pts... )
        # lines!(       ax8, cfls, vol2b_fd_n  ; label="V(B)/V₁ (FD)",           fd_ref... )        
        scatterlines!(ax9, sl_volab  ; label="ΔV(F₁+F₂)/V₁ (SL)",           sl_pts...)
        scatterlines!(ax9, sl_volapb ; label="ΔV(F₁)/V₁ + ΔV(F₂)/V₁ (SL)",  sl_pts2... )
        lines!(       ax9, cfls, dvol_ab_fd ; label="ΔV(F₁+F₂)/V₁ (FD)",        fd_ref... )

        # for (gl,ax) in ((gl4,ax4),(gl5,ax5),(gl6,ax6),(gl7,ax7),(gl8,ax8),(gl9,ax9))
        # for (gl,ax) in ((gl6,ax6),(gl9,ax9))
        #     Legend(gl[1,1], ax; framevisible=false, orientation=:horizontal, nbanks=2,
        #            labelsize=10, patchsize=(18,10), padding=(0,0,0,0),
        #            tellheight=true, tellwidth=false)
        #     rowgap!(gl, 4)
        # end
        for ax in (ax6, ax9)
            axislegend(ax; position=:lc,
                       labelsize=13, patchsize=(18,10), padding=(6,6,6,6))
        end
        # Equal columns: the cell axes above and the panels below share one
        # geometry, instead of the panels' y-decorations setting column widths
        foreach(c -> colsize!(fig.layout, c, Relative(1/6)), 1:6)

        # Row 1 height = two column widths, so the DataAspect cells are exactly
        # square with no leftover vertical space
        rowsize!(fig.layout, 1, Aspect(1, 2.0))

        # ---- Animate ----
        record(fig, "divVsTime_Velocity$case.gif", eachindex(dts); framerate=5) do n
            err = abs((diva[n]+divb[n]) - divab[n])
            @printf(" %+6.3e  %+6.3e  %+6.3e  %+6.3e  %+6.3e  %+6.3e  %6.3e \n",
                dts[n],diva[n],diva_fd[n],divb[n],diva[n]+divb[n],divab[n],err)

            # Projected cells
            for (ax,fld) in zip(axcell, ((ufa,vfa,wfa),
                                         (ufb,vfb,wfb),
                                         (ufa.+ufb,vfa.+vfb,wfa.+wfb)))
                empty!(ax)
                plotGrid(ax, fld[1], fld[2], fld[3], dts[n], param, mesh, color=pal[1])
            end

            # Growing SL series
            # sl_diva[]   = Point2f.(cfls[1:n], diva[1:n])
            # sl_divb[]   = Point2f.(cfls[1:n], divb[1:n])
            sl_divab[]  = Point2f.(cfls[1:n], divab[1:n])
            sl_divapb[] = Point2f.(cfls[1:n], diva[1:n].+divb[1:n])
            # sl_vola[]   = Point2f.(cfls[1:n], vol2a_n[1:n])
            # sl_volb[]   = Point2f.(cfls[1:n], vol2b_n[1:n])
            sl_volab[]  = Point2f.(cfls[1:n], dvol_ab[1:n])
            sl_volapb[] = Point2f.(cfls[1:n], dvol_apb[1:n])
        end

        # Save final figure
        save("divVsTime_Velocity$case.png", fig)

        
    end
end

function plotGrid(ax,uf,vf,wf,dt,param,mesh; color=Makie.wong_colors()[1], cellcolor=:gray)
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

        pt1_p = copy(pt1)
        pt2_p = copy(pt2)
        pt3_p = copy(pt3)
        pt4_p = copy(pt4)

        NS.project!(pt1_p,i,j,k,uf,vf,wf,dt,param,mesh)
        NS.project!(pt2_p,i,j,k,uf,vf,wf,dt,param,mesh)
        NS.project!(pt3_p,i,j,k,uf,vf,wf,dt,param,mesh)
        NS.project!(pt4_p,i,j,k,uf,vf,wf,dt,param,mesh)

        # Plot cell 
        plot_cell(ax,pt1,pt2,pt3,pt4; color=cellcolor, linestyle=:dash, linewidth=2, alpha=0.8)
    
        # Plot projection 
        # arrows2d! takes origins + direction vectors (arrow size is in pixels)
        origins = [Point2f(p[1],p[2]) for p in (pt1,pt2,pt3,pt4)]
        dirs    = [Vec2f(q[1]-p[1],q[2]-p[2]) for (p,q) in
                   ((pt1,pt1_p),(pt2,pt2_p),(pt3,pt3_p),(pt4,pt4_p))]
        arrows2d!(ax, origins, dirs; color=:black,
                  tiplength=6, tipwidth=7, shaftwidth=1.5)

        # Plot projected cell 
        plot_cell(ax,pt1_p,pt2_p,pt3_p,pt4_p; color=color, linewidth=2, fill=true)

    end
    return ax
end

function plot_cell(ax,pt1,pt2,pt3,pt4; color=:black, linestyle=:solid, linewidth=2,
                   alpha=1.0, fill=false, fillalpha=0.25)
    # Corners in loop order: pt1 → pt2 → pt4 → pt3 (→ pt1)
    loop = [Point2f(p[1],p[2]) for p in (pt1,pt2,pt4,pt3,pt1)]
    if fill 
        poly!(ax, loop[1:4]; color=(color,fillalpha), strokewidth=0)
    end
    lines!(ax, loop; color=(color,alpha), linestyle=linestyle, linewidth=linewidth)

    return ax
end


# Run test
test_SLdivergence()