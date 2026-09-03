# Serial implementation for full approximation scheme (FAS) multigrid method
"""
define the prolongation function (use trilinear interpolation for cell-centered and face-centered quantities)
"""
function prolong!(fine_field, coarse_field,fine_mesh,coarse_mesh)
    @unpack xm, ym, zm, imino_, imaxo_, jmino_, jmaxo_, kmino_, kmaxo_ = coarse_mesh
    @unpack x, y, z, imin_, imax_, jmin_, jmax_, kmin_, kmax_ = fine_mesh

    for k_f = kmin_:kmax_, j_f = jmin_:jmax_, i_f = imin_:imax_

        # Physical location of fine cell center
        pt = (fine_mesh.xm[i_f], fine_mesh.ym[j_f], fine_mesh.zm[k_f])
        
        i = clamp(i_f,imino_,imaxo_)
        j = clamp(j_f,jmino_,jmaxo_)
        k = clamp(k_f,kmino_,kmaxo_)

        # Find right i index on coarse mesh
        while pt[1]-xm[i  ] <  0.0 && i   > imino_
            i=i-1
        end
        while pt[1]-xm[i+1] >= 0.0 && i+1 < imaxo_
            i=i+1
        end
        # Find right j index
        while pt[2]-ym[j  ] <  0.0 && j   > jmino_
            j=j-1
        end
        while pt[2]-ym[j+1] >= 0.0 && j+1 < jmaxo_
            j=j+1
        end
        # Find right k index
        while pt[3]-zm[k  ] <  0.0 && k   > kmino_
            k=k-1
        end
        while pt[3]-zm[k+1] >= 0.0 && k+1 < kmaxo_
            k=k+1
        end

        # Trilinear interpolation weights
        wx1 = (pt[1] - xm[i]) / (xm[i+1] - xm[i]); wx2 = 1.0 - wx1
        wy1 = (pt[2] - ym[j]) / (ym[j+1] - ym[j]); wy2 = 1.0 - wy1
        wz1 = (pt[3] - zm[k]) / (zm[k+1] - zm[k]); wz2 = 1.0 - wz1
        
        # Interpolate scalar field
        fine_field[i_f, j_f, k_f] = (
            wz1*(wy1*(wx1*coarse_field[i+1,j+1,k+1] + wx2*coarse_field[i,j+1,k+1]) +
                 wy2*(wx1*coarse_field[i+1,j  ,k+1] + wx2*coarse_field[i,j  ,k+1])) +
            wz2*(wy1*(wx1*coarse_field[i+1,j+1,k  ] + wx2*coarse_field[i,j+1,k  ]) +
                 wy2*(wx1*coarse_field[i+1,j  ,k  ] + wx2*coarse_field[i,j  ,k  ]))
        )

    end
end


"""
Injection prolongation: each fine cell receives the value of its parent coarse cell directly.
Consistent with restrict! — since restrict! maps coarse index i to fine cells 2i-1 and 2i,
the inverse is i_c = (i_f + 1) ÷ 2.

This is the adjoint of the summation restriction (R = Pᵀ when P is injection), which is
the pair GeometricMultigrid.jl uses and for which standard MG convergence theory guarantees
h-independence. Use alongside restrict! (summation, no ÷8) for a consistent P/R pair.
"""
function prolong_inj!(fine_field, coarse_field, fine_mesh, coarse_mesh)
    @unpack imin_, imax_, jmin_, jmax_, kmin_, kmax_ = fine_mesh

    for k_f in kmin_:kmax_, j_f in jmin_:jmax_, i_f in imin_:imax_
        i_c = (i_f + 1) ÷ 2
        j_c = (j_f + 1) ÷ 2
        k_c = (k_f + 1) ÷ 2
        fine_field[i_f, j_f, k_f] = coarse_field[i_c, j_c, k_c]
    end
end

"""
define the restriction functions (with consistent 2nd order average)
uses 8-point average for cell-centered quantities and 4-point average for face-centered quantites
"""
function restrict!(coarse_field,fine_field,coarse_mesh,fine_mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        ii = 2i - 1
        jj = 2j - 1
        kk = 2k - 1

        coarse_field[i,j,k] = (
            fine_field[ii, jj, kk]     + fine_field[ii+1, jj, kk]   +
            fine_field[ii, jj+1, kk]   + fine_field[ii+1, jj+1, kk] +
            fine_field[ii, jj, kk+1]   + fine_field[ii+1, jj, kk+1] +
            fine_field[ii, jj+1, kk+1] + fine_field[ii+1, jj+1, kk+1]
        ) / 8
    end
end

function restrict_x_face!(coarse_field, fine_field,coarse_mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_+1
        ii = 2i - 1
        jj = 2j - 1
        kk = 2k - 1
        coarse_field[i,j,k] = (
            fine_field[ii,  jj,  kk]     + fine_field[ii, jj,   kk+1] +
            fine_field[ii,  jj+1,kk]     + fine_field[ii, jj+1, kk+1] 
        ) / 4
    end
end

function restrict_y_face!(coarse_field, fine_field,coarse_mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
    for k in kmin_:kmax_, j in jmin_:jmax_+1, i in imin_:imax_
        ii = 2i - 1
        jj = 2j - 1
        kk = 2k - 1
        coarse_field[i,j,k] = (
            fine_field[ii,  jj,   kk] + fine_field[ii+1, jj,   kk] +
            fine_field[ii,  jj,   kk+1] + fine_field[ii+1, jj, kk+1]
        ) / 4
    end
end

function restrict_z_face!(coarse_field, fine_field,coarse_mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
    for k in kmin_:kmax_+1, j in jmin_:jmax_, i in imin_:imax_
        ii = 2i - 1
        jj = 2j - 1
        kk = 2k - 1
        coarse_field[i,j,k] = (
            fine_field[ii,   jj,  kk] + fine_field[ii+1, jj,   kk] +
            fine_field[ii,   jj+1, kk] + fine_field[ii+1, jj+1, kk] 
        ) / 4
    end
end

"""
Harmonic-mean restriction of face densities (or any face coefficient that
enters the operator as 1/coeff, e.g. denx/deny/denz). Arithmetic averaging
of a discontinuous coefficient across an interface does not preserve the
coarse-grid resistance and is a known source of multigrid slowdown for
high-contrast-coefficient (e.g. high density ratio) problems. The harmonic
mean (equivalently: arithmetic-average 1/coeff, then invert) is the
flux-consistent choice.
"""
function restrict_x_face_harmonic!(coarse_field, fine_field, coarse_mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_+1
        ii = 2i - 1
        jj = 2j - 1
        kk = 2k - 1
        coarse_field[i,j,k] = 4 / (
            1/fine_field[ii,  jj,  kk]   + 1/fine_field[ii, jj,   kk+1] +
            1/fine_field[ii,  jj+1,kk]   + 1/fine_field[ii, jj+1, kk+1]
        )
    end
end

function restrict_y_face_harmonic!(coarse_field, fine_field, coarse_mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
    for k in kmin_:kmax_, j in jmin_:jmax_+1, i in imin_:imax_
        ii = 2i - 1
        jj = 2j - 1
        kk = 2k - 1
        coarse_field[i,j,k] = 4 / (
            1/fine_field[ii,  jj,   kk]   + 1/fine_field[ii+1, jj,   kk] +
            1/fine_field[ii,  jj,   kk+1] + 1/fine_field[ii+1, jj, kk+1]
        )
    end
end

function restrict_z_face_harmonic!(coarse_field, fine_field, coarse_mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
    for k in kmin_:kmax_+1, j in jmin_:jmax_, i in imin_:imax_
        ii = 2i - 1
        jj = 2j - 1
        kk = 2k - 1
        coarse_field[i,j,k] = 4 / (
            1/fine_field[ii,   jj,  kk]   + 1/fine_field[ii+1, jj,   kk] +
            1/fine_field[ii,   jj+1, kk] + 1/fine_field[ii+1, jj+1, kk]
        )
    end
end

function fill_ghost_cells!(field, mesh, par_env)
    @unpack imin_, imax_, jmin_, jmax_, kmin_, kmax_ = mesh
    @unpack imino_, imaxo_, jmino_, jmaxo_, kmino_, kmaxo_ = mesh

    # x-direction ghost cells
    for k in kmino_:kmaxo_, j in jmino_:jmaxo_
        for i in imino_:imin_-1
            field[i, j, k] = field[imin_, j, k]
        end
        for i in imax_+1:imaxo_
            field[i, j, k] = field[imax_, j, k]
        end
    end

    # y-direction ghost cells
    for k in kmino_:kmaxo_, i in imino_:imaxo_
        for j in jmino_:jmin_-1
            field[i, j, k] = field[i, jmin_, k]
        end
        for j in jmax_+1:jmaxo_
            field[i, j, k] = field[i, jmax_, k]
        end
    end

    # z-direction ghost cells
    for j in jmino_:jmaxo_, i in imino_:imaxo_
        for k in kmino_:kmin_-1
            field[i, j, k] = field[i, j, kmin_]
        end
        for k in kmax_+1:kmaxo_
            field[i, j, k] = field[i, j, kmax_]
        end
    end

    # Optional MPI halo exchange or periodic BC fill
    update_borders!(field, mesh, par_env)
end

function interface_update(band,P,coarse_sol,mesh,par_env)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mesh

    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        # near interface (dont include correction)
        if abs(band[i,j,k]) <= 1 
            nothing
        # away from interface error is reasonable
        else
            P[i,j,k] -= coarse_sol[i,j,k]
        end
    end
end

function mg_cycler(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,mg_arrays,mg_mesh,VF,verts,tets,face_BC!,param,par_env)
    @unpack pressureSolver,pressure_scheme,mg_lvl = param
    @unpack comm,irank = par_env
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[1]

    # pvd_data = mg_VTK_init_all(mg_lvl, par_env)
    pvd_data = nothing

    # set up arrays
    fields = (P = P,uf = uf,vf = vf,wf = wf,denx = denx,deny = deny,denz = denz,gradx = gradx,grady = grady,gradz = gradz,band = band)
    copy_to_mg!(mg_arrays,fields,1)

    iter = 0
    pvtk_iter = 0
    converged = false
    for i in 1:10000
        iter += 1
        pvtk_iter += 1
        if pressure_scheme == "finite-difference"
            # converged = mg_vc_lin!(1,mg_arrays,mg_mesh,dt,VF,pvd_data,param,par_env;iter)
            # converged = mg_fas_lin!(1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter;iter)
            converged = mg_fas_lin2!(1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,face_BC!,param,par_env,pvtk_iter;iter)
        elseif pressure_scheme == "semi-lagrangian"
            converged = mg_fas!(1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,face_BC!,param,par_env,pvtk_iter;iter)
        end
        
        if converged[]
            for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
                fields.P[i,j,k] = mg_arrays[1].P_h[i,j,k]
            end    
            Neumann!(fields.P,mg_mesh.mesh_lvls[1],par_env)
            update_borders!(P,mg_mesh.mesh_lvls[1],par_env)
            break
        end 
    end 
    return iter
end


# #! define recursive function for V cycle FAS method
function mg_fas!(lvl,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,face_BC!,param,par_env,pvtk_iter;iter = nothing,converged::Union{Nothing, Ref{Bool}}=nothing)
    @unpack mg_lvl = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]

    fine_lvl = mg_arrays[lvl]

    #! need to restructure to better handle VF
    if lvl ==  1
        VF_lvl = VF 
    else 
        VF_lvl = fine_lvl.tmplrg
    end

    # number of pre and post smooths
    v1 = 3
    v2 = 3

    if lvl == mg_lvl
        # relax on coarsest level ( residual now is stored tmp1)
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,50000;tol_lvl=1e-10,verts,tets)
        return
    end
    coarse_lvl = mg_arrays[lvl+1]

    if lvl == 1
        # Pre-smoothing on current level ( residual now is stored tmp1)
        converged_flag = Ref(false)
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v1;iter,verts,tets,converged_flag)
        if converged_flag[]
            return converged_flag
        end
    else
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v1;verts,tets)
    end

    # Restrict VF and compute band on coarse level
    restrict!(coarse_lvl.tmplrg,VF_lvl,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    update_borders!(coarse_lvl.tmplrg,mg_mesh.mesh_lvls[lvl+1],par_env)
    Neumann!(coarse_lvl.tmplrg,mg_mesh.mesh_lvls[lvl+1],par_env)
    computeBand!(coarse_lvl.band,coarse_lvl.tmplrg,param,mg_mesh.mesh_lvls[lvl+1],par_env)
    # fill!(coarse_lvl.band,2.0)

    # Restrict approximate solution for initial guess on coarse grid for initial guess
    restrict!(coarse_lvl.P_h,fine_lvl.P_h,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    Neumann!(coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders!(coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl+1],par_env)
    
    # Restrict densities
    # restrict_x_face!(coarse_lvl.denx,fine_lvl.denx,mg_mesh.mesh_lvls[lvl+1])
    # restrict_y_face!(coarse_lvl.deny,fine_lvl.deny,mg_mesh.mesh_lvls[lvl+1])
    # restrict_z_face!(coarse_lvl.denz,fine_lvl.denz,mg_mesh.mesh_lvls[lvl+1])
    compute_dens!(coarse_lvl.denx,coarse_lvl.deny,coarse_lvl.denz,coarse_lvl.tmplrg,param,mg_mesh.mesh_lvls[lvl+1])
    update_xface_borders!(coarse_lvl.denx,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_yface_borders!(coarse_lvl.deny,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_zface_borders!(coarse_lvl.denz,mg_mesh.mesh_lvls[lvl+1],par_env)



    # Restrict velocities
    # restrict_x_face!(coarse_lvl.uf,fine_lvl.uf,mg_mesh.mesh_lvls[lvl+1])
    # restrict_y_face!(coarse_lvl.vf,fine_lvl.vf,mg_mesh.mesh_lvls[lvl+1])
    # restrict_z_face!(coarse_lvl.wf,fine_lvl.wf,mg_mesh.mesh_lvls[lvl+1])
    face_BC!(coarse_lvl.uf,coarse_lvl.vf,coarse_lvl.wf,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_xface_borders!(coarse_lvl.uf,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_yface_borders!(coarse_lvl.vf,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_zface_borders!(coarse_lvl.wf,mg_mesh.mesh_lvls[lvl+1],par_env)

    # copmpute A(R(P^h)) (A operator applied to restricted approximate solution on finer level)
    fill!(coarse_lvl.AP_c,0.0)
    A!(coarse_lvl.AP_c,coarse_lvl.uf,coarse_lvl.vf,coarse_lvl.wf,coarse_lvl.P_h,dt,coarse_lvl.gradx,coarse_lvl.grady,coarse_lvl.gradz,coarse_lvl.band,coarse_lvl.denx,coarse_lvl.deny,coarse_lvl.denz,verts,tets,param,mg_mesh.mesh_lvls[lvl+1],par_env)

    # compute R(A(P^h)) (A operator applied to approximate solution on finer level)
    fill!(fine_lvl.AP_f,0.0)
    A!(fine_lvl.AP_f,fine_lvl.uf,fine_lvl.vf,fine_lvl.wf,fine_lvl.P_h,dt,fine_lvl.gradx,fine_lvl.grady,fine_lvl.gradz,fine_lvl.band,fine_lvl.denx,fine_lvl.deny,fine_lvl.denz,verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env)
    fine_lvl.AP_f .-= fine_lvl.tmp1
    Neumann!(fine_lvl.AP_f, mg_mesh.mesh_lvls[lvl], par_env) 
    restrict!(coarse_lvl.AP_f,fine_lvl.AP_f,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])

    # compute Tau correction (τ = R(A(P^h)) - A(R(P^h)))
    coarse_lvl.tmp1 .= coarse_lvl.AP_c .- coarse_lvl.AP_f

    # store restricted solution for error calc
    coarse_lvl.P_bar_H .= coarse_lvl.P_h

    if lvl < mg_lvl
        # recursively call mg_fas!
        mg_fas!(lvl+1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,face_BC!,param,par_env,pvtk_iter;iter,converged)
    end
    
    # calculate error ( P^2h-R(P^h) )
    coarse_lvl.P_h .-= coarse_lvl.P_bar_H

    # prolongate error into res scratch buffer
    fill!(fine_lvl.res,0.0)
    prolong!(fine_lvl.res,coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl],mg_mesh.mesh_lvls[lvl+1])
    update_borders!(fine_lvl.res,mg_mesh.mesh_lvls[lvl],par_env)

    # apply correction directly to pre-smoothed P_h (matches mg_fas_lin2! structure)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]
    @views fine_lvl.P_h[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+= fine_lvl.res[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
    fill!(fine_lvl.res,0.0)

    if lvl != 1
        # post smoothing on corrected P_h
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v2;verts,tets)
    else
        # post smoothing of finest field with corrected approximate solution
        converged_flag = Ref(false)
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v1;verts,tets,converged_flag)
        return converged_flag
    end
end


# #! define recursive function for V cycle FAS method
function mg_vc_lin!(lvl,mg_arrays,mg_mesh,dt,VF,pvd_data,param,par_env;iter=nothing,converged::Union{Nothing, Ref{Bool}}=nothing,fig = nothing)
    @unpack mg_lvl = param
    @unpack nproc,irank,comm = par_env
    
    # Initialize figure for vis
    if lvl == 1
        fig = Figure(size = (900, 850))
    end


    fine_lvl = mg_arrays[lvl]

    if lvl == 1
        VF_lvl = VF
    else 
        VF_lvl = fine_lvl.tmplrg
    end
    
    v1 = 10
    v2 = 10
    
    if lvl == mg_lvl 
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,100000;iter,tol_lvl=1e-9)

        @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]
        P_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
        RHS_slice = @view fine_lvl.RHS[div(imax_,2),jmin_:jmax_,kmin_]
        # P_slice = @view fine_lvl.P_h[imin_:imax_,jmin_:jmax_,kmin_]
        # RHS_slice = @view fine_lvl.RHS[imin_:imax_,jmin_:jmax_,kmin_]
        # Plot data for vis
        ax = Axis(fig[lvl,1]; title = "Multigrid Level: $lvl")
        scatterlines!(ax, P_slice, color = :blue, label = "post-smoothed Pressure")
        scatterlines!(ax, RHS_slice, color = :red, label = "RHS")
        # heatmap!(ax, P_slice, label = "pre-smoothed Pressure")
        # heatmap!(ax, RHS_slice, label = "RHS")
        axislegend(ax)
        return
    end
    coarse_lvl = mg_arrays[lvl+1]
    # compute RHS at finest level
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]

    if lvl == 1
        fill!(fine_lvl.RHS,0.0)
        update_borders!(fine_lvl.uf,mg_mesh.mesh_lvls[lvl],par_env)
        update_borders!(fine_lvl.vf,mg_mesh.mesh_lvls[lvl],par_env)
        update_borders!(fine_lvl.wf,mg_mesh.mesh_lvls[lvl],par_env)
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            # RHS
            fine_lvl.RHS[i,j,k] = ( 
                ( fine_lvl.uf[i+1,j,k] - fine_lvl.uf[i,j,k] )/(dx) +
                ( fine_lvl.vf[i,j+1,k] - fine_lvl.vf[i,j,k] )/(dy) +
                ( fine_lvl.wf[i,j,k+1] - fine_lvl.wf[i,j,k] )/(dz) )
        end
        update_borders!(fine_lvl.RHS,mg_mesh.mesh_lvls[lvl],par_env)
    end

    if lvl == 1
        converged_flag = Ref(false)
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v1;iter,converged_flag)
        if converged_flag[]
            return converged_flag
        end
    else
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v1;)
    end

    # Restrict residual and other neccessary quantities
    fill!(coarse_lvl.RHS,0.0)
    restrict!(coarse_lvl.tmplrg,VF_lvl,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    # Neumann!(fine_lvl.res,mg_mesh.mesh_lvls[lvl],par_env)
    
    update_borders!(fine_lvl.res,mg_mesh.mesh_lvls[lvl],par_env)
    restrict!(coarse_lvl.RHS,fine_lvl.res,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    Neumann!(coarse_lvl.RHS,mg_mesh.mesh_lvls[lvl+1],par_env)

    Neumann!(coarse_lvl.tmplrg,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders!(coarse_lvl.tmplrg,mg_mesh.mesh_lvls[lvl+1],par_env)

    # recompute band using restricted volume fraction
    computeBand!(coarse_lvl.band,coarse_lvl.tmplrg,param,mg_mesh.mesh_lvls[lvl+1],par_env)

    # Restrict densities
    restrict_x_face!(coarse_lvl.denx,fine_lvl.denx,mg_mesh.mesh_lvls[lvl+1])
    restrict_y_face!(coarse_lvl.deny,fine_lvl.deny,mg_mesh.mesh_lvls[lvl+1])
    restrict_z_face!(coarse_lvl.denz,fine_lvl.denz,mg_mesh.mesh_lvls[lvl+1])
    update_xface_borders!(coarse_lvl.denx,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_yface_borders!(coarse_lvl.deny,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_zface_borders!(coarse_lvl.denz,mg_mesh.mesh_lvls[lvl+1],par_env)

    # Slice data for vis    
    P_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
    RHS_slice = @view fine_lvl.RHS[div(imax_,2),jmin_:jmax_,kmin_]
    res_slice = @view fine_lvl.res[div(imax_,2),jmin_:jmax_,kmin_]
    # P_slice = @view fine_lvl.P_h[imin_:imax_,jmin_:jmax_,kmin_]
    # RHS_slice = @view fine_lvl.RHS[imin_:imax_,jmin_:jmax_,kmin_]
    # res_slice = @view fine_lvl.res[imin_:imax_,jmin_:jmax_,kmin_]
    # Plot data for vis
    ax = Axis(fig[lvl,1]; title = "Multigrid Level: $lvl")
    # ylims!(ax, -2.0, 2.0)
    scatterlines!(ax, P_slice, color = :blue, label = "pre-smoothed Pressure")
    scatterlines!(ax, RHS_slice, color = :red, label = "RHS")
    scatterlines!(ax, res_slice, color = :green, label = "residual")
    # heatmap!(ax, P_slice,  label = "pre-smoothed Pressure")
    # heatmap!(ax, RHS_slice,  label = "RHS")
    # heatmap!(ax, res_slice, label = "residual")
    axislegend(ax)

    # recursively call mg_vc_lin! (W-cycle: call twice for better coarse correction)
    if lvl < mg_lvl
        fill!(coarse_lvl.P_h,0.0)
        fill!(coarse_lvl.tmp1,0.0)
        mg_vc_lin!(lvl+1,mg_arrays,mg_mesh,dt,VF,pvd_data,param,par_env;iter,converged,fig=fig)
        # mg_vc_lin!(lvl+1,mg_arrays,mg_mesh,dt,VF,pvd_data,param,par_env;iter,converged)
    end

    # prolongate error and move up a level
    fill!(fine_lvl.tmp2,0.0)
    prolong!(fine_lvl.tmp2,coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl],mg_mesh.mesh_lvls[lvl+1])
    update_borders!(fine_lvl.tmp2,mg_mesh.mesh_lvls[lvl],par_env)

    ax = Axis(fig[lvl,2]; title = "Multigrid Level: $lvl")
    # ylims!(ax, -2.0, 2.0)
    P_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
    # P_slice = @view fine_lvl.P_h[imin_:imax_,jmin_:jmax_,kmin_]
    scatterlines!(ax, P_slice, color = :blue, label = "uncorrected Pressure")
    # heatmap!(ax, P_slice, label = "uncorrected Pressure")
    r_before = res_comp!(fine_lvl.tmp3,fine_lvl.RHS,fine_lvl.P_h,fine_lvl.denx,fine_lvl.deny,fine_lvl.denz,dt,param,mg_mesh.mesh_lvls[lvl],par_env)
    # correct approximate solution with error
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        fine_lvl.P_h[i,j,k] += fine_lvl.tmp2[i,j,k]
    end    
    update_borders!(fine_lvl.P_h,mg_mesh.mesh_lvls[lvl],par_env)

    err_slice = @view fine_lvl.tmp2[div(imax_,2),jmin_:jmax_,kmin_]
    # err_slice = @view fine_lvl.tmp2[imin_:imax_,jmin_:jmax_,kmin_]
    # Plot data for vis
    scatterlines!(ax, err_slice, color = :green, label = "Error")
    # heatmap!(ax, err_slice, color = :green, label = "Error")
    r_after = res_comp!(fine_lvl.tmp2,fine_lvl.RHS,fine_lvl.P_h,fine_lvl.denx,fine_lvl.deny,fine_lvl.denz,dt,param,mg_mesh.mesh_lvls[lvl],par_env)

    if lvl !== 1
        # post-smoothening on corrected solution
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v2)
        P_s_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
        # P_s_slice = @view fine_lvl.P_h[imin_:imax_,jmin_:jmax_,kmin_]
        scatterlines!(ax, P_slice, color = :red, label = "error correction")
        # heatmap!(ax, P_slice, label = "post-smoothed Pressure")
        axislegend(ax)
    else
        # final solve on corrected finest grid
        converged_flag = Ref(false) 
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v2;iter,converged_flag)
        P_s_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
        # P_s_slice = @view fine_lvl.P_h[imin_:imax_,jmin_:jmax_,kmin_]
        scatterlines!(ax, P_s_slice, color = :red, label = "post-smoothed Pressure")
        # heatmap!(ax, P_s_slice, label = "post-smoothed Pressure")
        Label(fig[0, :], "Iteration $iter", fontsize=20)
        axislegend(ax)
        # display(fig)
        # if iter == 1
        #     error("stop")
        # end
        return converged_flag
    end
end

function mg_fas_lin!(lvl,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter;iter = nothing,τ = nothing)
    @unpack mg_lvl = param
    @unpack comm = par_env

    fine_lvl = mg_arrays[lvl]
    
    #! need to restructure to better handle VF
    if lvl ==  1
        VF_lvl = VF
    else 
        VF_lvl = fine_lvl.tmplrg
    end
    
    v1 = 3
    v2 = 3
    # compute RHS at finest level
    #! test difference between recomputing RHS and restricting RHS 
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]
    if lvl == 1
        fill!(fine_lvl.RHS,0.0)
        update_borders!(fine_lvl.uf,mg_mesh.mesh_lvls[lvl],par_env)
        update_borders!(fine_lvl.vf,mg_mesh.mesh_lvls[lvl],par_env)
        update_borders!(fine_lvl.wf,mg_mesh.mesh_lvls[lvl],par_env)
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            # RHS
            fine_lvl.RHS[i,j,k] = ( 
                ( fine_lvl.uf[i+1,j,k] - fine_lvl.uf[i,j,k] )/(dx) +
                ( fine_lvl.vf[i,j+1,k] - fine_lvl.vf[i,j,k] )/(dy) +
                ( fine_lvl.wf[i,j,k+1] - fine_lvl.wf[i,j,k] )/(dz) )
        end
        update_borders!(fine_lvl.RHS,mg_mesh.mesh_lvls[lvl],par_env)
    end
    
    if lvl == mg_lvl 
        # relax on coarsest level ( residual now is stored tmp1)
        poisson_solve!(fine_lvl.P_h,fine_lvl.tmp1,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,50000;iter,tol_lvl=1e-10)
        return
    end

    coarse_lvl = mg_arrays[lvl+1]

    if lvl == 1
        # Pre-smoothing on current level ( residual now is stored tmp1)
        converged_flag = Ref(false) 
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v1;iter,converged_flag)
        if converged_flag[]
            return converged_flag
        end
    else
        poisson_solve!(fine_lvl.P_h,fine_lvl.tmp1,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v2)
    end
   
    # Restrict VF and compute band on coarse level
    restrict!(coarse_lvl.tmplrg,VF_lvl,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    update_borders!(coarse_lvl.tmplrg,mg_mesh.mesh_lvls[lvl+1],par_env)
    Neumann!(coarse_lvl.tmplrg,mg_mesh.mesh_lvls[lvl+1],par_env)
    computeBand!(coarse_lvl.band,coarse_lvl.tmplrg,param,mg_mesh.mesh_lvls[lvl+1],par_env)
    
    # Restrict approximate solution for initial guess on coarse grid for initial guess    
    restrict!(coarse_lvl.P_h,fine_lvl.P_h,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    Neumann!(coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders!(coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl+1],par_env)

    # Restrict densities
    restrict_x_face!(coarse_lvl.denx,fine_lvl.denx,mg_mesh.mesh_lvls[lvl+1])
    restrict_y_face!(coarse_lvl.deny,fine_lvl.deny,mg_mesh.mesh_lvls[lvl+1])
    restrict_z_face!(coarse_lvl.denz,fine_lvl.denz,mg_mesh.mesh_lvls[lvl+1])
    update_xface_borders!(coarse_lvl.denx,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_yface_borders!(coarse_lvl.deny,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_zface_borders!(coarse_lvl.denz,mg_mesh.mesh_lvls[lvl+1],par_env)
    
    # copmpute A(P^2h) (A operator applied to restricted approximate solution on finer level)
    fill!(coarse_lvl.AP_c,0.0)
    lap!(coarse_lvl.AP_c,coarse_lvl.P_h,coarse_lvl.denx,coarse_lvl.deny,coarse_lvl.denz,dt,param,mg_mesh.mesh_lvls[lvl+1],par_env) 
        
    # restrict RHS (for use in post-smoothening) and copy to tmp1 to compute coarse grid RHS (for use in pre-smoothening)
    fill!(coarse_lvl.RHS,0.0)
    fill!(coarse_lvl.res,0.0)
    restrict!(coarse_lvl.RHS,fine_lvl.RHS,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    Neumann!(fine_lvl.res,mg_mesh.mesh_lvls[lvl],par_env)
    update_borders!(fine_lvl.res,mg_mesh.mesh_lvls[lvl],par_env)
    restrict!(coarse_lvl.res,fine_lvl.res,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])

    coarse_lvl.tmp1 .= coarse_lvl.res .+ coarse_lvl.AP_c

    # store restricted pressure for error calc
    coarse_lvl.P_bar_H .= coarse_lvl.P_h

    # # store approximate solution for correction and post smoothening
    # fine_lvl.P_H .= fine_lvl.P_h

    if lvl < mg_lvl
        # recursively call mg_fas!
        mg_fas_lin!(lvl+1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter;iter)
        # mg_fas_lin!(lvl+1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter;iter)
    end
    # println("post-smoothening on level $lvl starting")
    # begin prolongation routine starting at the coarsest level (occurs after relaxation at coarsest level)
    # calculate error ( P^2h-R(P^h) )
    coarse_lvl.P_h .-= coarse_lvl.P_bar_H

    MPI.Barrier(comm)
    # prolongate error (corrected approximate solution)
    fill!(fine_lvl.res,0.0)
    prolong!(fine_lvl.res,coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl],mg_mesh.mesh_lvls[lvl+1])   
    update_borders!(fine_lvl.res,mg_mesh.mesh_lvls[lvl],par_env)

    @views fine_lvl.P_h[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+= fine_lvl.res[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
    
    fill!(fine_lvl.res,0.0)
    if lvl != 1
        # post smoothing of finest field wth corrected approximate solution
        poisson_solve!(fine_lvl.P_h,fine_lvl.tmp1,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v2;)
    else
        converged_flag = Ref(false) 
        # post smoothing of finest field wth corrected approximate solution
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v2;iter,converged_flag)
        return converged_flag
    end
end


function mg_fas_lin2!(lvl,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,face_BC!,param,par_env,pvtk_iter;iter = nothing,τ = nothing,fig = nothing)
    @unpack mg_lvl = param
    @unpack comm = par_env

    # Initialize figure for vis
    if lvl == 1
        fig = Figure(size = (900, 850))
    end

    # println("level $lvl starting pre-smoothening")
    fine_lvl = mg_arrays[lvl]
    
    #! need to restructure to better handle VF
    if lvl ==  1
        VF_lvl = VF
    else 
        VF_lvl = fine_lvl.tmplrg
    end
    
    v1 = 10
    v2 = 10
    # compute RHS at finest level
    #! test difference between recomputing RHS and restricting RHS 
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]
    if lvl == 1
        fill!(fine_lvl.RHS,0.0)
        update_borders!(fine_lvl.uf,mg_mesh.mesh_lvls[lvl],par_env)
        update_borders!(fine_lvl.vf,mg_mesh.mesh_lvls[lvl],par_env)
        update_borders!(fine_lvl.wf,mg_mesh.mesh_lvls[lvl],par_env)
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            # RHS
            fine_lvl.RHS[i,j,k] = ( 
                ( fine_lvl.uf[i+1,j,k] - fine_lvl.uf[i,j,k] )/(dx) +
                ( fine_lvl.vf[i,j+1,k] - fine_lvl.vf[i,j,k] )/(dy) +
                ( fine_lvl.wf[i,j,k+1] - fine_lvl.wf[i,j,k] )/(dz) )
        end
        update_borders!(fine_lvl.RHS,mg_mesh.mesh_lvls[lvl],par_env)
    end
    
    if lvl == mg_lvl 
        # relax on coarsest level ( residual now is stored tmp1)
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,50000;iter,tol_lvl=1e-10)
        @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]
        P_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
        RHS_slice = @view fine_lvl.RHS[div(imax_,2),jmin_:jmax_,kmin_]        
        ax = Axis(fig[lvl,1]; title = "Multigrid Level: $lvl")
        scatterlines!(ax, P_slice, color = :blue, label = "pre-smoothed Pressure")
        scatterlines!(ax, RHS_slice, color = :red, label = "RHS")
        axislegend(ax)
        return
    end

    coarse_lvl = mg_arrays[lvl+1]

    if lvl == 1
        # Pre-smoothing on current level ( residual now is stored tmp1)
        converged_flag = Ref(false) 
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v1;iter,converged_flag)
        if converged_flag[]
            return converged_flag
        end
    else
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v2;)
    end
   
    # Restrict VF and compute band on coarse level
    restrict!(coarse_lvl.tmplrg,VF_lvl,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    update_borders!(coarse_lvl.tmplrg,mg_mesh.mesh_lvls[lvl+1],par_env)
    Neumann!(coarse_lvl.tmplrg,mg_mesh.mesh_lvls[lvl+1],par_env)
    computeBand!(coarse_lvl.band,coarse_lvl.tmplrg,param,mg_mesh.mesh_lvls[lvl+1],par_env)
    
    # Restrict approximate solution for initial guess on coarse grid for initial guess 
    restrict!(coarse_lvl.P_h,fine_lvl.P_h,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    Neumann!(coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl+1],par_env)
    # update_borders!(coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl+1],par_env)

    # Restrict densities
    restrict_x_face!(coarse_lvl.denx,fine_lvl.denx,mg_mesh.mesh_lvls[lvl+1])
    restrict_y_face!(coarse_lvl.deny,fine_lvl.deny,mg_mesh.mesh_lvls[lvl+1])
    restrict_z_face!(coarse_lvl.denz,fine_lvl.denz,mg_mesh.mesh_lvls[lvl+1])
    update_xface_borders!(coarse_lvl.denx,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_yface_borders!(coarse_lvl.deny,mg_mesh.mesh_lvls[lvl+1],par_env)
    update_zface_borders!(coarse_lvl.denz,mg_mesh.mesh_lvls[lvl+1],par_env)

    # copmpute A(R(P^h)) (A operator applied to restricted approximate solution on finer level)
    fill!(coarse_lvl.AP_c,0.0)
    lap!(coarse_lvl.AP_c,coarse_lvl.P_h,coarse_lvl.denx,coarse_lvl.deny,coarse_lvl.denz,dt,param,mg_mesh.mesh_lvls[lvl+1],par_env) 
    

    # compute R(A(P^h)) (A operator applied to approximate solution on finer level)
    fill!(fine_lvl.AP_f,0.0)
    lap!(fine_lvl.AP_f,fine_lvl.P_h,fine_lvl.denx,fine_lvl.deny,fine_lvl.denz,dt,param,mg_mesh.mesh_lvls[lvl],par_env)
    Neumann!(fine_lvl.AP_f, mg_mesh.mesh_lvls[lvl], par_env) 
    fill!(coarse_lvl.AP_f, 0.0)
    restrict!(coarse_lvl.AP_f,fine_lvl.AP_f,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    

    # restrict RHS (for use in post-smoothening) and copy to tmp1 to compute coarse grid RHS (for use in pre-smoothening)
    fill!(coarse_lvl.RHS,0.0)
    fine_lvl.RHS .+= fine_lvl.tmp1
    Neumann!(fine_lvl.RHS, mg_mesh.mesh_lvls[lvl], par_env) 
    restrict!(coarse_lvl.RHS,fine_lvl.RHS,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    fine_lvl.RHS .-= fine_lvl.tmp1
    # Neumann!(coarse_lvl.RHS,mg_mesh.mesh_lvls[lvl+1],par_env)
    # update_borders!(coarse_lvl.RHS,mg_mesh.mesh_lvls[lvl+1],par_env)
    
    # Build Tau correction (including previous level Tau correction if it exists)
    coarse_lvl.tmp1 .= coarse_lvl.AP_c .- coarse_lvl.AP_f 
    
    # store restricted pressure for error calc
    coarse_lvl.P_bar_H .= coarse_lvl.P_h

    # # store approximate solution for correction and post smoothening
    # fine_lvl.P_H .= fine_lvl.P_h

    P_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
    RHS_slice = @view fine_lvl.RHS[div(imax_,2),jmin_:jmax_,kmin_]
    res_slice = @view fine_lvl.tmp1[div(imax_,2),jmin_:jmax_,kmin_]
    # Plot data for vis
    ax = Axis(fig[lvl,1]; title = "Multigrid Level: $lvl")
    scatterlines!(ax, P_slice, color = :blue, label = "pre-smoothed Pressure")
    scatterlines!(ax, RHS_slice, color = :red, label = "RHS")
    scatterlines!(ax, res_slice, color = :green, label = "residual")
    axislegend(ax)

    if lvl < mg_lvl
        # recursively call mg_fas!
        mg_fas_lin2!(lvl+1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,face_BC!,param,par_env,pvtk_iter;iter,fig=fig)

        # mg_fas_lin2!(lvl+1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter;iter,fig=fig)
    end
    # println("post-smoothening on level $lvl starting")
    # begin prolongation routine starting at the coarsest level (occurs after relaxation at coarsest level)
    # calculate error ( P^2h-R(P^h) )
    # println("coarse_lvl.P_bar_H max: ", maximum(abs.(coarse_lvl.P_bar_H)))
    # println("coarse_lvl.P_h max: ", maximum(abs.(coarse_lvl.P_h)))
    coarse_lvl.P_h .-= coarse_lvl.P_bar_H
    # println(maximum(abs.(coarse_lvl.P_h)))
    # println("correction: ", maximum(abs.(coarse_lvl.P_h)))
    MPI.Barrier(comm)
    # prolongate error (corrected approximate solution)
    fill!(fine_lvl.res,0.0)
    prolong!(fine_lvl.res,coarse_lvl.P_h,mg_mesh.mesh_lvls[lvl],mg_mesh.mesh_lvls[lvl+1])   
    update_borders!(fine_lvl.res,mg_mesh.mesh_lvls[lvl],par_env)

    ax = Axis(fig[lvl,2]; title = "Multigrid Level: $lvl")
    # ylims!(ax, -2.0, 2.0)
    P_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
    scatterlines!(ax, P_slice, color = :blue, label = "uncorrected Pressure")

    @views fine_lvl.P_h[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+= fine_lvl.res[imin_:imax_,jmin_:jmax_,kmin_:kmax_]
    
    err_slice = @view fine_lvl.res[div(imax_,2),jmin_:jmax_,kmin_]
    # Plot data for vis
    scatterlines!(ax, err_slice, color = :green, label = "Error")

    # fill!(fine_lvl.res,0.0)
    # fill!(fine_lvl.tmp1,0.0)
    # fill!(coarse_lvl.tmp1,0.0)
    if lvl != 1
        # post smoothing of finest field wth corrected approximate solution
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v2;)
        P_s_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
        scatterlines!(ax, P_s_slice, color = :red, label = "post-smoothed Pressure")
        axislegend(ax)
    else
        converged_flag = Ref(false) 
        # post smoothing of finest field wth corrected approximate solution
        poisson_solve!(fine_lvl.P_h,fine_lvl.RHS,fine_lvl.res,mg_arrays,lvl,mg_mesh,dt,param,par_env,v2;iter,converged_flag)
        P_s_slice = @view fine_lvl.P_h[div(imax_,2),jmin_:jmax_,kmin_]
        scatterlines!(ax, P_s_slice, color = :red, label = "post-smoothed Pressure")
        axislegend(ax)
        Label(fig[0, :], "Iteration $iter", fontsize=20)
        # display(fig)
        # if iter == 4
        #     error("stop")
        # end
        return converged_flag
    end
end
