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

function prolong_transpose!(fine_field, coarse_field, fine_mesh, coarse_mesh)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        ii = 2i - 1
        jj = 2j - 1
        kk = 2k - 1
        cval = coarse_field[i,j,k] / 8

        fine_field[ii,   jj,   kk]   += cval
        fine_field[ii+1, jj,   kk]   += cval
        fine_field[ii,   jj+1, kk]   += cval
        fine_field[ii+1, jj+1, kk]   += cval
        fine_field[ii,   jj,   kk+1] += cval
        fine_field[ii+1, jj,   kk+1] += cval
        fine_field[ii,   jj+1, kk+1] += cval
        fine_field[ii+1, jj+1, kk+1] += cval
    end
end

function prolong_x_face!(fine_field, coarse_field,fine_mesh,coarse_mesh)
    @unpack xm, ym, zm, imino_, imaxo_, jmino_, jmaxo_, kmino_, kmaxo_ = coarse_mesh
    @unpack x, y, z, imin_, imax_, jmin_, jmax_, kmin_, kmax_ = fine_mesh

    for k_f = kmin_:kmax_, j_f = jmin_:jmax_, i_f = imin_:imax_

        # Physical location of fine cell center
        pt = (x[i_f], y[j_f], z[k_f])

        # Find index in coarse grid just below pt
        i = searchsortedlast(coarse_mesh.x, pt[1])
        j = searchsortedlast(ym, pt[2])
        k = searchsortedlast(zm, pt[3])

        # Clamp to prevent OOB
        i = clamp(i, imino_, imaxo_-1)
        j = clamp(j, jmino_, jmaxo_-1)
        k = clamp(k, kmino_, kmaxo_-1)

        # Trilinear interpolation weights
        wx1 = (pt[1] - coarse_mesh.x[i]) / (coarse_mesh.x[i+1] - coarse_mesh.x[i]); wx2 = 1.0 - wx1
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

function prolong_y_face!(fine_field, coarse_field,fine_mesh,coarse_mesh)
    @unpack xm, ym, zm, imino_, imaxo_, jmino_, jmaxo_, kmino_, kmaxo_ = coarse_mesh
    @unpack x, y, z, imin_, imax_, jmin_, jmax_, kmin_, kmax_ = fine_mesh

    for k_f = kmin_:kmax_, j_f = jmin_:jmax_, i_f = imin_:imax_

        # Physical location of fine cell center
        pt = (x[i_f], y[j_f], z[k_f])

        # Find index in coarse grid just below pt
        i = searchsortedlast(xm, pt[1])
        j = searchsortedlast(coarse_mesh.y, pt[2])
        k = searchsortedlast(zm, pt[3])

        # Clamp to prevent OOB
        i = clamp(i, imino_, imaxo_-1)
        j = clamp(j, jmino_, jmaxo_-1)
        k = clamp(k, kmino_, kmaxo_-1)

        # Trilinear interpolation weights
        wx1 = (pt[1] - xm[i]) / (xm[i+1] - xm[i]); wx2 = 1.0 - wx1
        wy1 = (pt[2] - coarse_mesh.y[j]) / (coarse_mesh.y[j+1] - coarse_mesh.y[j]); wy2 = 1.0 - wy1
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

function prolong_z_face!(fine_field, coarse_field,fine_mesh,coarse_mesh)
    @unpack xm, ym, zm, imino_, imaxo_, jmino_, jmaxo_, kmino_, kmaxo_ = coarse_mesh
    @unpack x, y, z, imin_, imax_, jmin_, jmax_, kmin_, kmax_ = fine_mesh

    for k_f = kmin_:kmax_, j_f = jmin_:jmax_, i_f = imin_:imax_

        # Physical location of fine cell center
        pt = (x[i_f], y[j_f], z[k_f])

        # Find index in coarse grid just below pt
        i = searchsortedlast(xm, pt[1])
        j = searchsortedlast(ym, pt[2])
        k = searchsortedlast(coarse_mesh.z, pt[3])

        # Clamp to prevent OOB
        i = clamp(i, imino_, imaxo_-1)
        j = clamp(j, jmino_, jmaxo_-1)
        k = clamp(k, kmino_, kmaxo_-1)

        # Trilinear interpolation weights
        wx1 = (pt[1] - xm[i]) / (xm[i+1] - xm[i]); wx2 = 1.0 - wx1
        wy1 = (pt[2] - ym[j]) / (ym[j+1] - ym[j]); wy2 = 1.0 - wy1
        wz1 = (pt[3] - coarse_mesh.z[k]) / (coarse_mesh.z[k+1] - coarse_mesh.z[k]); wz2 = 1.0 - wz1

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

function inj_restrict!(coarse_field, fine_field, coarse_mesh, fine_mesh)
    @unpack imin_, imax_, jmin_, jmax_, kmin_, kmax_ = coarse_mesh
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        ii = 2i - 1
        jj = 2j - 1
        kk = 2k - 1

        # Simple injection: just take one value from the fine grid
        coarse_field[i, j, k] = fine_field[ii, jj, kk]
    end
end

function full_restrict!(coarse_field, fine_field, coarse_mesh, fine_mesh; ghost::Bool=false)
    @unpack imin_, imax_, jmin_, jmax_, kmin_, kmax_ = coarse_mesh
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        ii = 2i
        jj = 2j
        kk = 2k

        sum = 0.0

        for dk in -1:1, dj in -1:1, di in -1:1
            weight = 1.0
            num_nonzero = abs(di) + abs(dj) + abs(dk)

            if num_nonzero == 0
                weight = 1/8
            elseif num_nonzero == 1
                weight = 1/16
            elseif num_nonzero == 2
                weight = 1/32
            else  # num_nonzero == 3
                weight = 1/64
            end

            sum += weight * fine_field[ii + di, jj + dj, kk + dk]
        end

        coarse_field[i, j, k] = sum
    end
end


function restrict_x_face!(coarse_field, fine_field,coarse_mesh;ghost::Bool=false)
    if ghost 
        @unpack jmin,jmax,imin,imax,kmax,kmin,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = coarse_mesh
        for k in kmino_:kmaxo_, j in jmino_:jmaxo_, i in imino_:imaxo_

            if i < imin || j < jmin || k < kmin
                # simple injection from fine grid ghost cells
                coarse_field[i,j,k] = fine_field[i, j, k]
            elseif i > imax || j > jmax ||  k > kmax
                ii = 2*i -(i-imax_)
                jj = 2*j -(j-jmax_)
                kk = 2*k -(k-kmax_)
                # simple injection from fine grid ghost cells
                coarse_field[i,j,k] = fine_field[ii, jj, kk]
            else
                ii = 2i - 1
                jj = 2j - 1
                kk = 2k - 1
                coarse_field[i,j,k] = (
                    fine_field[ii,  jj,  kk]     + fine_field[ii,  jj+1,  kk] +
                    fine_field[ii,  jj,  kk+1]   + fine_field[ii,  jj+1,  kk+1] 
                ) / 4
            end
        end
    else
        @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
        for k in kmin_-1:kmax_+1, j in jmin_-1:jmax_+1, i in imin_-1:imax_+2
            ii = 2i - 1
            jj = 2j - 1
            kk = 2k - 1
            coarse_field[i,j,k] = (
                fine_field[ii,  jj,  kk]     + fine_field[ii,  jj+1,  kk] +
                fine_field[ii,  jj,  kk+1]   + fine_field[ii,  jj+1,  kk+1] 
            ) / 4
        end
    end
end

function restrict_y_face!(coarse_field, fine_field,coarse_mesh;ghost::Bool=false)
    if ghost
        @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = coarse_mesh
        for k in kmino_:kmaxo_, j in jmino_:jmaxo_, i in imino_:imaxo_
            if i < imin || j < jmin || k < kmin 
                # simple injection from fine grid ghost cells
                coarse_field[i,j,k] = fine_field[i, j, k]
            elseif i > imax || j > jmax  ||  k > kmax 
                ii = 2*i -(i-imax_)
                jj = 2*j -(j-jmax_)
                kk = 2*k -(k-kmax_)
                # simple injection from fine grid ghost cells
                coarse_field[i,j,k] = fine_field[ii, jj, kk]
            else
                ii = 2i - 1
                jj = 2j - 1
                kk = 2k - 1
                coarse_field[i,j,k] = (
                    fine_field[ii,  jj,    kk]     + fine_field[ii+1, jj,    kk] +
                    fine_field[ii,  jj,    kk+1]   + fine_field[ii+1, jj,  kk+1] 
                ) / 4
            end
        end
    else
        @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
        for k in kmin_-1:kmax_+1, j in jmin_-1:jmax_+2, i in imin_-1:imax_+1
            ii = 2i - 1
            jj = 2j - 1
            kk = 2k - 1
            coarse_field[i,j,k] = (
                fine_field[ii,  jj,    kk]     + fine_field[ii+1, jj,    kk] +
                fine_field[ii,  jj,    kk+1]   + fine_field[ii+1, jj,  kk+1] 
            ) / 4
        end
    end
end

function restrict_z_face!(coarse_field, fine_field,coarse_mesh;ghost::Bool=false)
    if ghost
        @unpack imin,imax,jmin,jmax,kmin,kmax,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = coarse_mesh
        
        for k in kmino_:kmaxo_, j in jmino_:jmaxo_, i in imino_:imaxo_
            if i < imin || j < jmin || k < kmin 
                # simple injection from fine grid ghost cells
                coarse_field[i,j,k] = fine_field[i, j, k]
            elseif i > imax || j > jmax ||  k > kmax
                ii = 2*i -(i-imax_)
                jj = 2*j -(j-jmax_)
                kk = 2*k -(k-kmax_)
                # simple injection from fine grid ghost cells
                coarse_field[i,j,k] = fine_field[ii, jj, kk]
            else
                ii = 2i - 1
                jj = 2j - 1
                kk = 2k - 1
                coarse_field[i,j,k] = (
                    fine_field[ii,  jj,  kk]    + fine_field[ii+1, jj,  kk] +
                    fine_field[ii,  jj+1, kk]   + fine_field[ii+1, jj+1, kk]  
                ) / 4
            end
        end
    else
        @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = coarse_mesh
        for k in kmin_-1:kmax_+2, j in jmin_-1:jmax_+1, i in imin_-1:imax_+1
            ii = 2i - 1
            jj = 2j - 1
            kk = 2k - 1
            coarse_field[i,j,k] = (
                fine_field[ii,  jj,  kk]    + fine_field[ii+1, jj,  kk] +
                fine_field[ii,  jj+1, kk]   + fine_field[ii+1, jj+1, kk] 
            ) / 4
        end
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

function mg_cycler(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,mg_arrays,mg_mesh,VF,verts,tets,param,par_env,BC!)
    @unpack pressureSolver,pressure_scheme,mg_lvl = param
    @unpack comm,irank = par_env
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[1]

    pvd_data = mg_VTK_init_all(mg_lvl, par_env)
    # pvd_data = nothing
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
            # converged = mg_vc_lin!(1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env;iter,tmp5)
            mg_fas_lin!(1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter,BC!;iter)
            # println("iter $i complete")
            # if iter == 5
            #     error("stop")
            # end
        elseif pressure_scheme == "semi-lagrangian"
            mg_fas!(1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter;iter)
            println("iter $i complete")
            # if iter == 10
            #     error("stop")
            # end
        end
        #! need to check convergence
        # all_conv = MPI.Gather(converged,0,comm)
        # if irank == 0 && all(all_conv[])
        # if converged[]
        if i == 20
            for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
                fields.P[i,j,k] = mg_arrays.P_h[1][i,j,k]
            end    
            Neumann!(fields.P,mg_mesh.mesh_lvls[1],par_env)
            update_borders!(P,mg_mesh.mesh_lvls[1],par_env)
            break
        end
 
    end
    
    return iter
end

# #! define recursive function for V cycle FAS method
function mg_fas!(lvl,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter;iter = nothing,τ::Union{Nothing, Any} = nothing)
    @unpack mg_lvl = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]

    #! need to restructure to better handle VF
    if lvl ==  1
        VF_lvl = VF 
    else 
        VF_lvl = mg_arrays.tmplrg[lvl]
    end

    v1 = 5
    v2 = 5

    println("on multrigrid level $lvl")

    # println(imax_)
    if lvl == mg_lvl
        arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.tmp1[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],wf = mg_arrays.wf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        m_iter = 0
        mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.uf,arr.vf,arr.wf,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
        # relax on coarsest level ( residual now is stored tmp1)
        # Secant_jacobian_hypre!(mg_arrays.P_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;τ)
        # nonlin_gs(mg_arrays.P_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = 10000,τ)
        res_iteration(mg_arrays.P_h[lvl],mg_arrays.uf[lvl],mg_arrays.vf[lvl],mg_arrays.wf[lvl],mg_arrays.gradx[lvl],mg_arrays.grady[lvl],mg_arrays.gradz[lvl],mg_arrays.band[lvl],dt,mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],mg_arrays.AP_f[lvl],mg_arrays.AP_c[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = 10000,τ=mg_arrays.tmp1[lvl])
        # mg_mesh_lvl = (p =mg_arrays.P_mg[lvl],tmp1 = mg_arrays.tmp1_mg[lvl],tmplrg = mg_arrays.tmplrg_mg[lvl],vf = mg_arrays.vf_mg[lvl],denx = mg_arrays.denx_mg[lvl])
        # mg_VTK!(pvtk_iter,pvd_data,mg_mesh_lvl.p,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_mesh_lvl.tmp1,mg_mesh_lvl.tmplrg,mg_mesh_lvl.vf,mg_mesh_lvl.denx,tau)
        # error("stop")
        arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.tmp1[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],wf = mg_arrays.wf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        m_iter = 1
        mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.uf,arr.vf,arr.wf,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
        return
    end

    if lvl == 1
        # Pre-smoothing on current level ( residual now is stored tmp1)
        res_iteration(mg_arrays.P_h[lvl],mg_arrays.uf[lvl],mg_arrays.vf[lvl],mg_arrays.wf[lvl],mg_arrays.gradx[lvl],mg_arrays.grady[lvl],mg_arrays.gradz[lvl],mg_arrays.band[lvl],dt,mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],mg_arrays.AP_f[lvl],mg_arrays.AP_c[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = v1)
        # nonlin_gs(mg_arrays.P_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = 8,iter)
        # Secant_jacobian_hypre!(mg_arrays.P_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = 2,τ)
        
        # mg_mesh_lvl = (p =mg_arrays.P_mg[1],tmp1 = mg_arrays.tmp1_mg[1],tmplrg = mg_arrays.tmplrg_mg[1],vf = mg_arrays.vf_mg[1],denx = mg_arrays.denx_mg[1])
        # mg_VTK!(pvtk_iter,pvd_data,mg_mesh_lvl.p,1,param,mg_mesh.mesh_lvls[1],par_env;mg_mesh_lvl.tmp1,VF,mg_mesh_lvl.vf,mg_mesh_lvl.denx)
    
    else
        res_iteration(mg_arrays.P_h[lvl],mg_arrays.uf[lvl],mg_arrays.vf[lvl],mg_arrays.wf[lvl],mg_arrays.gradx[lvl],mg_arrays.grady[lvl],mg_arrays.gradz[lvl],mg_arrays.band[lvl],dt,mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],mg_arrays.AP_f[lvl],mg_arrays.AP_c[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = v1,τ=mg_arrays.tmp1[lvl])
        # nonlin_gs(mg_arrays.P_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = 8,τ)
        # Secant_jacobian_hypre!(mg_arrays.P_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = 3,τ)
    end

    # mg_VTK!(m_iter,pvd_data,mg_arrays.P,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_arrays.tmp5,mg_arrays.uf,mg_arrays.vf)
    # iter +=1
    # mg_VTK!(iter,pvd_data,mg_arrays.P,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_arrays.band,mg_arrays.uf,mg_arrays.vf,VF)
    
    # Restrict VF and compute band on coarse level
    restrict!(mg_arrays.tmplrg[lvl+1],VF_lvl,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    update_borders!(mg_arrays.tmplrg[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    Neumann!(mg_arrays.tmplrg[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    computeBand!(mg_arrays.band[lvl+1],mg_arrays.tmplrg[lvl+1],param,mg_mesh.mesh_lvls[lvl+1],par_env)

    # Restrict approximate solution for initial guess on coarse grid for initial guess
    restrict!(mg_arrays.P_h[lvl+1],mg_arrays.P_h[lvl],mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    Neumann!(mg_arrays.P_h[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders!(mg_arrays.P_h[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)

    # Restrict residual for nonlinear defecit correction
    restrict!(mg_arrays.AP_f[lvl+1],mg_arrays.AP_f[lvl],mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])


    # Compute densities on coarse grid from restricted VF
    compute_dens!(mg_arrays.denx[lvl+1],mg_arrays.deny[lvl+1],mg_arrays.denz[lvl+1],mg_arrays.tmplrg[lvl+1],param,mg_mesh.mesh_lvls[lvl+1])
    update_borders_x!(mg_arrays.denx[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders_y!(mg_arrays.deny[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders_z!(mg_arrays.denz[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    
    # Restrict velocities
    restrict_x_face!(mg_arrays.uf[lvl+1],mg_arrays.uf[lvl],mg_mesh.mesh_lvls[lvl+1])
    restrict_y_face!(mg_arrays.vf[lvl+1],mg_arrays.vf[lvl],mg_mesh.mesh_lvls[lvl+1])
    restrict_z_face!(mg_arrays.wf[lvl+1],mg_arrays.wf[lvl],mg_mesh.mesh_lvls[lvl+1])
    update_borders_x!(mg_arrays.uf[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders_y!(mg_arrays.vf[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders_z!(mg_arrays.wf[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)

    # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.tmp1[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],wf = mg_arrays.wf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
    # m_iter = 0
    # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.uf,arr.vf,arr.wf,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
    
    # mg_VTK!(iter,pvd_data,c_fields.cc.band,lvl+1,param,mg_mesh.mesh_lvls[lvl+1],par_env;c_fields.cc.band,c_fields.fc.uf,c_fields.fc.vf,c_fields.cc.tmplrg)

    # grab restricted residual (R(A^h(P^h))), compute A^2h(R(P^h) and compute tau
    fill!(mg_arrays.AP_c[lvl+1],0.0)
    A!(mg_arrays.AP_c[lvl+1],mg_arrays.uf[lvl+1],mg_arrays.vf[lvl+1],mg_arrays.wf[lvl+1],mg_arrays.P_h[lvl+1],dt,mg_arrays.gradx[lvl+1],mg_arrays.grady[lvl+1],mg_arrays.gradz[lvl+1],mg_arrays.band[lvl+1],mg_arrays.denx[lvl+1],mg_arrays.deny[lvl+1],mg_arrays.denz[lvl+1],verts,tets,param,mg_mesh.mesh_lvls[lvl+1],par_env)
    mg_arrays.tmp1[lvl+1] = mg_arrays.AP_f[lvl+1] .- mg_arrays.AP_c[lvl+1]
    # println(maximum(abs.(mg_arrays.tmp1[lvl+1])))
    # mg_arrays.tmp1[lvl+1] = mg_arrays.AP_c[lvl+1] .- mg_arrays.AP_f[lvl+1]

    # store restricted solution for error calc 
    mg_arrays.P_bar_H[lvl+1] .= mg_arrays.P_h[lvl+1]

    # store approximate solution for correction and post smoothening
    mg_arrays.P_H[lvl] .= mg_arrays.P_h[lvl]

    
    # mg_mesh_lvl = (p =mg_arrays.P_mg[lvl+1],tmp1 = mg_arrays.tmp1_mg[lvl+1],tmplrg = mg_arrays.tmplrg_mg[lvl+1],vf = mg_arrays.vf_mg[lvl+1],denx = mg_arrays.denx_mg[lvl+1])
    # mg_VTK!(pvtk_iter,pvd_data,mg_mesh_lvl.p,lvl+1,param,mg_mesh.mesh_lvls[lvl+1],par_env;mg_mesh_lvl.tmp1,mg_mesh_lvl.tmplrg,mg_mesh_lvl.vf,mg_mesh_lvl.denx,tau)
    # if lvl == 3
        # error("stop")
    # end
    if lvl < mg_lvl
        # lvl += 1
        # recursively call mg_fas!
        mg_fas!(lvl+1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter;iter,τ)
    end
    # error("stop")
    # begin prolongation routine starting at the coarsest level (occurs after relaxation at coarsest level)
    # println("prolongation on level $lvl")
    # mg_VTK!(m_iter,pvd_data,mg_arrays.P,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_arrays.tmp5,mg_arrays.vf)
    # m_iter +=1

    # calculate error ( P^2h-R(P^h) )
    mg_arrays.P_h[lvl+1] .-= mg_arrays.P_bar_H[lvl+1]
    # @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl+1]
    # for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
    #     if abs(mg_arrays.band[lvl][i,j,k]) <= 1
    #     # if (mg_arrays.P_h[lvl][i,j,k] - mg_arrays.P_bar_H[lvl][i,j,k]) / mg_arrays.P_bar_H[lvl][i,j,k] < 1e-1
    #         # mg_arrays.P_h[lvl][i,j,k] = mg_arrays.P_bar_H[lvl][i,j,k]
    #         nothing
    #     else
    #         mg_arrays.P_h[lvl][i,j,k] -= mg_arrays.P_bar_H[lvl][i,j,k]
    #     end
    # end

    # mg_VTK!(m_iter,pvd_data,mg_arrays.P,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_arrays.tmp5,mg_arrays.vf)
    # temp5 = mg_arrays.tmp5_mg[lvl]
    # mg_VTK!(m_iter,pvd_data,mg_arrays.P,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_arrays.tmp5,temp5,mg_arrays.vf)
   
    # prolongate error (corrected approximate solution)
    fill!(mg_arrays.AP_f[lvl],0.0)
    prolong!(mg_arrays.AP_f[lvl],mg_arrays.P_h[lvl+1],mg_mesh.mesh_lvls[lvl],mg_mesh.mesh_lvls[lvl+1])   
    update_borders!(mg_arrays.AP_f[lvl],mg_mesh.mesh_lvls[lvl],par_env)
    # arr = (p = mg_arrays.P_mg[lvl],tmp2 = mg_arrays.tmp2_mg[lvl])
    # mg_VTK!(m_iter,pvd_data,arr.p,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;arr.tmp2)
    # mg_mesh_lvl = (p =mg_arrays.P_mg[lvl],tmp1 = mg_arrays.tmp2_mg[lvl],tmplrg = mg_arrays.tmplrg_mg[lvl],vf = mg_arrays.vf_mg[lvl],denx = mg_arrays.denx_mg[lvl])
    # mg_VTK!(pvtk_iter,pvd_data,mg_mesh_lvl.p,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_mesh_lvl.tmp1,mg_mesh_lvl.tmplrg,mg_mesh_lvl.vf,mg_mesh_lvl.denx)
    
    #apply correction to approximate solution from pre-smoothening
    # @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]
    mg_arrays.P_H[lvl][imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+= mg_arrays.AP_f[lvl][imin_:imax_,jmin_:jmax_,kmin_:kmax_]
    # for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
    #     if abs(mg_arrays.band[lvl][i,j,k]) <= 1
    #         nothing
    #         # mg_arrays.P_H[lvl][i,j,k] = mg_arrays.P_h[lvl][i,j,k]
    #     else
    #         mg_arrays.P_H[lvl][i,j,k] += mg_arrays.AP_f[lvl][i,j,k]
    #     end
    # end
    A!(mg_arrays.AP_c[lvl],mg_arrays.uf[lvl],mg_arrays.vf[lvl],mg_arrays.wf[lvl],mg_arrays.P_h[lvl],dt,mg_arrays.gradx[lvl],mg_arrays.grady[lvl],mg_arrays.gradz[lvl],mg_arrays.band[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env)
    arr = (p = mg_arrays.P_H[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.AP_c[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],wf = mg_arrays.wf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
    m_iter = 1
    mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.uf,arr.vf,arr.wf,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
    
   


    # m_iter += 1
    # mg_VTK!(m_iter,pvd_data,arr.p,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;arr.tmp2)
    # error("stop")
    
    if lvl != 1
        # post smoothing of finest field wth corrected approximate solution
        res_iteration(mg_arrays.P_H[lvl],mg_arrays.uf[lvl],mg_arrays.vf[lvl],mg_arrays.wf[lvl],mg_arrays.gradx[lvl],mg_arrays.grady[lvl],mg_arrays.gradz[lvl],mg_arrays.band[lvl],dt,mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],mg_arrays.AP_f[lvl],mg_arrays.AP_c[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env; max_iter = v2)
        # nonlin_gs(mg_arrays.tmp5_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = 8)
        # Secant_jacobian_hypre!(mg_arrays.P_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env,max_iter = 3)
        mg_arrays.P_h[lvl] .= mg_arrays.P_H[lvl] 
        arr = (p = mg_arrays.P_H[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.AP_f[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],wf = mg_arrays.wf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        m_iter = 2
        mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.uf,arr.vf,arr.wf,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
    else
        
        # post smoothing of finest field wth corrected approximate solution
        res_iteration(mg_arrays.P_H[lvl],mg_arrays.uf[lvl],mg_arrays.vf[lvl],mg_arrays.wf[lvl],mg_arrays.gradx[lvl],mg_arrays.grady[lvl],mg_arrays.gradz[lvl],mg_arrays.band[lvl],dt,mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],mg_arrays.AP_f[lvl],mg_arrays.AP_c[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;iter,max_iter = v2)
        # Secant_jacobian_hypre!(mg_arrays.P_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = 3)
        # nonlin_gs(mg_arrays.tmp5_mg[lvl],mg_arrays.uf_mg[lvl],mg_arrays.vf_mg[lvl],mg_arrays.wf_mg[lvl],mg_arrays.gradx_mg[lvl],mg_arrays.grady_mg[lvl],mg_arrays.gradz_mg[lvl],mg_arrays.band_mg[lvl],dt,mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],mg_arrays.tmp1_mg[lvl],mg_arrays.tmp2_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],verts,tets,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter = 8,iter)
        mg_arrays.P_h[lvl] .= mg_arrays.P_H[lvl] 
        # mg_mesh_lmvl = (p =mg_arrays.P_mg[lvl],tmp1 = mg_arrays.tmp1_mg[lvl],tmplrg = mg_arrays.tmplrg_mg[lvl],vf = mg_arrays.vf_mg[lvl],denx = mg_arrays.denx_mg[lvl])
        # mg_VTK!(pvtk_iter,pvd_data,mg_mesh_lvl.p,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_mesh_lvl.tmp1,mg_mesh_lvl.tmplrg,mg_mesh_lvl.vf,mg_mesh_lvl.denx)
        arr = (p = mg_arrays.P_H[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.AP_f[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],wf = mg_arrays.wf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        m_iter = 2
        mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.uf,arr.vf,arr.wf,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
        error("stop")
        return
    end
end

# #! define recursive function for V cycle FAS method
function mg_vc_lin!(lvl,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env;iter=nothing,tmp5 = nothing,converged::Union{Nothing, Ref{Bool}}=nothing)
    @unpack mg_lvl = param
    @unpack nproc,irank,comm = par_env
    
    if lvl == 1
        VF_lvl = VF
    else 
        VF_lvl = mg_arrays.tmplrg[lvl]
    end
    
    v1 = 10
    v2 = 10
    # println("restricting on multrigrid level $lvl")
    if lvl == mg_lvl 
        # @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]
        # println(argmin(mg_arrays.RHS[lvl][imin_:imax_,jmin_:jmax_,kmin_:kmax_]))
        # println(minimum((mg_arrays.RHS[lvl][imin_:imax_,jmin_:jmax_,kmin_:kmax_])))
        # println(mg_arrays.RHS[lvl][1,6,1])
        # println(mg_arrays.RHS[lvl][1,6,2])
        # error("stop")
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 0
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
        # jacobi(mg_arrays.P_h[lvl],mg_arrays.tmp1[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=1000)
        gs(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;iter,max_iter=1000,tol_lvl = 1e-10)
        # cg_pressure_solver(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl], mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],mg_arrays.res[lvl],dt, param, mg_mesh.mesh_lvls[lvl], par_env;)
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 1
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
        # FC_hypre_solver(mg_arrays.P_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],tmp5,dt,param,mg_mesh.mesh_lvls[lvl],par_env,1000)
 
        return
    end
    
    # compute RHS at finest level
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]

    if lvl == 1
        fill!(mg_arrays.RHS[lvl],0.0)
        update_borders!(mg_arrays.uf[lvl],mg_mesh.mesh_lvls[lvl],par_env)
        update_borders!(mg_arrays.vf[lvl],mg_mesh.mesh_lvls[lvl],par_env)
        update_borders!(mg_arrays.wf[lvl],mg_mesh.mesh_lvls[lvl],par_env)
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            # RHS
            mg_arrays.RHS[lvl][i,j,k]= ( 
                ( mg_arrays.uf[lvl][i+1,j,k] - mg_arrays.uf[lvl][i,j,k] )/(dx) +
                ( mg_arrays.vf[lvl][i,j+1,k] - mg_arrays.vf[lvl][i,j,k] )/(dy) +
                ( mg_arrays.wf[lvl][i,j,k+1] - mg_arrays.wf[lvl][i,j,k] )/(dz) )
        end
        # update_borders!(mg_arrays.RHS[lvl],mg_mesh.mesh_lvls[lvl],par_env)
    end

    # if lvl !== 1
    # FC_hypre_solver(mg_arrays.P,mg_arrays.tmp3,mg_arrays.tmp4,mg_arrays.denx,mg_arrays.deny,mg_arrays.denz,tmp5,param,mg_mesh.mesh_lvls[lvl],par_env,1)
    # else
    if lvl == 1
        gs(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;iter,max_iter=v1)
        # jacobi(mg_arrays.P_h[lvl],mg_arrays.tmp1[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=5)
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 0
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
    else
        gs(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;iter,max_iter=v1)
        # jacobi(mg_arrays.P_h[lvl],mg_arrays.tmp1[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=5)
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 0
        # # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
    end
    # m_iter = 0
    # mg_mesh_lvl = (p =mg_arrays.P_mg[lvl],tmp4 = mg_arrays.tmp4_mg[lvl],tmplrg = mg_arrays.tmplrg_mg[lvl],vf = mg_arrays.vf_mg[lvl],denx = mg_arrays.denx_mg[lvl])
    # mg_VTK!(m_iter,pvd_data,mg_mesh_lvl.p,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_mesh_lvl.tmp4,mg_mesh_lvl.tmplrg,mg_mesh_lvl.vf,mg_mesh_lvl.denx)
    

    # Restrict residual and other neccessary quantities
    fill!(mg_arrays.RHS[lvl+1],0.0)
    restrict!(mg_arrays.tmplrg[lvl+1],VF_lvl,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    restrict!(mg_arrays.RHS[lvl+1],mg_arrays.res[lvl],mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    Neumann!(mg_arrays.tmplrg[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders!(mg_arrays.tmplrg[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders!(mg_arrays.RHS[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)

    # recompute densities and band using restricted volume fraction
    compute_dens!(mg_arrays.denx[lvl+1],mg_arrays.deny[lvl+1],mg_arrays.denz[lvl+1],mg_arrays.tmplrg[lvl+1],param,mg_mesh.mesh_lvls[lvl+1])
    computeBand!(mg_arrays.band[lvl+1],mg_arrays.tmplrg[lvl+1],param,mg_mesh.mesh_lvls[lvl+1],par_env)

    update_borders_x!(mg_arrays.denx[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders_y!(mg_arrays.deny[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders_z!(mg_arrays.denz[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)

    # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
    # m_iter = 0
    # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
    # recursively call mg_vc_lin!
    if lvl < mg_lvl
        mg_vc_lin!(lvl+1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env;iter,tmp5,converged)
    end
    # println(mg_arrays.P_h[lvl+1])
    # error("stop")
    # prolongate error and move up a level
    fill!(mg_arrays.tmp1[lvl],0.0)
    prolong!(mg_arrays.tmp1[lvl],mg_arrays.P_h[lvl+1],mg_mesh.mesh_lvls[lvl],mg_mesh.mesh_lvls[lvl+1])   
    update_borders!(mg_arrays.tmp1[lvl],mg_mesh.mesh_lvls[lvl],par_env)


    # correct approximate solution with error
    for k in kmin_:kmax_, j in jmin_:jmax_, i in imin_:imax_
        mg_arrays.P_h[lvl][i,j,k] += mg_arrays.tmp1[lvl][i,j,k]
    end    
    update_borders!(mg_arrays.P_h[lvl],mg_mesh.mesh_lvls[lvl],par_env)

    # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
    # m_iter = 0
    # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
    # fill!(mg_arrays.res[lvl],0.0)
    # error("stop")
    if lvl !== 1
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 1
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        # post-smoothening on corrected solution
        # println("post smoothening on lvl $lvl")
        gs(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;iter,max_iter=v2)
        # jacobi(mg_arrays.P_h[lvl],mg_arrays.tmp1[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=5)
        # FC_hypre_solver(mg_arrays.P_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],tmp5,param,mg_mesh.mesh_lvls[lvl],par_env,2,true)
        
        
    else
        # final solve on corrected finest grid
        # println("made it to then end of a cycle")
        # error("stop/")
        converged_flag = Ref(false) 
        # jacobi(mg_arrays.P_h[lvl],mg_arrays.tmp1[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;iter,max_iter=5)
        gs(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;iter,max_iter=v2,converged=converged_flag)
        # FC_hypre_solver(mg_arrays.P_mg[lvl],mg_arrays.tmp3_mg[lvl],mg_arrays.tmp4_mg[lvl],mg_arrays.denx_mg[lvl],mg_arrays.deny_mg[lvl],mg_arrays.denz_mg[lvl],tmp5,param,mg_mesh.mesh_lvls[lvl],par_env,10)
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 1
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
        # error("stop")
        
        return converged_flag
    end
end

function mg_fas_lin!(lvl,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter,BC!;iter = nothing,τ = nothing)
    @unpack mg_lvl = param
    @unpack comm = par_env
    #! need to restructure to better handle VF
    if lvl ==  1
        VF_lvl = VF
    else 
        VF_lvl = mg_arrays.tmplrg[lvl]
    end
    # println("on multrigrid level $lvl")
    v1 = 10
    v2 = 10
    # compute RHS at finest level
    #! test difference between recomputing RHS and restricting RHS 
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[lvl]
    if lvl == 1
        fill!(mg_arrays.RHS[lvl],0.0)
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            # RHS
            mg_arrays.RHS[lvl][i,j,k]= ( 
                ( mg_arrays.uf[lvl][i+1,j,k] - mg_arrays.uf[lvl][i,j,k] )/(dx) +
                ( mg_arrays.vf[lvl][i,j+1,k] - mg_arrays.vf[lvl][i,j,k] )/(dy) +
                ( mg_arrays.wf[lvl][i,j,k+1] - mg_arrays.wf[lvl][i,j,k] )/(dz) )
        end
        update_borders!(mg_arrays.RHS[lvl],mg_mesh.mesh_lvls[lvl],par_env)
    end

    if lvl == mg_lvl       
        # relax on coarsest level ( residual now is stored tmp1)
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 0
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        
        # gs(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=100)
        gs(mg_arrays.P_h[lvl],mg_arrays.tmp1[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=1000,tol_lvl=1e-10)
        # cg_pressure_solver(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl], mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],mg_arrays.res[lvl],dt, param, mg_mesh.mesh_lvls[lvl], par_env;τ)
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.tmp1[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 1
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        return
    end

    if lvl == 1
        # Pre-smoothing on current level ( residual now is stored tmp1)
        # gs(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=v1)
        gs(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=v1)
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 0
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
    else
        gs(mg_arrays.P_h[lvl],mg_arrays.tmp1[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=v1)
        # gs(mg_arrays.P_h[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=v1,τ)
        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.tmp1[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 0
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
    end

    
    # # compute A(P^h) and restrict to be used in τ calculation
    # lap!(mg_arrays.AP_f[lvl],mg_arrays.P_h[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env) 
    # restrict!(mg_arrays.AP_f[lvl+1],mg_arrays.AP_f[lvl],mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    # update_borders!(mg_arrays.AP_f[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
   
    # Restrict VF and compute band on coarse level
    restrict!(mg_arrays.tmplrg[lvl+1],VF_lvl,mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    update_borders!(mg_arrays.tmplrg[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    Neumann!(mg_arrays.tmplrg[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    computeBand!(mg_arrays.band[lvl+1],mg_arrays.tmplrg[lvl+1],param,mg_mesh.mesh_lvls[lvl+1],par_env)
    
    # Restrict approximate solution for initial guess on coarse grid for initial guess
    restrict!(mg_arrays.P_h[lvl+1],mg_arrays.P_h[lvl],mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    Neumann!(mg_arrays.P_h[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders!(mg_arrays.P_h[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)

    # purely restricted densities
    # restrict_x_face!(mg_arrays.denx[lvl+1],mg_arrays.denx[lvl],mg_mesh.mesh_lvls[lvl+1])
    # restrict_y_face!(mg_arrays.deny[lvl+1],mg_arrays.deny[lvl],mg_mesh.mesh_lvls[lvl+1])
    # restrict_z_face!(mg_arrays.denz[lvl+1],mg_arrays.denz[lvl],mg_mesh.mesh_lvls[lvl+1])
    compute_dens!(mg_arrays.denx[lvl+1],mg_arrays.deny[lvl+1],mg_arrays.denz[lvl+1],mg_arrays.tmplrg[lvl+1],param,mg_mesh.mesh_lvls[lvl+1])
    update_borders_x!(mg_arrays.denx[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders_y!(mg_arrays.deny[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    update_borders_z!(mg_arrays.denz[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    
    # copmpute A(P^2h) (A operator applied to restricted approximate solution on finer level)
    fill!(mg_arrays.AP_c[lvl+1],0.0)
    lap!(mg_arrays.AP_c[lvl+1],mg_arrays.P_h[lvl+1],mg_arrays.denx[lvl+1],mg_arrays.deny[lvl+1],mg_arrays.denz[lvl+1],dt,param,mg_mesh.mesh_lvls[lvl+1],par_env) 
        
    # restrict RHS (for use in post-smoothening) and copy to tmp1 to compute coarse grid RHS (for use in pre-smoothening)
    fill!(mg_arrays.RHS[lvl+1],0.0)
    fill!(mg_arrays.res[lvl+1],0.0)
    restrict!(mg_arrays.RHS[lvl+1],mg_arrays.RHS[lvl],mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    restrict!(mg_arrays.res[lvl+1],mg_arrays.res[lvl],mg_mesh.mesh_lvls[lvl+1],mg_mesh.mesh_lvls[lvl])
    # for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
    #     mg_arrays.tmp1[lvl+1][i,j,k] = mg_arrays.RHS[lvl+1][i,j,k] + mg_arrays.AP_c[lvl+1][i,j,k]
    # end
    mg_arrays.tmp1[lvl+1] .= mg_arrays.res[lvl+1] .+ mg_arrays.AP_c[lvl+1]
    # mg_arrays.tmp1[lvl+1] .= mg_arrays.RHS[lvl+1]

    # compute tau correction τ = A(P^2h) - A(P^h) (nonlinear defect)
    # τ = mg_arrays.AP_c[lvl+1] .- mg_arrays.AP_f[lvl+1]
    # τ = mg_arrays.AP_f[lvl+1] .- mg_arrays.AP_c[lvl+1]

    # store restricted pressure for error calc
    mg_arrays.P_bar_H[lvl+1] .= mg_arrays.P_h[lvl+1]

    # store approximate solution for correction and post smoothening
    mg_arrays.P_H[lvl] .= mg_arrays.P_h[lvl]

    # tau_full = mg_arrays.tmp4_mg[lvl+1] .- mg_arrays.tmp2_mg[lvl+1]
    # arr = (p = mg_arrays.P_h[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl])
    # m_iter = 0
    # mg_VTK!(m_iter,pvd_data,arr.p,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;arr.RHS,arr.uf,arr.vf,arr.denx,arr.deny,arr.denz)
    # if lvl == 1
    #     error("stop")
    # end
    if lvl < mg_lvl
        # recursively call mg_fas!
        mg_fas_lin!(lvl+1,mg_arrays,mg_mesh,dt,VF,verts,tets,pvd_data,param,par_env,pvtk_iter,BC!;iter)
    end
    # error("stop")
    # begin prolongation routine starting at the coarsest level (occurs after relaxation at coarsest level)
    # println("prolongation on level $lvl")

    # calculate error ( P^2h-R(P^h) )
    # interface_update(mg_arrays.band[lvl+1],mg_arrays.P_h[lvl+1],mg_arrays.P_bar_H[lvl+1],mg_mesh.mesh_lvls[lvl+1],par_env)
    
    mg_arrays.P_h[lvl+1] .-= mg_arrays.P_bar_H[lvl+1]
    # println(mg_arrays.P_h[lvl+1])
    # error("stop")
    # for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
    #     if abs(mg_arrays.band[lvl][i,j,k]) <= 1
    #         nothing
    #     else
    #         mg_arrays.P_h[lvl][i,j,k] -= mg_arrays.P_bar_H[lvl][i,j,k]
    #     end
    # end
    MPI.Barrier(comm)
    # prolongate error (corrected approximate solution)
    fill!(mg_arrays.res[lvl],0.0)
    prolong!(mg_arrays.res[lvl],mg_arrays.P_h[lvl+1],mg_mesh.mesh_lvls[lvl],mg_mesh.mesh_lvls[lvl+1])   
    update_borders!(mg_arrays.res[lvl],mg_mesh.mesh_lvls[lvl],par_env)

    # arr = (p = mg_arrays.P_mg[lvl],tmp2 = mg_arrays.tmp2_mg[lvl])
    # mg_VTK!(m_iter,pvd_data,arr.p,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;arr.tmp2)
    # mg_mesh_lvl = (p =mg_arrays.P_mg[lvl],tmp1 = mg_arrays.tmp2_mg[lvl],tmplrg = mg_arrays.tmplrg_mg[lvl],vf = mg_arrays.vf_mg[lvl],denx = mg_arrays.denx_mg[lvl])
    # mg_VTK!(pvtk_iter,pvd_data,mg_mesh_lvl.p,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;mg_mesh_lvl.tmp1,mg_mesh_lvl.tmplrg,mg_mesh_lvl.vf,mg_mesh_lvl.denx)
    
    mg_arrays.P_H[lvl][imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+= mg_arrays.res[lvl][imin_:imax_,jmin_:jmax_,kmin_:kmax_]
    # mg_arrays.P_H[lvl][imin_:imax_,jmin_:jmax_,kmin_:kmax_] .= mg_arrays.tmp1[lvl][imin_:imax_,jmin_:jmax_,kmin_:kmax_]
    # Neumann!(mg_arrays.P_H[lvl],mg_mesh.mesh_lvls[lvl],par_env)
    # fill!(mg_arrays.tmp1[lvl],0.0)
    fill!(mg_arrays.res[lvl],0.0)
    if lvl != 1
        # arr = (p = mg_arrays.P_H[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 1
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        # post smoothing of finest field wth corrected approximate solution
        gs(mg_arrays.P_H[lvl],mg_arrays.tmp1[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;max_iter=v2)
        mg_arrays.P_h[lvl] .= mg_arrays.P_H[lvl]
        
    else
        # post smoothing of finest field wth corrected approximate solution
        gs(mg_arrays.P_H[lvl],mg_arrays.RHS[lvl],mg_arrays.res[lvl],mg_arrays.denx[lvl],mg_arrays.deny[lvl],mg_arrays.denz[lvl],dt,param,mg_mesh.mesh_lvls[lvl],par_env;iter,max_iter=v2)
        # arr = (p = mg_arrays.P_H[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 1
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        mg_arrays.P_h[lvl] .= mg_arrays.P_H[lvl]

        # arr = (p = mg_arrays.P_h[lvl],VF = VF_lvl,band = mg_arrays.band[lvl],RHS = mg_arrays.RHS[lvl],uf = mg_arrays.uf[lvl],vf = mg_arrays.vf[lvl],denx = mg_arrays.denx[lvl],deny = mg_arrays.deny[lvl],denz = mg_arrays.denz[lvl],res = mg_arrays.res[lvl])
        # m_iter = 1
        # mg_VTK!(m_iter,pvd_data,arr.p,arr.denx,arr.deny,arr.denz,arr.VF,arr.band,arr.RHS,arr.res,lvl,param,mg_mesh.mesh_lvls[lvl],par_env;)
        # error("stop")
        return
    end
end