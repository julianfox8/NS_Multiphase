using DataFrames, CSV

# Wrapper structs for use in mg_cycler()
struct SLOp{M,P}
    lvl::MG_level; mesh::M; param::P; par_env; dt::Float64
    verts; tets; face_BC!
end

struct FDOp{M,P}
    lvl::MG_level; mesh::M; param::P; par_env; dt::Float64; face_BC!
end

function restrict_operator!(c::SLOp, f::SLOp, VF_f)
    restrict!(c.lvl.tmplrg, VF_f, c.mesh, f.mesh)
    Neumann!(c.lvl.tmplrg, c.mesh, c.par_env)
    # computeBand!(c.lvl.band, c.lvl.tmplrg, c.param, c.mesh, c.par_env)
    fill!(c.lvl.band,2.0)
    compute_dens!(c.lvl.denx, c.lvl.deny, c.lvl.denz, c.lvl.tmplrg, c.param, c.mesh)
    restrict_x_face!(c.lvl.uf, f.lvl.uf, c.mesh)
    restrict_y_face!(c.lvl.vf, f.lvl.vf, c.mesh)
    restrict_z_face!(c.lvl.wf, f.lvl.wf, c.mesh)
    c.face_BC!(c.lvl.uf, c.lvl.vf, c.lvl.wf, c.mesh, c.par_env)
end

function restrict_operator!(c::FDOp, f::FDOp, VF_f)
    restrict!(c.lvl.tmplrg, VF_f, c.mesh, f.mesh)
    Neumann!(c.lvl.tmplrg, c.mesh, c.par_env)
    compute_dens!(c.lvl.denx, c.lvl.deny, c.lvl.denz, c.lvl.tmplrg, c.param, c.mesh)
    restrict!(c.lvl.RHS, f.lvl.RHS, c.mesh, f.mesh)
end

apply!(op::SLOp, r, P) = A!(r, op.lvl.uf, op.lvl.vf, op.lvl.wf, P, op.dt,
                            op.lvl.gradx, op.lvl.grady, op.lvl.gradz, op.lvl.band,
                            op.lvl.denx, op.lvl.deny, op.lvl.denz,
                            op.verts, op.tets, op.param, op.mesh, op.par_env)

apply!(op::FDOp, r, P) = res_comp!(r, op.lvl.RHS, P, op.lvl.denx, op.lvl.deny,
                                   op.lvl.denz, op.dt, op.param, op.mesh, op.par_env)


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

function mg_cycler(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,mg_arrays,mg_mesh,VF,verts,tets,face_BC!,param,par_env;max_iter = 10000)
    @unpack pressureSolver,pressure_scheme,mg_lvl,smoother = param
    @unpack comm,irank = par_env
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mg_mesh.mesh_lvls[1]

    # set up arrays
    fields = (P = P,uf = uf,vf = vf,wf = wf,denx = denx,deny = deny,denz = denz,gradx = gradx,grady = grady,gradz = gradz,band = band)
    copy_to_mg!(mg_arrays,fields,1)

    iter = 0
    pvtk_iter = 0
    converged = false
    r_old = nothing
    r_new = nothing
    row = DataFrame(cycle = Int[], res_norm = Float64[], rho = Float64[])
    for i in 1:max_iter
        iter += 1
        pvtk_iter += 1
        if pressure_scheme == "finite-difference"
            ops = [FDOp(mg_arrays[l], mg_mesh.mesh_lvls[l], param, par_env, dt, face_BC!) for l in 1:mg_lvl]
            converged = mg_CS!(1, ops, VF, par_env, pvtk_iter;iter)
            # converged = mg_fas_new!(1, ops, VF, par_env, pvtk_iter;iter)
            
            # # compute residual norm living in res
            r_new = sqrt(sum(abs2, mg_arrays[1].res[imin_:imax_,jmin_:jmax_,kmin_:kmax_]))
        elseif pressure_scheme == "semi-lagrangian"
            ops = [SLOp(mg_arrays[l], mg_mesh.mesh_lvls[l], param, par_env, dt, verts, tets, face_BC!) for l in 1:mg_lvl]
            converged = mg_fas!(1, ops, VF, par_env, pvtk_iter;iter)
            
            # compute residual norm living in AP_f
            r_new = sqrt(sum(abs2, mg_arrays[1].AP[imin_:imax_,jmin_:jmax_,kmin_:kmax_]))
        end
        
        if iter > 1
            rho = r_new/r_old
            push!(row, (iter, r_new, rho))
        end
        r_old = r_new
    
        if converged[] || max_iter == i
            CSV.write("mg_convergence_$(pressure_scheme)_dtlow_2D.csv", row)
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
function mg_fas!(lvl, ops, VF, par_env, pvtk_iter;iter = nothing, converged::Union{Nothing, Ref{Bool}}=nothing)
    @unpack mg_lvl,v1,v2 = ops[1].param
    
    op_f, fine, mesh_f = ops[lvl], ops[lvl].lvl, ops[lvl].mesh
    
    #! need to restructure to better handle VF
    if lvl ==  1
        VF_lvl = VF 
    else 
        VF_lvl = fine.tmplrg
    end

    if lvl == mg_lvl
        coarse_sol!(op_f,fine)
        return
    end

    op_c, coarse, mesh_c = ops[lvl+1], ops[lvl+1].lvl, ops[lvl+1].mesh

    if lvl == 1
        # Pre-smoothing on current level ( residual now is stored tmp1)
        converged_flag = Ref(false)
        smooth!(op_f,fine,v1;iter,converged_flag)
        if converged_flag[]
            return converged_flag
        end
    else
        smooth!(op_f,fine,v1)
    end

    # Restrict approximate solution for initial guess on coarse grid for initial guess
    restrict!(coarse.P_h, fine.P_h, mesh_c, mesh_f)
    Neumann!(coarse.P_h, mesh_c, par_env)
    restrict_operator!(op_c,op_f,VF_lvl)

    # copmpute A(R(P^h)) (A operator applied to restricted approximate solution on finer level)
    fill!(coarse.AP,0.0); apply!(op_c, coarse.AP, coarse.P_h)

    # compute R(A(P^h)) (A operator applied to approximate solution on finer level and restricted down)
    fill!(fine.AP,0.0); apply!(op_f, fine.AP, fine.P_h)
    fine.AP .-= fine.tau
    Neumann!(fine.AP, mesh_f, par_env) 
    restrict!(coarse.AP_R, fine.AP, mesh_c, mesh_f)

    # compute Tau correction (τ = R(A(P^h)) - A(R(P^h)))
    coarse.tau .= coarse.AP .- coarse.AP_R

    # store restricted solution for error calc
    coarse.P_bar_H .= coarse.P_h

    if lvl < mg_lvl
        # recursively call mg_fas!
        mg_fas!(lvl+1,ops,VF,par_env,pvtk_iter;iter,converged)
    end
    
    # calculate error ( P^2h-R(P^h) ) and prolongate
    coarse.P_h .-= coarse.P_bar_H
    fill!(fine.corr,0.0)
    prolong!(fine.corr, coarse.P_h, mesh_f, mesh_c)
    # update_borders!(fine_lvl.res,mg_mesh.mesh_lvls[lvl],par_env)

    # apply correction directly to pre-smoothed P_h (matches mg_fas_lin2! structure)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh_f
    @views fine.P_h[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+= fine.corr[imin_:imax_,jmin_:jmax_,kmin_:kmax_]

    if lvl != 1
        # post smoothing on corrected P_h
        smooth!(op_f,fine,v1)
    else
        # post smoothing of finest field with corrected approximate solution
        converged_flag = Ref(false)
        smooth!(op_f,fine,v2;iter,converged_flag)
        return converged_flag
    end
end

function mg_CS!(lvl, ops, VF, par_env, pvtk_iter;iter = nothing, converged::Union{Nothing, Ref{Bool}}=nothing)
    @unpack mg_lvl,v1,v2 = ops[1].param
    
    op_f, fine, mesh_f = ops[lvl], ops[lvl].lvl, ops[lvl].mesh
    
    if lvl ==  1
        VF_lvl = VF 
    else 
        VF_lvl = fine.tmplrg
    end

    if lvl == mg_lvl
        coarse_sol!(op_f,fine)
        return
    end

    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,dx,dy,dz = mesh_f
    if lvl == 1
        update_borders!(fine.uf,mesh_f,par_env)
        update_borders!(fine.vf,mesh_f,par_env)
        update_borders!(fine.wf,mesh_f,par_env)
        @loop op_f.param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            # RHS
            fine.RHS[i,j,k] = ( 
                ( fine.uf[i+1,j,k] - fine.uf[i,j,k] )/(dx) +
                ( fine.vf[i,j+1,k] - fine.vf[i,j,k] )/(dy) +
                ( fine.wf[i,j,k+1] - fine.wf[i,j,k] )/(dz) )
        end
        update_borders!(fine.RHS,mesh_f,par_env)
    end


    op_c, coarse, mesh_c = ops[lvl+1], ops[lvl+1].lvl, ops[lvl+1].mesh
    if lvl == 1
        converged_flag = Ref(false)
        smooth!(op_f,fine,v1;iter,converged_flag)
        if converged_flag[]
            return converged_flag
        end
    else
        smooth!(op_f,fine,v1)
    end

    # Restrict VF down to coarse tmplrg array and compute band
    restrict!(coarse.tmplrg,VF_lvl,mesh_c,mesh_f)
    Neumann!(coarse.tmplrg,mesh_c,par_env)
    update_borders!(coarse.tmplrg,mesh_c,par_env)
    computeBand!(coarse.band,coarse.tmplrg,op_c.param,mesh_c,par_env)

    # Restrict residual down to coarse RHS array
    Neumann!(fine.res,mesh_f,par_env)
    update_borders!(fine.res,mesh_f,par_env)
    restrict!(coarse.RHS,fine.res,mesh_c,mesh_f)
    Neumann!(coarse.RHS,mesh_c,par_env)

    # Restrict densities
    restrict_x_face!(coarse.denx,fine.denx,mesh_c)
    restrict_y_face!(coarse.deny,fine.deny,mesh_c)
    restrict_z_face!(coarse.denz,fine.denz,mesh_c)
    update_xface_borders!(coarse.denx,mesh_c,par_env)
    update_yface_borders!(coarse.deny,mesh_c,par_env)
    update_zface_borders!(coarse.denz,mesh_c,par_env)

    if lvl < mg_lvl
        fill!(coarse.P_h,0.0)
        fill!(coarse.tau,0.0)
        # recursively call mg_fas!
        mg_CS!(lvl+1,ops,VF,par_env,pvtk_iter;iter,converged)
    end

    fill!(fine.corr,0.0)
    prolong!(fine.corr, coarse.P_h, mesh_f, mesh_c)

    # apply correction directly to pre-smoothed P_h (matches mg_fas_lin2! structure)
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh_f
    @views fine.P_h[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .+= fine.corr[imin_:imax_,jmin_:jmax_,kmin_:kmax_]

    if lvl != 1
        # post smoothing on corrected P_h
        smooth!(op_f,fine,v1)
    else
        # post smoothing of finest field with corrected approximate solution
        converged_flag = Ref(false)
        smooth!(op_f,fine,v2;iter,converged_flag)
        return converged_flag
    end
end