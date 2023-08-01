# Solve Poisson equation: δP form

function pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mesh,par_env,denx,deny,denz,step)

    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    RHS = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    gradx = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    grady = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    gradz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    #LHS = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)

    iter = poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,VF,dt,param,mesh,par_env,denx,deny,denz,step)

    return iter
end



function poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,VF,dt,param,mesh,par_env,denx,deny,denz,step)
    @unpack pressureSolver = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh


    if pressureSolver == "GaussSeidel"
        iter = GaussSeidel!(P,RHS,param,mesh,par_env)
    elseif pressureSolver == "ConjugateGradient"
        iter = conjgrad!(P,RHS,param,mesh,par_env)
    elseif pressureSolver == "Secant"
        iter = Secant_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,param,mesh,par_env,step)
    elseif pressureSolver == "NLsolve"
        iter = computeNLsolve!(P,uf,vf,wf,gradx,grady,gradz,band,den,dt,param,mesh,par_env)
    elseif pressureSolver == "Jacobi"
        Pois = Poisson(P,uf,vf,wf,denx,deny,denz,band,dt,param,par_env,mesh)
        iter = Jacobi!(Pois)
     else
        error("Unknown pressure solver $pressureSolver")
    end

    return iter
end

struct Poisson{T<:AbstractArray,T16<:AbstractArray,parameters,par_env_struct,mesh_struct}
    p :: T # Pressure
    f :: Tuple{T,T,T} # flow (can we store three objects of type T here or do we need individual face velocity arrays)
    den :: Tuple{T,T,T}
    band :: T16
    step:: Float64
    z :: T # source
    e :: T # error #? do these also need to be of type T or just a float value?
    r :: T # residual
    n :: Int16 # iterations
    param :: parameters # param object
    par :: par_env_struct # parallel environement structure
    mesh :: mesh_struct # mesh structure
    function Poisson(p::T,uf::T,vf::T,wf::T,denx::T,deny::T,denz::T,_band::T16,dt::Float64,param::parameters,par::par_env_struct,mesh::mesh_struct) where {T,T16,parameters,par_env_struct,mesh_struct} #? is this where argument redundant
        #need to initilize all types 
        f = (uf,vf,wf)
        den = (denx,deny,denz)
        step,band = dt,_band
        #want to compute grad(u*-dt/rho) in each direction and store it in f 
        r = similar(p); fill!(r,0.0)
        z,e = copy(r),copy(r)
        n = Int16(0)
        new{T,T16,typeof(param),typeof(par),typeof(mesh)}(p,f,den,band,step,z,e,r,n,param,par,mesh)
    end
end

function lap!(L,P,param,mesh)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        L[i,j,k] = (
            (P[i-1,j,k] - 2P[i,j,k] + P[i+1,j,k]) / dx^2 +
            (P[i,j-1,k] - 2P[i,j,k] + P[i,j+1,k]) / dy^2 +
            (P[i,j,k-1] - 2P[i,j,k] + P[i,j,k+1]) / dz^2 )
    end
    return nothing
end

# LHS of pressure poisson equation

function A!(LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env,step)

    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh


    fill!(gradx,0.0)
    fill!(grady,0.0)
    fill!(gradz,0.0)

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env) # (overwrites BCs if periodic)

    #suspect that the correct gradient is being calculate due to loop
    #! need cell centered densities
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        gradx[i,j,k]=dt/denx[i,j,k]*(P[i,j,k]-P[i-1,j,k])/dx
    end

    for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        grady[i,j,k]=dt/deny[i,j,k]*(P[i,j,k]-P[i,j-1,k])/dy
    end

    for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        gradz[i,j,k]=dt/denz[i,j,k]*(P[i,j,k]-P[i,j,k-1])/dz
    end

    uf1 = uf-gradx
    vf1 = vf-grady
    wf1 = wf-gradz

    fill!(LHS,0.0)

    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        if abs(band[i,j,k]) <= 1
            tets, inds = cell2tets_withProject_uvwf(i,j,k,uf1,vf1,wf1,dt,mesh)
            if any(isnan,tets)
                error("Nan in tets at ", i,j,k)
            end
            v2 = dx*dy*dz
            v1 = tets_vol(tets)

            LHS[i,j,k] = (v2-v1) /̂ v2 /̂ dt
        else 
            # Calculate divergence with finite differnce
            du_dx = ( uf1[i+1,j,k] - uf1[i,j,k] )/(dx)
            dv_dy = ( vf1[i,j+1,k] - vf1[i,j,k] )/(dy)
            dw_dz = ( wf1[i,j,k+1] - wf1[i,j,k] )/(dz)
            LHS[i,j,k] = du_dx + dv_dy + dw_dz
        end
    end
    # println(LHS)
    return nothing
end


#local A! matrix
function A!(i,j,k,LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    fill!(gradx,0.0)
    fill!(grady,0.0)
    fill!(gradz,0.0)

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env) # (overwrites BCs if periodic)

    #probably dont need to calculate every pt but need a 3x3 stencil for velocity projection with i,j,k being in a corner
    #maybe want to use diff finite difference approx
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        gradx[i,j,k]=dt/denx[i,j,k]*(P[i,j,k]-P[i-1,j,k])/̂dx
    end
    
    for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        grady[i,j,k]=dt/deny[i,j,k]*(P[i,j,k]-P[i,j-1,k])/̂dy
    end

    for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        gradz[i,j,k]=dt/denz[i,j,k]*(P[i,j,k]-P[i,j,k-1])/̂dz
    end

    #? might want to use these smaller loops
    # for ii = i:i+1
    #     gradx[ii,j,k]=(P[ii,j,k]-P[ii-1,j,k])/̂dx
    # end
    # for jj = j:j+1
    #     grady[i,jj,k]=(P[i,jj,k]-P[i,jj-1,k])/̂dy
    # end
    # for kk = k:k+1
    #     gradz[i,j,kk]=(P[i,j,kk]-P[i,j,kk-1])/̂dz
    # end

    uf1 = uf-gradx
    vf1 = vf-grady
    wf1 = wf-gradz

    if abs(band[i,j,k]) <= 1
        tets, inds = cell2tets_withProject_uvwf(i,j,k,uf1,vf1,wf1,dt,mesh)
        if any(isnan,tets)
            error("Nan in tets at ", i,j,k)
        end
        v2 = dx*dy*dz
        v1 = tets_vol(tets)
        LHS[i,j,k] = (v2-v1) /̂ v2 /̂ dt

    else 
        # Calculate divergence with finite differnce
        du_dx = ( uf1[i+1,j,k] - uf1[i,j,k] )/̂(dx)
        dv_dy = ( vf1[i,j+1,k] - vf1[i,j,k] )/̂(dy)
        dw_dz = ( wf1[i,j,k+1] - wf1[i,j,k] )/̂(dz)
        LHS[i,j,k] = du_dx + dv_dy + dw_dz
    end
    return LHS[i,j,k]
end

function computeJacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
    @unpack Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    J = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    dp = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    LHS1 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    LHS2 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    delta = 1.0

    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        fill!(LHS1,0.0)
        fill!(LHS2,0.0)
        fill!(dp,0.0)
        J[i,j,k] = (
            (A!(i,j,k,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env)
            - A!(i,j,k,LHS2,uf,vf,wf,P.-dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env))
            ./̂2delta)
    end
    return J 
end

#want to define step and del operators
CI(a...) = CartesianIndex(a...)
step(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
step(i,I::CartesianIndex{N}) where N = step(i, Val{N}())

# partial derivative of scalar field
@inline del(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-step(a,I)]
# partial derivative of vector field
@inline del(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I,a]-u[I-step(a,I),a]
#might want to define a del function in the a_th direction that takes the correpsonding flow field, density and dt
@inline del(a,I::CartesianIndex{m},u::AbstractArray{T,n}, f::AbstractArray{T,d},den::AbstractArray{T,n},dt::Float64) where {T,n,m,d} = @inbounds u[I]-dt/̂den[I]*̂del(a,I,f)


@inline inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))

# non local A matrix for Poisson struct
function A!(P::Poisson{T}) where {T}
    param,mesh = P.param,P.mesh
    @unpack dx,dy,dz = mesh
    d_step = dx,dy,dz

    Neumann!(P.p,mesh,P.par)
    update_borders!(P.p,mesh,P.par) # (overwrites BCs if periodic)

    
    v_fields = (copy(P.f[1]), copy(P.f[2]), copy(P.f[3]))
    for ii in 1:length(P.f)
        for I in inside(P.f[ii])
            v_fields[ii][I] -= P.step/P.den[ii][I]*del(ii,I,P.p)/d_step[ii]
        end
    end
    # println("made it through")

    for I in inside(P.band)
        if abs(P.band[I]) <= 1
            tets, inds = cell2tets_withProject_uvwf(I[1],I[2],I[3],v_fields[1],v_fields[2],v_fields[3],P.step,mesh)
            if any(isnan,tets)
                error("Nan in tets at ", I)
            end
            v2 = dx*dy*dz
            v1 = tets_vol(tets)
            P.z[I] = (v2-v1) /̂ v2 /̂ P.step
        else 
            # Calculate divergence with finite differnce
            du_dx = (v_fields[1][I+step(1,I)]-v_fields[1][I])/(dx)
            dv_dy = (v_fields[2][I+step(2,I)]-v_fields[2][I])/(dy)
            dw_dz = (v_fields[3][I+step(3,I)]-v_fields[3][I])/(dz)
            P.z[I] = du_dx + dv_dy + dw_dz
        end
    end
end
# non local A! matrix with delta
function A!(P::Poisson{T},delta=0.0) where {T}
    param,mesh = P.param,P.mesh
    @unpack dx,dy,dz = mesh
    d_step = dx,dy,dz

    Neumann!(P.p,mesh,P.par)
    update_borders!(P.p,mesh,P.par) # (overwrites BCs if periodic)

    p = copy(P.p)
    p[I] += delta
    v_fields = (copy(P.f[1]), copy(P.f[2]), copy(P.f[3]))
    for ii in 1:length(P.f)
        for I in inside(P.f[ii])
            v_fields[ii][I] -= P.step/P.den[ii][I]*del(ii,I,P.p)/d_step[ii]
        end
    end
    # println("made it through")

    for I in inside(P.band)
        if abs(P.band[I]) <= 1
            tets, inds = cell2tets_withProject_uvwf(I[1],I[2],I[3],v_fields[1],v_fields[2],v_fields[3],P.step,mesh)
            if any(isnan,tets)
                error("Nan in tets at ", I)
            end
            v2 = dx*dy*dz
            v1 = tets_vol(tets)
            P.z[I] = (v2-v1) /̂ v2 /̂ P.step
        else 
            # Calculate divergence with finite differnce
            du_dx = (v_fields[1][I+step(1,I)]-v_fields[1][I])/(dx)
            dv_dy = (v_fields[2][I+step(2,I)]-v_fields[2][I])/(dy)
            dw_dz = (v_fields[3][I+step(3,I)]-v_fields[3][I])/(dz)
            P.z[I] = du_dx + dv_dy + dw_dz
        end
    end
end

# local A matrix that recieves Poisson struct and cartesian index
function A!(P::Poisson{T}, I::CartesianIndex{d},delta) where {T,d}
    param,mesh,par_env = P.param,P.mesh,P.par
    @unpack dx,dy,dz = mesh
    d_step = dx,dy,dz

    v_fields = (copy(P.f[1]), copy(P.f[2]), copy(P.f[3]))
    p = copy(P.p)
    p[I] += delta
    Neumann!(p,mesh,par_env)
    update_borders!(p,mesh,par_env) # (overwrites BCs if periodic)

    for ii in 1:length(v_fields)
        for Ii in inside(P.f[ii])
            v_fields[ii][Ii] -= P.step/P.den[ii][Ii]*del(ii,Ii,p)/d_step[ii]
        end
    end

    if abs(P.band[I]) <= 1
        tets, inds = cell2tets_withProject_uvwf(I[1],I[2],I[3],v_fields[1],v_fields[2],v_fields[3],P.step,mesh)
        if any(isnan,tets)
            error("Nan in tets at ", I)
        end
        v2 = dx*dy*dz
        v1 = tets_vol(tets)
        P.r[I] = (v2-v1) /̂ v2 /̂ P.step 
    else 
        # Calculate divergence with finite differnce
        du_dx = (v_fields[1][I+step(1,I)]-v_fields[1][I])/(dx)
        dv_dy = (v_fields[2][I+step(2,I)]-v_fields[2][I])/(dy)
        dw_dz = (v_fields[3][I+step(3,I)]-v_fields[3][I])/(dz)
        P.r[I] = du_dx + dv_dy + dw_dz
    end
end

function Jacobian!(P::Poisson)
    delta = 1.0
    ndelta = -1.0
    #! need to loop over p-field
    for I in inside(P.p)
        A!(P,I,delta)
        A_pos = P.r[I]
        fill!(P.r,0.0)
        A!(P,I,ndelta)
        A_neg = P.r[I]
        #! calc jacobian at each pt in mesh
        P.e[I] = (A_pos-A_neg)/̂2delta        
    end
end

# Secant method
function Secant_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,param,mesh,par_env,step)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh

    AP = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    fill!(AP,0.0)
    # outflowCorrection!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)

    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env,step)

    # Iterate 
    iter=0
    while true
        iter += 1

        # compute jacobian
        J = computeJacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)

        P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .-= 0.8AP./̂J

        P .-=mean(P)

        #Need to introduce outflow correction
        # outflowCorrection!(RHS,AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,param,mesh,par_env)
        # end
        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env,step)
        
        res = maximum(abs.(AP))
        if res < tol
            return iter
        end
    
        # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res,sum(AP))
    end    
end


# NLsolve Library
function computeNLsolve!(P,uf,vf,wf,gradx,grady,gradz,band,den,dt,param,mesh,par_env)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    # Ghost cell indices
    gx = imino_:imaxo_; gy = jmino_:jmaxo_; gz = kmino_:kmaxo_

    LHS = OffsetArray{Float64}(undef, gx,gy,gz)


    function f!(LHS, P)
        A!(LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,den,mesh,par_env)
        # println("maxRes=",maximum(abs.(F)))
        return LHS
    end

    # Run solver
    out = nlsolve(f!,P,
        ftol = tol,
        method = :trust_region, #default 
        # method = :newton,
        # method = :anderson, # diverges
        # m=20,
        )

    # Get output
    P .= out.zero

    return out.iterations
end



"""
GaussSeidel Poisson Solver
"""
function GaussSeidel!(P,RHS,param,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack isroot = par_env
    @unpack tol = param
    maxIter=1000
    iter = 0

    while true
        iter += 1
        max_update=0.0
        @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            Pnew = ( (RHS[i,j,k]
                    - (P[i-1,j,k]+P[i+1,j,k])/dx^2
                    - (P[i,j-1,k]+P[i,j+1,k])/dy^2 
                    - (P[i,j,k-1]+P[i,j,k+1])/dz^2) 
                    / (-2.0/dx^2 - 2.0/dy^2 - 2.0/dz^2) )
            max_update=max(max_update,abs(Pnew-P[i,j,k]))
            P[i,j,k] = Pnew
        end
        update_borders!(P,mesh,par_env)
        Neumann!(P,mesh,par_env)
        # Check if converged
        if iter == 10
            print(max_update)
        end
        max_update = parallel_max_all(max_update,par_env)
        max_update < tol && return iter # Converged
        # Check if hit max iteration
        if iter == maxIter 
            isroot && println("Failed to converged Poisson equation max_upate = $max_update")
            return iter
        end
    end
end


"""
Conjugate gradient
"""
function conjgrad!(P,RHS,param,mesh,par_env)
    @unpack dx,dy,dz = mesh
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack tol = param
    @unpack irank,isroot = par_env

    # Interior indices
    ix = imin_:imax_; iy = jmin_:jmax_;  iz = kmin_:kmax_
    # Ghost cell indices
    gx = imino_:imaxo_; gy = jmino_:jmaxo_; gz = kmino_:kmaxo_
    
    # Allocat work arrays (with ghost cells for comm)
    r  = OffsetArray{Float64}(undef, gx,gy,gz)
    p  = OffsetArray{Float64}(undef, gx,gy,gz)
    Ap = OffsetArray{Float64}(undef, gx,gy,gz)

    lap!(r,P,param,mesh)

    r[ix,iy,iz] = RHS.parent - r[ix,iy,iz]
    Neumann!(r,mesh,par_env)
    update_borders!(r,mesh,par_env) # (overwrites BCs if periodic)
    p .= r
    rsold = parallel_sum_all(r[ix,iy,iz].^2,par_env)
    rsnew = 0.0
    for iter = 1:length(RHS)
        lap!(Ap,p,param,mesh)
        sum = parallel_sum_all(p[ix,iy,iz].*Ap[ix,iy,iz],par_env)
        alpha = rsold / sum
        P .+= alpha*p
        r -= alpha * Ap
        rsnew = parallel_sum_all(r[ix,iy,iz].^2,par_env)
        if sqrt(rsnew) < tol
            return iter
        end
        p = r + (rsnew / rsold) * p
        Neumann!(p,mesh,par_env)   
        update_borders!(p,mesh,par_env) # (overwrites BCs if periodic)
        rsold = rsnew

    end
    
    isroot && println("Failed to converged Poisson equation rsnew = $rsnew")
    
    return length(RHS)
end


function Jacobi!(P::Poisson,tol=1e-4)
    p,z,e,n = P.p,P.z,P.e,P.n
    # calc A(P) as p.z
    A!(P)

    n = 0
    while true 
        n +=1
        # calc jacobian
        Jacobian!(P)

        for I in inside(p)
            P.p[I] -= 0.8z[I]/e[I]
        end

        # avoid drift
        p .-=mean(p)

        # calc A(P)
        fill!(z,0.0)
        A!(P)

        # check residual to tol for new A(P)
        res = maximum(abs.(z))
     
        # @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",n,res,sum(z))
        if res < tol
            return n
        end  
    end
end

#! need to think about how to construct the domain wide line search 
function arm_gold(P::Poisson, del = 0.5, c=1e-4)
    f,x,d = P.z,P.p,P.e
    g = copy(d)
    f1 = copy(f)
    t = 1.0
    delta = t*d
    while A!(P,delta) > f1 + c*t*dot(d,g)
        t *= del
    end
end