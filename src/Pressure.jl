using SparseArrays
using LinearAlgebra

# Solve Poisson equation: δP form
function pressure_solver!(P,uf,vf,wf,dt,band,VF,param,mesh,par_env,denx,deny,denz,outflow,step)

    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    RHS = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    gradx = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    grady = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    gradz = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    #LHS = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)

    # @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
    #     # RHS
    #     RHS[i,j,k]= 1/dt * ( 
    #         ( uf[i+1,j,k] - uf[i,j,k] )*(denx[i+1,j,k]-denx[i,j,k])/(2dx) +
    #         ( vf[i,j+1,k] - vf[i,j,k] )*(deny[i,j+1,k]-deny[i,j,k])/(2dy) +
    #         ( wf[i,j,k+1] - wf[i,j,k] )*(denz[i,j,k+1]-denz[i,j,k])/(2dz) )
    # end
    
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        # RHS
        RHS[i,j,k]= 1/dt* ( 
            ( uf[i+1,j,k] - uf[i,j,k] )/(dx) +
            ( vf[i,j+1,k] - vf[i,j,k] )/(dy) +
            ( wf[i,j,k+1] - wf[i,j,k] )/(dz) )
    end
    iter = poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,VF,dt,param,mesh,par_env,denx,deny,denz,outflow)

    return iter
end



function poisson_solve!(P,RHS,uf,vf,wf,gradx,grady,gradz,band,VF,dt,param,mesh,par_env,denx,deny,denz,outflow)
    @unpack pressureSolver = param
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    if pressureSolver == "GaussSeidel"
        iter = GaussSeidel!(P,RHS,uf,vf,wf,denx,deny,denz,dt,param,mesh,par_env)
    elseif pressureSolver == "ConjugateGradient"
        iter = conjgrad!(P,RHS,denx,deny,denz,param,mesh,par_env)

    elseif pressureSolver == "Secant"
        iter = Secant_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,outflow,param,mesh,par_env,step)
    elseif pressureSolver == "sparseSecant"
        iter = Secant_full_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,outflow,param,mesh,par_env,step)
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
    j :: T # Jacobian
    r :: T # Jacobian residual
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
        z,j = copy(r),copy(r)
        n = Int16(0)
        new{T,T16,typeof(param),typeof(par),typeof(mesh)}(p,f,den,band,step,z,j,r,n,param,par,mesh)
    end
end

function lap!(L,P,denx,deny,denz,param,mesh)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    fill!(L,0.0)
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        L[i,j,k] = (
            (P[i-1,j,k] - 2P[i,j,k] + P[i+1,j,k]) /̂ (denx[i,j,k]*dx^2) +
            (P[i,j-1,k] - 2P[i,j,k] + P[i,j+1,k]) /̂ (deny[i,j,k]*dy^2) +
            (P[i,j,k-1] - 2P[i,j,k] + P[i,j,k+1]) /̂ (denz[i,j,k]*dz^2) )
    end
    return nothing
end

# LHS of pressure poisson equation

function A!(LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env)

    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh


    fill!(gradx,0.0)
    fill!(grady,0.0)
    fill!(gradz,0.0)

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env) # (overwrites BCs if periodic)

    #suspect that the correct gradient is being calculate due to loop
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        gradx[i,j,k]=dt/denx[i,j,k]*(P[i,j,k]-P[i-1,j,k])/dx
    end

    @loop param for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        grady[i,j,k]=dt/deny[i,j,k]*(P[i,j,k]-P[i,j-1,k])/dy
    end

    @loop param for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
        gradz[i,j,k]=dt/denz[i,j,k]*(P[i,j,k]-P[i,j,k-1])/dz
    end

    uf1 = uf-gradx
    vf1 = vf-grady
    wf1 = wf-gradz

    fill!(LHS,0.0)

    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
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
function A!(i,j,k,LHS,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh

    fill!(gradx,0.0)
    fill!(grady,0.0)
    fill!(gradz,0.0)

    Neumann!(P,mesh,par_env)
    update_borders!(P,mesh,par_env) # (overwrites BCs if periodic)

    #probably dont need to calculate every pt but need a 3x3 stencil for velocity projection with i,j,k being in a corner
    #maybe want to use diff finite difference approx
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_+1
        gradx[i,j,k]=dt/denx[i,j,k]*(P[i,j,k]-P[i-1,j,k])/̂dx
    end
    
    @loop param for k=kmin_:kmax_, j=jmin_:jmax_+1, i=imin_:imax_
        grady[i,j,k]=dt/deny[i,j,k]*(P[i,j,k]-P[i,j-1,k])/̂dy
    end

    @loop param for k=kmin_:kmax_+1, j=jmin_:jmax_, i=imin_:imax_
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
        dp[i,j,k] += delta
        J[i,j,k] = (
            (A!(i,j,k,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env)
            - A!(i,j,k,LHS2,uf,vf,wf,P.-dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env))
            ./̂2delta)
    end
    return J 
end


function n(i,j,k,Ny,Nz) 
    val = i + (j-1)*Ny + (k-1)*Nz*Ny
    # @show i,j,k,Ny,Nz,val
    return val
end 

function compute_sparse_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
    @unpack Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    J = OffsetArray{Float64}(undef,1:Nx*Ny*Nz,1:Nx*Ny*Nz)
    dp = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    LHS1 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    LHS2 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    delta = 1.0
    fill!(J,0.0)


    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        for kk=max(1,k-1):min(Nz,j+1), jj=max(1,j-1):min(Ny,j+1), ii=max(1,i-1):min(Nx,i+1)
            fill!(LHS1,0.0)
            fill!(LHS2,0.0)
            fill!(dp,0.0)
            dp[ii,jj,kk] += delta
            J[n(i,j,k,Ny,Nz),n(ii,jj,kk,Ny,Nz)] = (
                (A!(i,j,k,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env)
                - A!(i,j,k,LHS2,uf,vf,wf,P.-dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env))
                ./̂2delta)
        end
    end
    return J 
end

function compute_sparse2D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
    @unpack Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    diags = OffsetArray{Float64}(undef,1:Nx*Ny*Nz,9)
    dp = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    LHS1 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    LHS2 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    delta = 100000
    fill!(diags,0.0)
    offset=[-(Nx+1), -Nx, -(Nx-1), -1, 0, 1, Nx-1, Nx, Nx+1]


    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        nNeigh = 0

        for kk = 1 ,jj = j-1:j+1, ii = i-1:i+1
            if jj < 1 || jj > Nx || ii < 1 || ii > Nx
                # Outside domain 
                nNeigh += 1
            elseif kk < 1 || kk > Nz
                nNeigh += 0
            else
                fill!(LHS1,0.0)
                fill!(LHS2,0.0)
                fill!(dp,0.0)
                dp[ii,jj,kk] += delta
                J = ((A!(i,j,k,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env)
                    - A!(i,j,k,LHS2,uf,vf,wf,P.-dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env))
                    ./̂2delta)
                nNeigh +=1 
                row = n(ii,jj,kk,Ny,Nz)-max(0,offset[nNeigh]) # Row in diagonal array
                # println(n(ii,jj,kk,Ny,Nz))
                # println(max(0,offset[nNeigh])) 
                # @show i,j,ii,jj,nNeigh, row,n(ii,jj,kk,Ny,Nz),max(0,offset[nNeigh])
                diags[row,nNeigh] = J
            end
        end
    end
    J = spdiagm(
        offset[1] => diags[1:Nx*Ny*Nz-abs(offset[1]),1],
        offset[2] => diags[1:Nx*Ny*Nz-abs(offset[2]),2],
        offset[3] => diags[1:Nx*Ny*Nz-abs(offset[3]),3],
        offset[4] => diags[1:Nx*Ny*Nz-abs(offset[4]),4],
        offset[5] => diags[1:Nx*Ny*Nz-abs(offset[5]),5],
        offset[6] => diags[1:Nx*Ny*Nz-abs(offset[6]),6],
        offset[7] => diags[1:Nx*Ny*Nz-abs(offset[7]),7],
        offset[8] => diags[1:Nx*Ny*Nz-abs(offset[8]),8],
        offset[9] => diags[1:Nx*Ny*Nz-abs(offset[9]),9]
    )
    return J 
end


function compute_sparse3D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
    @unpack Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    diags = OffsetArray{Float64}(undef,1:Nx*Ny*Nz,27)
    dp = OffsetArray{Float64}(undef, imino_:imaxo_,jmino_:jmaxo_,kmino_:kmaxo_)
    LHS1 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    LHS2 = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)

    delta = 1.0
    fill!(diags,0.0)
    offset=[-Nx*Ny-Nx-1,-Nx*Ny-Nx,-Nx*Ny-Nx+1,-Nx*Ny-1,-Nx*Ny,-Nx*Ny+1,-Nx*Ny+Nx-1,
            -Nx*Ny+Nx,-Nx*Ny+Nx+1,-(Nx+1), -Nx, -(Nx-1), -1, 0, 1, Nx-1, Nx, Nx+1,
            Nx*Ny-Nx-1, Nx*Ny-Nx, Nx*Ny-Nx+1, Nx*Ny-1, Nx*Ny, Nx*Ny+1,Nx*Ny+Nx-1,
            Nx*Ny+Nx,Nx*Ny+Nx+1]


    @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        nNeigh = 0

        for kk = k-1:k+1 ,jj = j-1:j+1, ii = i-1:i+1
            if jj < 1 || jj > Nx || ii < 1 || ii > Nx || kk < 1 || kk > Nz
                # Outside domain 
                nNeigh += 1
            else
                fill!(LHS1,0.0)
                fill!(LHS2,0.0)
                fill!(dp,0.0)
                dp[ii,jj,kk] += delta
                J = ((A!(i,j,k,LHS1,uf,vf,wf,P.+dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env)
                    - A!(i,j,k,LHS2,uf,vf,wf,P.-dp,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env))
                    ./̂2delta)
                nNeigh +=1 
                row = n(ii,jj,kk,Ny,Nz)-max(0,offset[nNeigh]) # Row in diagonal array
                # println(n(ii,jj,kk,Ny,Nz))
                # println(max(0,offset[nNeigh])) 
                # @show i,j,ii,jj,nNeigh, row,n(ii,jj,kk,Ny,Nz),max(0,offset[nNeigh])
                diags[row,nNeigh] = J
            end
        end
    end
    J = spdiagm(
        offset[1] => diags[1:Nx*Ny*Nz-abs(offset[1]),1],
        offset[2] => diags[1:Nx*Ny*Nz-abs(offset[2]),2],
        offset[3] => diags[1:Nx*Ny*Nz-abs(offset[3]),3],
        offset[4] => diags[1:Nx*Ny*Nz-abs(offset[4]),4],
        offset[5] => diags[1:Nx*Ny*Nz-abs(offset[5]),5],
        offset[6] => diags[1:Nx*Ny*Nz-abs(offset[6]),6],
        offset[7] => diags[1:Nx*Ny*Nz-abs(offset[7]),7],
        offset[8] => diags[1:Nx*Ny*Nz-abs(offset[8]),8],
        offset[9] => diags[1:Nx*Ny*Nz-abs(offset[9]),9],
        offset[10] => diags[1:Nx*Ny*Nz-abs(offset[10]),10],
        offset[11] => diags[1:Nx*Ny*Nz-abs(offset[11]),11],
        offset[12] => diags[1:Nx*Ny*Nz-abs(offset[12]),12],
        offset[13] => diags[1:Nx*Ny*Nz-abs(offset[13]),13],
        offset[14] => diags[1:Nx*Ny*Nz-abs(offset[14]),14],
        offset[15] => diags[1:Nx*Ny*Nz-abs(offset[15]),15],
        offset[16] => diags[1:Nx*Ny*Nz-abs(offset[16]),16],
        offset[17] => diags[1:Nx*Ny*Nz-abs(offset[17]),17],
        offset[18] => diags[1:Nx*Ny*Nz-abs(offset[18]),18],
        offset[19] => diags[1:Nx*Ny*Nz-abs(offset[19]),19],
        offset[20] => diags[1:Nx*Ny*Nz-abs(offset[20]),20],
        offset[21] => diags[1:Nx*Ny*Nz-abs(offset[21]),21],
        offset[22] => diags[1:Nx*Ny*Nz-abs(offset[22]),22],
        offset[23] => diags[1:Nx*Ny*Nz-abs(offset[23]),23],
        offset[24] => diags[1:Nx*Ny*Nz-abs(offset[24]),24],
        offset[25] => diags[1:Nx*Ny*Nz-abs(offset[25]),25],
        offset[26] => diags[1:Nx*Ny*Nz-abs(offset[26]),26],
        offset[27] => diags[1:Nx*Ny*Nz-abs(offset[27]),27]
    )
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
# # non local A! matrix with delta
# function A!(P::Poisson{T},delta=0.0) where {T}
#     param,mesh = P.param,P.mesh
#     @unpack dx,dy,dz = mesh
#     d_step = dx,dy,dz

#     Neumann!(P.p,mesh,P.par)
#     update_borders!(P.p,mesh,P.par) # (overwrites BCs if periodic)

#     # p = copy(P.p)
#     # p[I] += delta
#     v_fields = (copy(P.f[1]), copy(P.f[2]), copy(P.f[3]))
#     for ii in 1:length(P.f)
#         for I in inside(P.f[ii])
#             v_fields[ii][I] -= P.step/P.den[ii][I]*del(ii,I,P.p)/d_step[ii]
#         end
#     end
#     # println("made it through")

#     for I in inside(P.band)
#         if abs(P.band[I]) <= 1
#             tets, inds = cell2tets_withProject_uvwf(I[1],I[2],I[3],v_fields[1],v_fields[2],v_fields[3],P.step,mesh)
#             if any(isnan,tets)
#                 error("Nan in tets at ", I)
#             end
#             v2 = dx*dy*dz
#             v1 = tets_vol(tets)
#             P.z[I] = (v2-v1) /̂ v2 /̂ P.step
#         else 
#             # Calculate divergence with finite differnce
#             du_dx = (v_fields[1][I+step(1,I)]-v_fields[1][I])/(dx)
#             dv_dy = (v_fields[2][I+step(2,I)]-v_fields[2][I])/(dy)
#             dw_dz = (v_fields[3][I+step(3,I)]-v_fields[3][I])/(dz)
#             P.z[I] = du_dx + dv_dy + dw_dz
#         end
#     end
# end

# local A matrix that recieves Poisson struct and cartesian index
# A! matrix for the jacobian calculation at P+/-dP acts on Jacobian residual field
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
        P.j[I] = (A_pos-A_neg)/̂2delta        
    end
end

function full_Jacobian!(P::Poisson)
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
        P.j[I] = (A_pos-A_neg)/̂2delta        
    end
end

function convert3d_1d(matrix)
    m1D = reshape(matrix,size(matrix,1)*size(matrix,2)*size(matrix,3),1)
    return m1D
end

function convert1d_3d(matrix,x,y,z)
    m3D = reshape(matrix,(x,y,z))
    return m3D
end

# Secant method
function Secant_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,outflow,param,mesh,par_env,step)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh

    AP = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    fill!(AP,0.0)
    outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env,step)

    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,param,mesh,par_env)

    # Iterate 
    iter=0
    while true
        iter += 1

        # compute jacobian
        J = computeJacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
        jacobian_2d = reshape(J, :, size(J, 3))
        cond_num = cond(jacobian_2d,2)
        println(cond_num)
        println(size(J))
        println(size(AP))
        println("determinant of J : ",det(Array(J)))
        P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] .-= 0.8AP./̂J

        P .-=mean(P)

        #Need to introduce outflow correction
        outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env,step)
        
        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,par_env,step)
        
        res = maximum(abs.(AP))
        if res < tol
            return iter
        end
        
        if iter % 1000 == 0
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res,sum(AP))
        end
    end    
end

function Secant_full_jacobian!(P,uf,vf,wf,gradx,grady,gradz,band,dt,denx,deny,denz,outflow,param,mesh,par_env,step)
    @unpack tol,Nx,Ny,Nz = param
    @unpack imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh
    @unpack dx,dy,dz = mesh

    AP = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    fill!(AP,0.0)
    outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env,step)

    A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env)

    # Iterate 
    iter=0
    while true
        iter += 1

        # compute jacobian
        J = compute_sparse3D_Jacobian(P,uf,vf,wf,gradx,grady,gradz,band,dt,param,denx,deny,denz,mesh,par_env)
        Pv = convert3d_1d(P[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
        APv = convert3d_1d(AP)
        # cond_num = cond(Array(J),2)
        # println(cond_num)
        # determ = det(J)
        # println("J determinant : ",determ)
        # println(size(J))
        # println(size(APv))
        # println(size(Pv))
        # error("stop")

        # #! attempt Tikhonov Regularization (Ridge Regression)
        # alpha = 100000.0
        # reg_term = alpha*I(size(J,1))
        # reg_J = J + reg_term
        # reg_determ = det(reg_J)
        # println("Regularized J determinant : ",determ)

        Pv -= J\APv
        
        P[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = convert1d_3d(Pv,Nx,Ny,Nz)

        P .-=mean(P)

        #Need to introduce outflow correction
        outflowCorrection!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,outflow,param,mesh,par_env,step)
        
        #update new Ap
        A!(AP,uf,vf,wf,P,dt,gradx,grady,gradz,band,denx,deny,denz,mesh,param,par_env)
        
        res = maximum(abs.(AP))
        if res < tol
            return iter
        end
        
        if iter % 1000 == 0
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",iter,res,sum(AP))
        end
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
function GaussSeidel!(P,RHS,uf,vf,wf,denx,deny,denz,dt,param,mesh,par_env)
    @unpack dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
    @unpack isroot = par_env
    @unpack tol = param
    maxIter=1000000
    iter = 0

    # println(vf[:, jmin_+1, :]) 
    # println(vf[:, jmax_-1, :])
    #apply outflow correction
    # outflowCorrection!(RHS,P,uf,vf,wf,denx,deny,denz,param,mesh,par_env)
    # @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
    #     # RHS
    #     RHS[i,j,k]=  ( 
    #         ( uf[i+1,j,k] - uf[i,j,k] )/(dx) +
    #         ( vf[i,j+1,k] - vf[i,j,k] )/(dy) +
    #         ( wf[i,j,k+1] - wf[i,j,k] )/(dz) )
    # end
    
    while true
        iter += 1
        max_update::Float64 = 0.0
        # for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
            # Pnew = (-denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*RHS[i,j,k]*dx^2*dy^2*dz^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*dt*dx^2*dy^2*P[i,j,k+1] + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k+1]*dt*dx^2*dy^2*P[i,j,k-1] + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*denz[i,j,k]*denz[i,j,k+1]*dt*dx^2*dz^2*P[i,j+1,k] + denx[i,j,k]*denx[i+1,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dt*dx^2*dz^2*P[i,j-1,k] + denx[i,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dt*dy^2*dz^2*P[i+1,j,k] + denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dt*dy^2*dz^2*P[i-1,j,k])/(dt*(denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*dx^2*dy^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k+1]*dx^2*dy^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2 + denx[i,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2 + denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2))
            Pnew = (-denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*RHS[i,j,k]*dx^2*dy^2*dz^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*dx^2*dy^2*P[i,j,k+1] + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k+1]*dx^2*dy^2*P[i,j,k-1] + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2*P[i,j+1,k] + denx[i,j,k]*denx[i+1,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2*P[i,j-1,k] + denx[i,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2*P[i+1,j,k] + denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2*P[i-1,j,k])/(denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*dx^2*dy^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k+1]*dx^2*dy^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2 + denx[i,j,k]*denx[i+1,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dx^2*dz^2 + denx[i,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2 + denx[i+1,j,k]*deny[i,j,k]*deny[i,j+1,k]*denz[i,j,k]*denz[i,j,k+1]*dy^2*dz^2)
            # # println(Pnew)
            max_update=max(max_update,abs(Pnew-P[i,j,k]))
            P[i,j,k] = Pnew 
        end

        update_borders!(P,mesh,par_env)
        Neumann!(P,mesh,par_env)
        # if iter % 1000 == 0
        #     println("Max updat = ",max_update)
        # end    
        max_update = parallel_max_all(max_update,par_env)

        max_update < tol && return iter # Converged
        # Check if hit max iteration
        if iter == maxIter 
            isroot && println("Failed to converged Poisson equation max_upate = $max_update")
            return iter
        end
    end
end


function n(i,j,k,Ny,Nz) 
    val = i + (j-1)*Ny + (k-1)*Nz*Ny
    # @show i,j,k,Ny,Nz,val
    return val
end 

"""
 GaussSeidel Poisson Solver( update)
"""
# function GaussSeidel!(P,RHS,uf,vf,wf,denx,deny,denz,dt,param,mesh,par_env)
#     @unpack Nx,Ny,Nz,dx,dy,dz,imin_,imax_,jmin_,jmax_,kmin_,kmax_ = mesh
#     @unpack isroot = par_env
#     @unpack tol = param
#     maxIter=100
#     iter = 0

#     # println(vf[:, jmin_+1, :]) 
#     # println(vf[:, jmax_-1, :])
#     #apply outflow correction
#     # outflowCorrection!(RHS,P,uf,vf,wf,denx,deny,denz,param,mesh,par_env)
#     @loop param for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
#         # RHS
#         RHS[i,j,k]=  ( 
#             ( uf[i+1,j,k] - uf[i,j,k] )/(dx) +
#             ( vf[i,j+1,k] - vf[i,j,k] )/(dy) +
#             ( wf[i,j,k+1] - wf[i,j,k] )/(dz) )
#     end

#     A_lap = OffsetArray{Float64}(undef,1:Nx*Ny*Nz,1:Nx*Ny*Nz)
#     LHS = OffsetArray{Float64}(undef, 1:Nx,1:Ny,1:Nz)
#     fill!(A_lap,0.0)
    
#     for k = kmin_:kmax_,j = jmin_:jmax_,i = imin_:imax_
#         Pi = zeros(0:Nx+1,0:Ny+1,0:Nz+1)
#         Pi[i,j,k] += 1.0
#         for kk = kmin_:kmax_, jj = jmin_:jmax_, ii = imin_:imax_
#             A_lap[n(i,j,k,Ny,Nz),n(ii,jj,kk,Ny,Nz)] = (
#                 (Pi[ii-1,jj,kk] - 2Pi[ii,jj,kk] + Pi[ii+1,jj,kk]) /̂ dx^2 +
#                 (Pi[ii,jj-1,kk] - 2Pi[ii,jj,kk] + Pi[ii,jj+1,kk]) /̂ dy^2 +
#                 (Pi[ii,jj,kk-1] - 2Pi[ii,jj,kk] + Pi[ii,jj,kk+1]) /̂ dz^2 )
#         end
#     end


#     while true
#         iter += 1
#         max_update::Float64 = 0.0
#         P_old = P
#         lap!(LHS,P,param,mesh)
#         for i= 1:Nx*Ny*Nz
#             P[i] = (RHS[i] - LHS[i] + A_lap[i,i]*P[i])/A_lap[i,i]
#         end
#         if norm(P-P_old)/norm(P) > tol
#             return iter
#         end
#         # for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
#         #     res_factor = (-denx[i,j,k] * denx[i+1,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k] * denz[i,j,k+1] * RHS[i,j,k] * dx^2 * dy^2 * dz^2)
#         #     Pk_pos = (denx[i,j,k] * denx[i+1,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k] * dt * dx^2 * dy^2)
#         #     Pk_neg = (denx[i,j,k] * denx[i+1,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k+1] * dt * dx^2 * dy^2)
#         #     Pj_pos = (denx[i,j,k] * denx[i+1,j,k] * deny[i,j,k] * denz[i,j,k] * denz[i,j,k+1] * dt * dx^2 * dz^2)
#         #     Pj_neg = (denx[i,j,k] * denx[i+1,j,k] * deny[i,j+1,k] * denz[i,j,k] * denz[i,j,k+1] * dt * dx^2 * dz^2)
#         #     Pi_pos = (denx[i,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k] * denz[i,j,k+1] * dt * dy^2 * dz^2)
#         #     Pi_neg = (denx[i+1,j,k] * deny[i,j,k] * deny[i,j+1,k] * denz[i,j,k] * denz[i,j,k+1] * dt * dy^2 * dz^2)
#         #     Pnew = (res_factor + Pk_pos*P[i,j,k+1] + Pk_neg*P[i,j,k-1] + Pj_pos*P[i,j+1,k] + Pj_neg*P[i,j-1,k] + Pi_pos*P[i+1,j,k] + Pi_neg*P[i-1,j,k])/̂
#         #             dt*(Pk_pos + Pk_neg + Pj_pos + Pj_neg + Pi_pos + Pi_neg)
#         #     # println(Pnew)
#         #     max_update=max(max_update,abs(Pnew-P[i,j,k]))
#         #     P[i,j,k] = Pnew 
#         # end
#         # # error("stop")
#         update_borders!(P,mesh,par_env)
#         Neumann!(P,mesh,par_env)
#         # println("Max update = ",max_update)
#         # max_update = parallel_max_all(max_update,par_env)

#         # max_update < tol && return iter # Converged
#         # Check if hit max iteration
#         if iter == maxIter 
#             isroot && println("Failed to converged Poisson equation")
#             return iter
#         end
#     end
# end

"""
Conjugate gradient
"""
function conjgrad!(P,RHS,denx,deny,denz,param,mesh,par_env)
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

    lap!(r,P,denx,deny,denz,param,mesh)
    r[ix,iy,iz] = RHS.parent - r[ix,iy,iz]
    Neumann!(r,mesh,par_env)
    update_borders!(r,mesh,par_env) # (overwrites BCs if periodic)
    p .= r
    rsold = parallel_sum_all(r[ix,iy,iz].^2,par_env)
    rsnew = 0.0
    for iter = 1:10
        lap!(Ap,p,denx,deny,denz,param,mesh)

        sum = parallel_sum_all(p[ix,iy,iz].*Ap[ix,iy,iz],par_env)
        alpha = rsold /̂ sum
        P .+= alpha*p
        r -= alpha * Ap
        rsnew = parallel_sum_all(r[ix,iy,iz].^2,par_env)
        if sqrt(rsnew) < tol
            return iter
        end
        p = r + (rsnew /̂ rsold) * p
        Neumann!(p,mesh,par_env)   
        update_borders!(p,mesh,par_env) # (overwrites BCs if periodic)
        rsold = rsnew

    end
    
    isroot && println("Failed to converged Poisson equation rsnew = $rsnew")
    
    return length(RHS)
end


function Jacobi!(P::Poisson,tol=1e-4)
    p,z,j,n = P.p,P.z,P.j,P.n
    #!apply outflow correction to P.p calc A! using P.z

    # calc A(P) as p.z
    A!(P)

    n = 0
    while true 
        n +=1
        # calc jacobian
        Jacobian!(P)

        for I in inside(p)
            P.p[I] -= 0.8z[I]/j[I]
        end

        # avoid drift
        p .-=mean(p)

        #!add outlfow correction to P.p

        # calc A(P)
        fill!(z,0.0)
        A!(P)

        # check residual to tol for new A(P)
        res = maximum(abs.(z))
     
        if res < tol
            return n
        end
        if n % 1000 == 0
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",n,res,sum(z))
        end  
    end
end

function guided_Jacobi!(P::Poisson,tol=1e-4)
    p,z,e,n = P.p,P.z,P.e,P.n
    # calc A(P) as p.z
    A!(P)

    n = 0
    alpha = similar(p)
    fill!(alpha,0.0)
    while true 
        n +=1
        # calc jacobian
        Jacobian!(P)

        #! compute alpha at each point in the domain

        for I in inside(p)
            alpha[I] = arm_gold(P,I)
        end
        # println(alpha)
        for I in inside(p)
            P.p[I] -= alpha[I]*z[I]/e[I]
        end

        # avoid drift
        p .-=mean(p)

        # calc A(P)
        fill!(z,0.0)
        A!(P)

        # check residual to tol for new A(P)
        res = maximum(abs.(z))
     
        if res < tol
            return n
        end
        if n % 500 == 0
            println(alpha)
            @printf("Iter = %4i  Res = %12.3g  sum(divg) = %12.3g \n",n,res,sum(z))
        end  
    end
end

#! need to think about how to construct the domain wide line search for the armijo-goldstein condition
function arm_gold(P::Poisson, I::CartesianIndex,del = 0.5, c=1000)
    f,x,d = P.z,P.p,P.e
    g = copy(d)
    f1 = copy(f)
    t = 1.0
    delta = -t*d
    A!(P,I,delta[I])
    while f[I] < f1[I] + c*t*g[I]^2
        t *= del
        delta = t*d
        A!(P,I,delta[I])

    end
    return t
end