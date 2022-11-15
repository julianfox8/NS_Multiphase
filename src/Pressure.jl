
# Serial multigrid Poisson Solver
function pressure_solver!(P,param,mesh,par_env)

    @unpack xm,ym,zm,imin_,imax_,jmin_,jmax_,kmin_,kmax_,imino_,imaxo_,jmino_,jmaxo_,kmino_,kmaxo_ = mesh

    RHS = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_,kmin_:kmax_)
    sig=0.1
    for k=kmin_:kmax_, j=jmin_:jmax_, i=imin_:imax_
        RHS[i,j,k]=(
            +1.0/(sig*sqrt(2.0*pi))*exp(-((xm[i]-0.5)^2+(ym[j]-0.5)^2)/(2.0*sig^2))
            -1.0/(sig*sqrt(2.0*pi))*exp(-((xm[i]-2.5)^2+(ym[j]-0.5)^2)/(2.0*sig^2)))
    end

    poisson_solve!(P,RHS,param,mesh,par_env)

    return nothing
end
