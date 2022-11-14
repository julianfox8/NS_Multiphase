
# Serial multigrid Poisson Solver
function pressure_solver!(P,mesh,par_env)

    @unpack xm,ym,imin_,imax_,jmin_,jmax_,imino_,imaxo_,jmino_,jmaxo_ = mesh

    RHS = OffsetArray{Float64}(undef, imin_:imax_,jmin_:jmax_)
    sig=0.1
    for j=jmin_:jmax_
        for i=imin_:imax_
            RHS[i,j]=(
                +1.0/(sig*sqrt(2.0*pi))*exp(-((xm[i]-0.5)^2+(ym[j]-0.5)^2)/(2.0*sig^2))
                -1.0/(sig*sqrt(2.0*pi))*exp(-((xm[i]-2.5)^2+(ym[j]-0.5)^2)/(2.0*sig^2)))
        end
    end


    poisson_solve!(P,RHS,mesh,par_env)

    return nothing
end
