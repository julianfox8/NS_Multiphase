# export JULIA_NUM_THREADS=2

function nestedloops1(nx, ny, nz)

   state = ones(nx,ny,nz)

   for k = 1:nz
      for j = 1:ny
         for i = 1:nx
            state[i,j,k] *= sin(i*j*k)
         end
      end
   end

   #println(state[2,2,2])

   return
end


function nestedloops2(nx, ny, nz)

   state = ones(nx,ny,nz)

   @inbounds for k = 1:nz
      @inbounds for j = 1:ny
         @inbounds for i = 1:nx
            state[i,j,k] *= sin(i*j*k)
         end
      end
   end

   #println(state[2,2,2])

   return
end

function nestedloops3(nx, ny, nz)

   state = ones(nx,ny,nz)

   Threads.@threads for k = 1:nz
      for j = 1:ny
         for i = 1:nx
            state[i,j,k] *= sin(i*j*k)
         end
      end
   end

   #println(state[2,2,2])

   return

end

function nestedloops3_2(nx, ny, nz)

    state = ones(nx,ny,nz)
 
    Threads.@threads for ind in CartesianIndices(state)
        i,j,k = ind[1],ind[2],ind[3]
        state[i,j,k] *= sin(i*j*k)
    end
 
    #println(state[2,2,2])
 
    return
 
 end

function nestedloops4(nx, ny, nz)

   state = ones(nx,ny,nz)

   @sync Threads.@spawn for k = 1:nz
      for j = 1:ny
         for i = 1:nx
            state[i,j,k] *= sin(i*j*k)
         end
      end
   end

   #println(state[2,2,2])

   return

end

function nestedloops5(nx, ny, nz)

   state = ones(nx,ny,nz)

   for k = 1:nz, j = 1:ny, i = 1:nx
      state[i,j,k] *= sin(i*j*k)
   end

   #println(state[2,2,2])

   return

end

function nestedloops6(nx, ny, nz)

   state = ones(nx,ny,nz)

   Threads.@threads for k = 1:nz
      Threads.@threads for j = 1:ny
         Threads.@threads for i = 1:nx
            state[i,j,k] *= sin(i*j*k)
         end
      end
   end

   #println(state[2,2,2])

   return

end

function nestedloops7(nx, ny, nz)

   state = ones(nx,ny,nz)

   @sync Threads.@spawn for k = 1:nz
      @sync Threads.@spawn for j = 1:ny
         @sync Threads.@spawn for i = 1:nx
            state[i,j,k] *= sin(i*j*k)
         end
      end
   end

   #println(state[2,2,2])

   return

end

function nestedloops8(nx, ny, nz)

   state = ones(nx,ny,nz)

   @sync Threads.@spawn for k = 1:nz, j = 1:ny, i = 1:nx
      state[i,j,k] *= sin(i*j*k)
   end

   #println(state[2,2,2])

   return

end



##
nx, ny, nk = 200, 200, 200
nestedloops1(nx, ny, nk)
nestedloops2(nx, ny, nk)
nestedloops3(nx, ny, nk)
nestedloops4(nx, ny, nk)
nestedloops5(nx, ny, nk)
nestedloops6(nx, ny, nk)
nestedloops7(nx, ny, nk)
nestedloops8(nx, ny, nk)

println("Number of threads = ",Threads.nthreads())
println("base line:")
@btime nestedloops1(nx, ny, nk)
println("explicit @inbound:")
@btime nestedloops2(nx, ny, nk)
println("@threads on the outer loop:")
@btime nestedloops3(nx, ny, nk)
@btime nestedloops3_2(nx, ny, nk)
println("@spawn on the outer loop:")
@btime nestedloops4(nx, ny, nk)
println("nested loop:")
@btime nestedloops5(nx, ny, nk)
println("@threads on the triple loops:")
@btime nestedloops6(nx, ny, nk)
println("@spawn on the triple loops:")
@btime nestedloops7(nx, ny, nk)
println("@spawn on the nested loops:")
@btime nestedloops8(nx, ny, nk)