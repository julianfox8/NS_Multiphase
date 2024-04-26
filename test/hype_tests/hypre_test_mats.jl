using HYPRE.LibHYPRE
using MPI

# MPI.Init()
# if ~MPI.Initialized()
    MPI.Init()
# end

comm = MPI.COMM_WORLD

nproc = MPI.Comm_size(comm)




A_ref = Ref{HYPRE_IJMatrix}(C_NULL)
# HYPRE_IJMatrixCreate(comm,temp_min,temp_max,temp_min,temp_max,A_ref)
HYPRE_IJMatrixCreate(comm,1,5,1,5,A_ref)
# HYPRE_IJMatrixCreate(comm,temp_min,temp_max,temp_min,temp_max,A_ref)
A = A_ref[]

HYPRE_IJMatrixSetObjectType(A,HYPRE_PARCSR)

HYPRE_IJMatrixInitialize(A)


HYPRE_IJMatrixSetValues(A,1,pointer(Int32.([5])),pointer(Int32.([1])),pointer(Int32.([5,4,3,2,1])),pointer(Float64.([1.0,2.0,3.0,4.0,5.0])))
HYPRE_IJMatrixSetValues(A,1,pointer(Int32.([2])),pointer(Int32.([2])),pointer(Int32.([2,3])),pointer(Float64.([1.0,3.0])))
# HYPRE_IJMatrixSetValues(A,1,pointer([3]),pointer([3]),pointer([3,5,0,0,0]),pointer([3.0,0.0,1.0,0.0,0.0]))
# HYPRE_IJMatrixSetValues(A,1,pointer([4]),pointer([4]),pointer([3,5,1,0,0]),pointer([3.0,1.0,4.0,0.0,0.0]))
# HYPRE_IJMatrixSetValues(A,1,pointer([1]),pointer([5]),pointer([3,2,0,0,0]),pointer([3.0,1.0,0.0,0.0,0.0]))


HYPRE_IJMatrixAssemble(A)

HYPRE_IJMatrixPrint(A,"A")


