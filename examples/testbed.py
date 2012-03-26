from mpi4py import MPI
import elemental

ctx = elemental.Context()

comm = MPI.COMM_WORLD
print comm.Get_rank()

import time
time.sleep(1)

grid = elemental.Grid(ctx, comm, 2, 2)
A = elemental.DistMatrix(grid, 10, 10)
A.dump()
#1/0
B = elemental.DistMatrix(grid, 10, 10)
C = elemental.DistMatrix(grid, 10, 10)

#A.set_to_identity()
B.set_to_identity()
C.set_to_identity()

elemental.gemm('N', 'N', 1, A, B, 1, C)

#print A.local_buf
#print B.local_buf
C.dump()

#for i in range(comm.Get_size()):
#    if i == comm.Get_rank():
#        print A.local_buf
#    comm.Barrier()

#C.dump()
