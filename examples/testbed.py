from mpi4py import MPI
from elemental import *

comm = MPI.COMM_WORLD
print comm.Get_rank()

import time
time.sleep(1)

grid = Grid(comm, 2, 2)
A = DistMatrix(10, 10, grid)
A.dump()
#1/0
B = DistMatrix(10, 10, grid)
C = DistMatrix(10, 10, grid)

#A.set_to_identity()
B.set_to_identity()
C.set_to_identity()

gemm('N', 'N', 1, A, B, 1, C)

#print A.local_buf
#print B.local_buf
C.dump()

#for i in range(comm.Get_size()):
#    if i == comm.Get_rank():
#        print A.local_buf
#    comm.Barrier()

#C.dump()
