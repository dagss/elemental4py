from mpi4py import MPI
from elemental import *

comm = MPI.COMM_WORLD
print comm.Get_rank()

import time
time.sleep(1)

initialize()

grid = Grid(comm, 2, 2)
A = DistMatrix_d_MC_MR(10, 10, grid)
B = DistMatrix_d_MC_MR(10, 10, grid)
C = DistMatrix_d_MC_MR(10, 10, grid)

A.set_to_identity()
B.set_to_identity()
C.set_to_identity()

gemm(A, B, C)

C.dump()
