"""
"""

cimport libelemental as lib
from mpi4py cimport MPI
import mpi4py

cdef class Grid:
    cdef lib.Grid *wrapped

    def __init__(self, MPI.Comm comm, width, height):
        self.wrapped = new lib.Grid(comm.ob_mpi, width, height)

    def __dealloc__(self):
        if self.wrapped != NULL:
            del self.wrapped
            self.wrapped = NULL

cdef class DistMatrixBase:
    pass

cdef class DistMatrix_d_MC_MR(DistMatrixBase):
    cdef lib.DistMatrix[double, lib.MC, lib.MR, int] *wrapped

    def __init__(self, height, width, Grid grid):
        self.wrapped = new lib.DistMatrix[double, lib.MC, lib.MR, int](
            height, width, grid.wrapped[0])

    def __dealloc__(self):
        if self.wrapped != NULL:
            del self.wrapped
            self.wrapped = NULL

    def set_to_identity(self):
        self.wrapped.SetToIdentity()

    def dump(self):
        self.wrapped.Print()


def initialize():
    cdef int argc = 1
    cdef char **argv = NULL
    lib.Initialize(argc, argv)

def finalize():
    lib.Finalize()

def gemm(DistMatrix_d_MC_MR A, DistMatrix_d_MC_MR B, DistMatrix_d_MC_MR C):
    lib.Gemm(lib.NORMAL, lib.NORMAL, 2.0, A.wrapped[0],
             B.wrapped[0], 3.0, C.wrapped[0])    
