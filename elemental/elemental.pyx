"""
"""

# cimports
cimport numpy as np
cimport libelemental as lib
from mpi4py cimport MPI

# imports
import numpy as np
import mpi4py

class ElementalError(RuntimeError):
    pass

cdef object elemental_error():
    raise ElementalError(lib.elem_errmsg())

cdef object check(int retcode):
    if retcode == lib.ELEM_ERROR_OK:
        return
    elif retcode == lib.ELEM_ERROR_LOGIC:
        raise ValueError('elemental: %s' % lib.elem_errmsg())
    else:
        elemental_error()

cdef lib.elem_complex as_complex(object obj):
    cdef lib.elem_complex z
    z.real = obj.real
    z.imag = obj.imag
    return z

cdef int as_orientation(object obj):
    if obj == 'N':
        return lib.ELEM_NORMAL
    elif obj == 'T':
        return lib.ELEM_TRANSPOSE
    elif obj == 'C':
        return lib.ELEM_ADJOINT
    else:
        raise ValueError('orientation must be one of "N", "T", "C"')

cdef class Grid:
    cdef lib.elem_grid *wrapped

    def __init__(self, MPI.Comm comm, width, height):
        self.wrapped = lib.elem_create_grid(comm.ob_mpi, width, height)
        if self.wrapped == NULL:
            elemental_error()

    def __dealloc__(self):
        if self.wrapped != NULL:
            lib.elem_destroy_grid(self.wrapped)
            self.wrapped = NULL

    cpdef mcrank(self):
        return lib.elem_grid_mcrank(self.wrapped)

    cpdef mrrank(self):
        return lib.elem_grid_mrrank(self.wrapped)

    cpdef width(self):
        return lib.elem_grid_width(self.wrapped)

    cpdef height(self):
        return lib.elem_grid_height(self.wrapped)


_distribution_to_code = {
    'MC' : lib.ELEM_MC,
    'MD' : lib.ELEM_MD,
    'MR' : lib.ELEM_MR,
    'VC' : lib.ELEM_VC,
    'VR' : lib.ELEM_VR,
    'STAR' : lib.ELEM_STAR,
}

cdef class DistMatrix:
    cdef lib.elem_matrix *wrapped
    cdef readonly object local_buf

    def __init__(self, height, width, Grid grid, dtype=np.double,
                 coldist='MC', rowdist='MR'):
        cdef int local_height, local_width
        cdef int colAlignment = 0, rowAlignment = 0
        local_height = lib.elem_local_length(height, grid.mcrank(),
                                             colAlignment, grid.height())
        local_width = lib.elem_local_length(width, grid.mrrank(),
                                            rowAlignment, grid.width())

        cdef np.ndarray[double, ndim=2] local_buf = (
            np.arange(local_height * local_width, dtype=np.double).reshape(
            (local_height, local_width), order='F'))
        self.local_buf = local_buf


        self.wrapped = lib.elem_create_matrix(lib.ELEM_DOUBLE_REAL,
                                              _distribution_to_code[coldist],
                                              _distribution_to_code[rowdist],
                                              height, width,
                                              colAlignment, rowAlignment,
                                              <double*>np.PyArray_DATA(local_buf),
                                              local_height,
                                              grid.wrapped)
        if self.wrapped == NULL:
            elemental_error()

    def __dealloc__(self):
        if self.wrapped != NULL:
            lib.elem_destroy_matrix(self.wrapped)
            self.wrapped = NULL

    def set_to_identity(self):
        check(lib.elem_set_to_identity(self.wrapped))

    def dump(self):
        check(lib.elem_print(self.wrapped))

def gemm(orientation_A, orientation_B, alpha,
         DistMatrix A, DistMatrix B, beta, DistMatrix C):
    lib.elem_gemm(as_orientation(orientation_A),
                  as_orientation(orientation_B),
                  as_complex(alpha),
                  A.wrapped,
                  B.wrapped,
                  as_complex(beta),
                  C.wrapped)

cdef void _initialize():
    cdef int argc = 1
    cdef char **argv = NULL
    lib.elem_initialize(&argc, argv)

cdef void _finalize():
    lib.elem_finalize()

_initialize()
cdef extern from "stdlib.h":
    int atexit(void (*func)())
atexit(&_finalize)


