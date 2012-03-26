"""
"""

# cimports
cimport numpy as np
from libelemental cimport *
from mpi4py cimport MPI

# imports
import numpy as np
import mpi4py

cdef check(Context ctx, int retval):
    if retval != 0:
        ctx.raise_error()

class ElementalValueError(ValueError):
    pass

class ElementalRuntimeError(RuntimeError):
    pass

cdef class Context:
    cdef elem_context *wrapped

    def __init__(self):
        self.wrapped = elem_create_context()
        check(self, elem_initialize(self.wrapped))

    def __dealloc__(self):
        if self.wrapped != NULL:
            check(self, elem_finalize(self.wrapped))
            elem_destroy_context(self.wrapped)
            self.wrapped = NULL

    def raise_error(self):
        cdef char *msg
        cdef int errtype
        elem_lasterror(self.wrapped, &msg, &errtype)
        if errtype == ELEM_ERROR_LOGIC:
            raise ElementalValueError((<bytes>msg).decode('ASCII'))
        elif errtype == ELEM_ERROR_RUNTIME:
            raise ElementalRuntimeError((<bytes>msg).decode('ASCII'))
        else:
            raise AssertionError()

cdef elem_complex as_complex(object obj):
    cdef elem_complex z
    z.real = obj.real
    z.imag = obj.imag
    return z

cdef int as_uplo(object obj):
    if obj == 'L':
        return ELEM_LOWER
    elif obj == 'U':
        return ELEM_UPPER
    else:
        raise ValueError('upper/lower must be one of "U", "L"')

cdef int as_orientation(object obj):
    if obj == 'N':
        return ELEM_NORMAL
    elif obj == 'T':
        return ELEM_TRANSPOSE
    elif obj == 'C':
        return ELEM_ADJOINT
    else:
        raise ValueError('orientation must be one of "N", "T", "C"')

cdef int as_type_enum(object obj):
    if obj == np.float64:
        return ELEM_DOUBLE_REAL
    elif obj == np.complex128:
        return ELEM_DOUBLE_COMPLEX
    elif obj == np.float32:
        return ELEM_SINGLE_REAL
    elif obj == np.complex64:
        return ELEM_SINGLE_COMPLEX
    else:
        raise ValueError('unsupported dtype: %r' % obj)

cdef class Grid:
    cdef elem_grid *wrapped
    cdef Context context

    def __init__(self, Context context, MPI.Comm comm, width, height):
        self.context = context
        self.wrapped = elem_create_grid(context.wrapped, comm.ob_mpi,
                                        width, height)
        if self.wrapped == NULL:
            context.raise_error()

    def __dealloc__(self):
        if self.wrapped != NULL:
            elem_destroy_grid(self.wrapped)
            self.wrapped = NULL

    cpdef mcrank(self):
        return elem_grid_mcrank(self.wrapped)

    cpdef mrrank(self):
        return elem_grid_mrrank(self.wrapped)

    cpdef width(self):
        return elem_grid_width(self.wrapped)

    cpdef height(self):
        return elem_grid_height(self.wrapped)


_distribution_to_code = {
    'MC' : ELEM_MC,
    'MD' : ELEM_MD,
    'MR' : ELEM_MR,
    'VC' : ELEM_VC,
    'VR' : ELEM_VR,
    'STAR' : ELEM_STAR,
}

cdef class DistMatrix:
    cdef elem_matrix *wrapped
    cdef readonly object local
    cdef Context context

    def __init__(self, Grid grid,
                 height, width, dtype=np.double,
                 coldist='MC', rowdist='MR'):
        cdef:
            int local_height, local_width, col_alignment, row_alignment
            np.ndarray local_buf

        col_alignment = 0
        row_alignment = 0
        local_height = elem_local_length(height, grid.mcrank(),
                                         col_alignment, grid.height())
        local_width = elem_local_length(width, grid.mrrank(),
                                        row_alignment, grid.width())

        local_buf = np.zeros((local_height, local_width),
                             dtype=dtype,
                             order='F')

        self.context = grid.context
        self.local = local_buf

        self.wrapped = elem_create_matrix(grid.wrapped,
                                          as_type_enum(dtype),
                                          _distribution_to_code[coldist],
                                          _distribution_to_code[rowdist],
                                          height, width,
                                          col_alignment, row_alignment,
                                          np.PyArray_DATA(local_buf),
                                          local_height)
        if self.wrapped == NULL:
            self.context.raise_error()

    def __dealloc__(self):
        if self.wrapped != NULL:
            check(self.context, elem_destroy_matrix(self.wrapped))
            self.wrapped = NULL

    def set_to_identity(self):
        check(self.context, elem_set_to_identity(self.wrapped))

    def dump(self):
        check(self.context, elem_print(self.wrapped))

def gemm(orientation_A, orientation_B, alpha,
         DistMatrix A, DistMatrix B, beta, DistMatrix C):
    check(A.context, elem_gemm(as_orientation(orientation_A),
                               as_orientation(orientation_B),
                               as_complex(alpha),
                               A.wrapped,
                               B.wrapped,
                               as_complex(beta),
                               C.wrapped))

def cholesky(uplo, DistMatrix A):
    check(A.context, elem_cholesky(as_uplo(uplo), A.wrapped))

## cdef void _initialize():
##     cdef int argc = 1
##     cdef char **argv = NULL
##     elem_initialize(&argc, argv)

## cdef void _finalize():
##     elem_finalize()

## _initialize()
## cdef extern from "Python.h":
##     int Py_AtExit(void (*func)()) except -1

## Py_AtExit(&_finalize)


