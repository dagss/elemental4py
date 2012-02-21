from mpi4py.mpi_c cimport MPI_Comm, MPI_COMM_WORLD

cdef extern from "elemental.hpp" namespace "::elemental":


    void Initialize(int argc, char **argv)
    void Finalize()

    cdef cppclass Grid:
        Grid()
        Grid(MPI_Comm comm)
        Grid(MPI_Comm comm, int height, int width)

    cdef enum Orientation:
        NORMAL,
        TRANSPOSE,
        ADJOINT

    # The distributions are really an enum and not types; however
    # Cython only allows type template arguments
    ctypedef int MC
    ctypedef int MD
    ctypedef int MR
    ctypedef int VC
    ctypedef int VR
    ctypedef int STAR

    cdef cppclass DistMatrix[T, ColDist, RowDist, Int]:
        DistMatrix(Int height, Int width, Grid g)
        DistMatrix(Int height, Int width, Int colAlignment, Int rowAlignment,
                   T* buffer, Int ldim, Grid g)

        void SetToIdentity()
        void Print()

    cdef void Gemm(Orientation orientationOfA,
                   Orientation orientationOfB,
                   double alpha,
                   DistMatrix[double, MC, MR, int] A,
                   DistMatrix[double, MC, MR, int] B,
                   double beta,
                   DistMatrix[double, MC, MR, int] C)
