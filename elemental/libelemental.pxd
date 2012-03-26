from mpi4py.mpi_c cimport MPI_Comm, MPI_COMM_WORLD

cdef extern from "elemental_wrapper.h":

    cdef enum:
        ELEM_SINGLE_REAL
        ELEM_SINGLE_COMPLEX
        ELEM_DOUBLE_REAL
        ELEM_DOUBLE_COMPLEX

        ELEM_MC
        ELEM_MD
        ELEM_MR
        ELEM_VC
        ELEM_VR
        ELEM_STAR

        ELEM_NORMAL
        ELEM_TRANSPOSE
        ELEM_ADJOINT

        ELEM_ERROR_LOGIC
        ELEM_ERROR_RUNTIME

        ELEM_UPPER
        ELEM_LOWER

    ctypedef struct elem_context:
        pass

    ctypedef struct elem_complex:
        double real, imag

    ctypedef struct elem_grid:
        pass
    
    ctypedef struct elem_matrix:
        int dtype
        int col_dist
        int row_dist

    # global
    elem_context *elem_create_context()
    void elem_destroy_context(elem_context *ctx)
    int elem_initialize(elem_context *ctx)
    int elem_finalize(elem_context *ctx)
    void elem_lasterror(elem_context *ctx, char **errmsg, int *errtype)

    # matrix/matrix ops
    elem_matrix *elem_create_matrix(elem_grid *grid,
                                    int dtype, int col_dist, int row_dist,
                                    int height, int width, int col_alignment,
                                    int row_alignment, void *buffer,
                                    int lda)
    int elem_destroy_matrix(elem_matrix *matrix)


    int elem_gemm(int orientation_of_a,
                  int orientation_of_b,
                  elem_complex alpha,
                  elem_matrix *A,
                  elem_matrix *B,
                  elem_complex beta,
                  elem_matrix *C)

    int elem_cholesky(int uplo, elem_matrix *A)

    int elem_print(elem_matrix *matrix)
    int elem_set_to_identity(elem_matrix *matrix)

    # grid
    elem_grid *elem_create_grid(elem_context *ctx, MPI_Comm comm,
                                int height, int width)
    int elem_destroy_grid(elem_grid *grid)
    int elem_grid_mcrank(elem_grid *grid)
    int elem_grid_mrrank(elem_grid *grid)
    int elem_grid_height(elem_grid *grid)
    int elem_grid_width(elem_grid *grid)
    
    # stateless non-error utils
    int elem_local_length(int n, int index, int alignment, int modulus)
    
