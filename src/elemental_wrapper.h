#ifndef _POLYMORPHIC_ELEMENTAL_H_
#define _POLYMORPHIC_ELEMENTAL_H_

#include <mpi.h>

typedef struct {
  double real, imag;
} elem_complex;

#ifdef __cplusplus
extern "C" {
#endif


#define ELEM_SINGLE_REAL 0
#define ELEM_SINGLE_COMPLEX 1
#define ELEM_DOUBLE_REAL 2
#define ELEM_DOUBLE_COMPLEX 3

#define ELEM_ERROR_LOGIC 1
#define ELEM_ERROR_RUNTIME 2

#define ELEM_MC 0
#define ELEM_MD 1
#define ELEM_MR 2
#define ELEM_VC 3
#define ELEM_VR 4
#define ELEM_STAR 5

#define ELEM_NORMAL 0
#define ELEM_TRANSPOSE 1
#define ELEM_ADJOINT 2

#define ELEM_LOWER 0
#define ELEM_UPPER 1

typedef struct {
} elem_context;


typedef struct {
  elem_context *ctx;
  void *cpp_obj;
} elem_grid;

/* Instances of elem_create_matrix must be constructed with elem_create
   and freed with elem_destroy. The struct is larger than the fields
   exposed here. */
typedef struct {
  int dtype;
  int col_dist;
  int row_dist;
  elem_grid *grid;
  void *cpp_obj;
} elem_matrix;

/* stateless utilities */
int elem_local_length(int n, int index, int alignment, int modulus);

/* global environment, utilities */
elem_context *elem_create_context();
void elem_destroy_context(elem_context *ctx);

int elem_initialize(elem_context *ctx);
int elem_finalize(elem_context *ctx);
void elem_lasterror(elem_context *ctx, char **errmsg, int *errtype);

/* matrix/matrix ops */

elem_matrix *elem_create_matrix(elem_grid *grid,
                                int dtype, int col_dist, int row_dist,
                                int height, int width, int col_alignment,
                                int row_alignment, void *buffer,
                                int lda);
int elem_destroy_matrix(elem_matrix *matrix);

int elem_print(elem_matrix *matrix);
int elem_set_to_identity(elem_matrix *matrix);


int elem_gemm(int orientation_of_a,
              int orientation_of_b,
              elem_complex alpha,
              elem_matrix *A,
              elem_matrix *B,
              elem_complex beta,
              elem_matrix *C);

int elem_cholesky(int uplo, elem_matrix *A);



/* grid */
elem_grid *elem_create_grid(elem_context *ctx, MPI_Comm comm, int height, int width);
int elem_destroy_grid(elem_grid *grid);
int elem_grid_mcrank(elem_grid *grid);
int elem_grid_mrrank(elem_grid *grid);
int elem_grid_height(elem_grid *grid);
int elem_grid_width(elem_grid *grid);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
