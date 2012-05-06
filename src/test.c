#include "elemental_wrapper.h"

/* TODO: Error checking

After almost every operation, one can check the return code
and if there's a failure, call elem_lasterror to extract
the error message from the elem_context.

The primary reason for the context is to have a place to
store the error message of the last error. This was easier
than mess around with some thread-safe global error storage.

Also, tuning options etc. could be stored in the context, though
currently the underlying C++ Elemental don't suipport this.
*/

int main(int argc, char *argv[]) {
  elem_matrix *A, *B, *C;
  elem_context *ctx;
  elem_grid *grid;
  elem_complex alpha, beta;

  double A_buf[16], B_buf[16], C_buf[16];

  MPI_Init(&argc, &argv);
  ctx = elem_create_context();
  /* reason initialize is a separate function is that it could raise
      C++ exceptions that we want to report in the ctx object */
  elem_initialize(ctx);
  grid = elem_create_grid(ctx, MPI_COMM_WORLD, 2, 2);


  /* Todo: allow passing -1/0 for alignment/lda for "use default" */
  A = elem_create_matrix(grid, ELEM_DOUBLE_REAL, ELEM_MC, ELEM_MR,
                         4, 4, 0, 0, A_buf, 4);
  B = elem_create_matrix(grid, ELEM_DOUBLE_REAL, ELEM_MC, ELEM_MR,
                         4, 4, 0, 0, B_buf, 4);
  C = elem_create_matrix(grid, ELEM_DOUBLE_REAL, ELEM_MC, ELEM_MR,
                         4, 4, 0, 0, C_buf, 4);

  elem_set_to_identity(A);
  elem_set_to_identity(B);
  elem_set_to_identity(C);

  alpha = (elem_complex){2.0, 0.0};
  beta = (elem_complex){3.0, 0.0};
  elem_gemm(ELEM_NORMAL, ELEM_NORMAL, alpha, A, B, beta, C);
  elem_print(C);

  elem_destroy_matrix(A);
  elem_destroy_matrix(B);
  elem_destroy_matrix(C);
  elem_destroy_grid(grid);
  elem_finalize(ctx);
  elem_destroy_context(ctx);
  MPI_Finalize();
}
