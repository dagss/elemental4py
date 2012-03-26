from .common import *
import elemental

ctx = elemental.Context()

@mpi(4)
def test_basic_set_identity(comm):
    grid = elemental.Grid(ctx, comm, 2, 2)
    A = elemental.DistMatrix(grid, 8, 8)
    A.set_to_identity()
    if comm.Get_rank() in (0, 3):
        assert np.all(A.local == np.eye(4))
    else:
        assert np.all(A.local == 0)

