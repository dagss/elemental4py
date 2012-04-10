from nose.tools import eq_, ok_
from mpi4py import MPI
import numpy as np
import cPickle as pickle
import sys

from ..mpiutils import *

import functools

def format_exc_info():
    import traceback
    type_, value, tb = sys.exc_info()
    msg = traceback.format_exception(type_, value, tb)
    return ''.join(msg)

def first_nonzero(arr):
    """
    Find index of first nonzero element in the 1D array `arr`, or raise
    IndexError if no such element exists.
    """
    hits = np.nonzero(arr)
    assert len(hits) == 1
    if len(hits[0]) == 0:
        raise IndexError("No non-zero elements")
    else:
        return hits[0][0]

def mpi(nprocs):
    """
    Runs a testcase using a `nprocs`-sized subset of COMM_WORLD. Also
    synchronizes results, so that a failure or error in one process
    causes all ranks to fail or error. The algorithm is:

     - If a process fails (AssertionError) or errors (any other exception),
       it propagates that

     - If a process succeeds, it reports the error of the lowest-ranking
       process that err-ed (by raising an error containing the stack trace
       as a string). If not other processes errored, the same is repeated
       with failures. Finally, the process succeeds.
    """
    def dec(func):
        @functools.wraps(func)
        def mpi_test():
            n = MPI.COMM_WORLD.Get_size()
            rank = MPI.COMM_WORLD.Get_rank()
            if n < nprocs:
                raise RuntimeError('Number of available MPI processes (%d) '
                                   'too small' % n)
            sub_comm = MPI.COMM_WORLD.Split(0 if rank < nprocs else 1, 0)
            SUCCESS, ERROR, FAILED = range(3)
            status = SUCCESS
            exc_msg = ''
            try:
                if rank < nprocs:
                    try:
                        func(sub_comm)
                    except AssertionError:
                        status = FAILED
                        exc_msg = format_exc_info()
                        raise
                    except:
                        status = ERROR
                        exc_msg = format_exc_info()
                        raise
            finally:
                # Do communication of error results in a final block, so
                # that also erring/failing processes participate

                # First, figure out status of other nodes
                statuses = MPI.COMM_WORLD.allgather(status)
                try:
                    first_non_success = first_nonzero(statuses)
                except IndexError:
                    first_non_success_status = SUCCESS
                else:
                    # First non-success gets to broadcast it's error
                    first_non_success_status, msg = MPI.COMM_WORLD.bcast(
                        (status, exc_msg), root=first_non_success)
                    
                # Exit finally-block -- erring/failing processes return here

            # Did not return -- so raise some other process' error or failure
            fmt = '%s in MPI rank %d:\n\n"""\n%s"""\n'
            if first_non_success_status == ERROR:
                msg = fmt % ('ERROR', first_non_success, msg)
                raise RuntimeError(msg)
            elif first_non_success_status == FAILED:
                msg = fmt % ('FAILURE', first_non_success, msg)
                raise AssertionError(msg)

        return mpi_test
    return dec


