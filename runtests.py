import nose
from nose.plugins import Plugin
from mpi4py import MPI

is_root = MPI.COMM_WORLD.Get_rank() == 0


class NoopStream(object):
    def write(self, *args):
        pass
    
    def writeln(self, *args):
        pass

    def flush(self):
        pass

class MpiOutput(Plugin):
    """
    Have only rank 0 report test results. Test results are aggregated
    across processes, i.e., if an exception happens in a single
    process then that is reported, otherwise if an assertion failed in any
    process then that is reported, otherwise it's a success.
    """
    # Required attributes:
    
    name = 'mpi'
    enabled = True

    def setOutputStream(self, stream):
        if not is_root:
            return NoopStream()

#    def stopTest(self, test):
#        if is_root:
#            print 'after', test

#    def addSuccess(self, test):
#        if is_root:
#            print 'success', test
#self.stream.writeln("passed")
        
#    def addError(self, test):
#        print 'error'
            
#    def addFailure(self, test):
#        print 'failure'


if __name__ == '__main__':
    import sys
    nose.main(addplugins=[MpiOutput()], argv=sys.argv)