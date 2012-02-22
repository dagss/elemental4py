#./waf-light --tools=compat15,swig,fc,compiler_fc,fc_config,fc_scan,gfortran,g95,ifort,gccdeps;

import os
from textwrap import dedent

top = '.'
out = 'build'

def options(opt):
    opt.add_option('--with-elemental', help='path to Elemental')
    opt.add_option('--with-goto2', help='path to GotoBLAS2')
    opt.load('compiler_c')
    opt.load('compiler_cxx')
    opt.load('python')
    opt.load('inplace', tooldir='tools')

def configure(conf):
    conf.add_os_flags('PATH')
    conf.add_os_flags('PYTHON')
    conf.add_os_flags('PYTHONPATH')
    conf.add_os_flags('INCLUDES')
    conf.add_os_flags('LIB')
    conf.add_os_flags('LIBPATH')
    conf.add_os_flags('STLIB')
    conf.add_os_flags('STLIBPATH')
    conf.add_os_flags('CFLAGS')
    conf.add_os_flags('CXXFLAGS')
    conf.add_os_flags('LINKFLAGS')
    conf.add_os_flags('CYTHONFLAGS')
    conf.add_os_flags('CXX')
    conf.add_os_flags('CXXFLAGS')

    conf.load('compiler_c')
    conf.load('compiler_cxx')

    conf.load('python')
    conf.check_python_version((2,5))
    conf.check_python_headers()

    conf.check_tool('numpy', tooldir='tools')
    conf.check_numpy_version(minver=(1,3))
    conf.check_tool('cython', tooldir='tools')
    conf.check_cython_version(minver=(0,11,1))
    conf.check_tool('inplace', tooldir='tools')

    conf.env.INCLUDES_ELEMENTAL = [
        os.path.join(conf.options.with_elemental, 'include')]
    conf.env.LIBPATH_ELEMENTAL = [
        os.path.join(conf.options.with_elemental, 'lib')]
    conf.env.LIB_ELEMENTAL = ['elemental', 'plcg', 'lapack-addons']

    conf.env.RPATH_BLAS = conf.options.with_goto2
    conf.env.LIBPATH_BLAS = conf.options.with_goto2
    conf.env.LIB_BLAS = ['goto2', 'gfortran']

def build(bld):
    bld(source=['elemental/elemental.pyx', 'src/elemental_wrapper.cpp.in'],
        includes=['src'],
        target='elemental',
        use='ELEMENTAL BLAS',
        features='pyext cxxshlib cxx')


from waflib.Configure import conf
from os.path import join as pjoin

from waflib import TaskGen


def run_tempita(task):
    import tempita
    import re
    assert len(task.inputs) == len(task.outputs) == 1
    tmpl = task.inputs[0].read()
    result = tempita.sub(tmpl)
    result, n = re.subn(r'/\*.*?\*/', '', result, flags=re.DOTALL)
    result = '\n'.join('/*!*/  %s' % x for x in result.splitlines())
    result = '/* DO NOT EDIT THIS FILE, IT IS GENERATED */\n%s' % result
    task.outputs[0].write(result)

TaskGen.declare_chain(
        name = "tempita",
        rule = run_tempita,
        ext_in = ['.cpp.in'],
        ext_out = ['.cpp'],
        reentrant = True,
        )
