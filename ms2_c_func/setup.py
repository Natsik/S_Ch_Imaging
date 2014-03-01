__author__ = 'aynroot'

from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("MS2",
                           sources=["_invdilerr.pyx", "invdilerr.c"],
                           include_dirs=[numpy.get_include()])],
)