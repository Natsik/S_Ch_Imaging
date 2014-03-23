__author__ = 'natsik'

from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("MS5",
                           sources=["_diff.pyx", "diff.c"],
                           include_dirs=[numpy.get_include()])],
)