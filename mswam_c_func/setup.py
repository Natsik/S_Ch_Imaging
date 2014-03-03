__author__ = 'natsik'

from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("MSWAM",
                           sources=["_filtrenoise.pyx", "filtrenoise.c"],
                           include_dirs=[numpy.get_include()])],
)