__author__ = 'natsik'

from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("MS3",
                           sources=["_filternoise.pyx", "filternoise.c"],
                           include_dirs=[numpy.get_include()])],
)