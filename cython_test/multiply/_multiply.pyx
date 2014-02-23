"""
multiply.pyx

simple cython test of accessing a numpy array's data

the C function: c_multiply multiplies all the values in a 2-d array by a scalar, in place.

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

# cdefine the signature of our c function
cdef extern from "multiply.h":
    void c_multiply (double* array, double multiplier, int m, int n)




# create the wrapper code, with numpy type annotations
def multiply(np.ndarray[double, ndim=2, mode="c"] input not None, double value):
    c_multiply(<double*> np.PyArray_DATA(input), value, input.shape[0], input.shape[1])