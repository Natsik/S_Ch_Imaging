import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "median.h":
    void c_median_filter(const unsigned char * src, unsigned char * dst, int channelStride, int h, int matrix_radius)

def c_median_filter_func(np.ndarray[char, ndim=1, mode="c"] in_array not None,
                    np.ndarray[char, ndim=1, mode="c"] out_array not None,
                    channel_strides, h, matrix_radius):
    c_median_filter(<unsigned char*> np.PyArray_DATA(in_array),
                    <unsigned char*> np.PyArray_DATA(out_array),
                    channel_strides, h, matrix_radius)
