import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "diff.h":
    double c_diff_images(const unsigned char * src1, const unsigned char * src2,  unsigned char * diff, int channelStride, int h)

def c_diff_images_func(np.ndarray[char, ndim=1, mode="c"] in_array1 not None,
                np.ndarray[char, ndim=1, mode="c"] in_array2 not None,
                np.ndarray[char, ndim=1, mode="c"] out_array not None,
                channel_strides, h):
    return c_diff_images(<unsigned char*> np.PyArray_DATA(in_array1),
           <unsigned char*> np.PyArray_DATA(in_array2),
           <unsigned char*> np.PyArray_DATA(out_array),
           channel_strides, h)