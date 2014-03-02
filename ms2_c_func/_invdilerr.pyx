import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "invdilerr.h":
    void c_inversion(const unsigned char *src, unsigned char *dst, int channelStride, int h)
    void c_dilatation(const unsigned char *src, unsigned char *dst, int channelStride, int h)
    void c_errosion(const unsigned char *src, unsigned char *dst, int channelStride, int h)

def c_inversion_func(np.ndarray[char, ndim=1, mode="c"] in_array not None,
                     np.ndarray[char, ndim=1, mode="c"] out_array not None,
                     channel_strides, h):
    c_inversion(<unsigned char*> np.PyArray_DATA(in_array),
                <unsigned char*> np.PyArray_DATA(out_array),
                channel_strides, h)

def c_dilatation_func(np.ndarray[char, ndim=1, mode="c"] in_array not None,
                      np.ndarray[char, ndim=1, mode="c"] out_array not None,
                      channel_strides, h):
    c_dilatation(<unsigned char*> np.PyArray_DATA(in_array),
                 <unsigned char*> np.PyArray_DATA(out_array),
                 channel_strides, h)

def c_errosion_func(np.ndarray[char, ndim=1, mode="c"] in_array not None,
                    np.ndarray[char, ndim=1, mode="c"] out_array not None,
                    channel_strides, h):
    c_errosion(<unsigned char*> np.PyArray_DATA(in_array),
               <unsigned char*> np.PyArray_DATA(out_array),
               channel_strides, h)