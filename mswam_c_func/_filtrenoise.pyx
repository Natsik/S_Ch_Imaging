import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "filtrenoise.h":
    void c_linear_filtre(const unsigned char * src, unsigned char * dst, int channelStride, int h, int * matrix, int matrix_dimension, int delimeter)
    void c_white_noise(const unsigned char * src, unsigned char * dst, int channelStride, int h, int p, int d)
    void c_bil(const unsigned char * src, unsigned char * dst, int channelStride, int h, int p, int min)
    void c_mesh(const unsigned char * src, unsigned char * dst, int channelStride, int h, int mesh_w, int mesh_h, unsigned char tone)

def c_linear_filtre_func(np.ndarray[char, ndim=1, mode="c"] in_array not None,
                         np.ndarray[char, ndim=1, mode="c"] out_array not None,
                         channel_strides, h,
		                 np.ndarray[int, ndim=1, mode="c"] in_matrix_array not None,
		                 matrix_dimension, delimeter):
    c_linear_filtre(<unsigned char*> np.PyArray_DATA(in_array),
                    <unsigned char*> np.PyArray_DATA(out_array),
                    channel_strides, h,
                    <int*> np.PyArray_DATA(in_matrix_array),
                    matrix_dimension, delimeter)

def c_white_noise_func(np.ndarray[char, ndim=1, mode="c"] in_array not None,
                       np.ndarray[char, ndim=1, mode="c"] out_array not None,
                       channel_strides, h, p, d):
    c_white_noise(<unsigned char*> np.PyArray_DATA(in_array),
                 <unsigned char*> np.PyArray_DATA(out_array),
                 channel_strides, h, p, d)

def c_bil_func(np.ndarray[char, ndim=1, mode="c"] in_array not None,
               np.ndarray[char, ndim=1, mode="c"] out_array not None,
               channel_strides, h, p, min):
    c_bil(<unsigned char*> np.PyArray_DATA(in_array),
          <unsigned char*> np.PyArray_DATA(out_array),
          channel_strides, h, p, min)

def c_mesh_func(np.ndarray[char, ndim=1, mode="c"] in_array not None,
                np.ndarray[char, ndim=1, mode="c"] out_array not None,
                channel_strides, h, mesh_w, mwsh_h, tone):
    c_mesh(<unsigned char*> np.PyArray_DATA(in_array),
           <unsigned char*> np.PyArray_DATA(out_array),
           channel_strides, h, mesh_w, mwsh_h, tone)