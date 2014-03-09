__author__ = 'aynroot'

import numpy as np
from ms2_c_func import MS2
from mswam_c_func import MSWAM


def c_call(function):
    def c_call_wrapper(self, *args):
        self.c_img = self.to_c_format()
        function(self, *args)
        self.np_img = self.from_c_format()
        return self.np_img
    return c_call_wrapper


class ImageEditor(object):

    integration_filter_matrix = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    blur_matrix = np.matrix([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    sharpen_matrix = np.matrix([[0, -3, 0], [-3, 21, -3], [0, -3, 0]])

    integration_filter_divisor = 9
    blur_divisor = 5
    sharpen_divisor = 9

    def __init__(self):
        self.np_img = None
        self.c_img = None
        self.np_shape = None
        self.stored_alpha_channel = []

    def update_image(self, np_img):
        self.np_img = np_img
        self.np_shape = self.np_img.shape

    def to_c_format(self):
        if self.np_shape[2] == 3:
            self.c_img = self.np_img.flatten()
        elif self.np_shape[2] == 4:
            r, g, b, a = np.rollaxis(self.np_img, axis=-1)
            self.np_img = np.dstack([r, g, b])
            self.c_img = self.np_img.flatten()
            self.stored_alpha_channel = a
        return self.c_img

    def from_c_format(self):
        if self.np_shape[2] == 3:
            self.np_img = self.c_img.reshape(self.np_shape)
        elif self.np_shape[2] == 4:
            shape = (self.np_shape[0], self.np_shape[1], 3)
            self.np_img = self.c_img.reshape(shape)
            r, g, b = np.rollaxis(self.np_img, axis=-1)
            self.np_img = np.dstack([r, g, b, self.stored_alpha_channel])
        return self.np_img

    @c_call
    def erosion(self):
        new_c_img = self.c_img.copy()
        MS2.c_errosion_func(self.c_img, new_c_img, self.np_shape[1] * 3, self.np_shape[0])
        self.c_img = new_c_img

    @c_call
    def dilatation(self):
        new_c_img = self.c_img.copy()
        MS2.c_dilatation_func(self.c_img, new_c_img, self.np_shape[1] * 3, self.np_shape[0])
        self.c_img = new_c_img

    @c_call
    def inversion(self):
        new_c_img = self.c_img.copy()
        MS2.c_inversion_func(self.c_img, new_c_img, self.np_shape[1] * 3, self.np_shape[0])
        self.c_img = new_c_img

    @c_call
    def linear_filter(self, matrix, divisor):
        new_c_img = self.c_img.copy()
        MSWAM.c_linear_filtre_func(self.c_img, new_c_img, self.np_shape[1] * 3, self.np_shape[0],
                                   np.array(matrix).flatten(), matrix.shape[0], divisor)
        self.c_img = new_c_img
