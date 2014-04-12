__author__ = 'aynroot'

import numpy as np
import traceback
from ms2_c_func import MS2
from ms3_c_func import MS3
from ms5_c_func import MS5
from msmf_c_func import MSMED
from utils import NumpyCImageConverter


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
        MS3.c_linear_filter_func(self.c_img, new_c_img, self.np_shape[1] * 3, self.np_shape[0],
                                   np.array(matrix).flatten(), matrix.shape[0], divisor)
        self.c_img = new_c_img

    @c_call
    def white_noise(self, p, d):
        new_c_img = self.c_img.copy()
        MS3.c_white_noise_func(self.c_img, new_c_img, self.np_shape[1] * 3, self.np_shape[0], int(p), int(d))
        self.c_img = new_c_img


    @c_call
    def dust(self, p, min_value):
        new_c_img = self.c_img.copy()
        MS3.c_fog_func(self.c_img, new_c_img, self.np_shape[1] * 3, self.np_shape[0], int(p), int(min_value))
        self.c_img = new_c_img

    @c_call
    def grid(self, grid_w, grid_h):
        new_c_img = self.c_img.copy()
        MS3.c_mesh_func(self.c_img, new_c_img, self.np_shape[1] * 3, self.np_shape[0], int(grid_w), int(grid_h), 225)
        self.c_img = new_c_img

    @c_call
    def median_filter(self, r):
        new_c_img = self.c_img.copy()
        MSMED.c_median_filter_func(self.c_img, new_c_img, self.np_shape[1] * 3, self.np_shape[0], r)
        self.c_img = new_c_img

    def diff_images(self, golden_np_img):
        try:
            if self.np_img.shape != golden_np_img.shape:
                return False, None, None

            converter = NumpyCImageConverter()
            converter.update_image(golden_np_img)

            c_img1 = self.to_c_format()
            c_img2 = converter.to_c_format()
            c_diff_img = np.zeros_like(c_img1)
            percentage = MS5.c_diff_images_func(c_img1, c_img2, c_diff_img, self.np_shape[1] * 3, self.np_shape[0]) * 100
            converter.c_img = c_diff_img
            return True, converter.from_c_format(), percentage
        except:
            traceback.print_exc()
            return False, None, None

