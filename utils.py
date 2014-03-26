__author__ = 'aynroot'

from PyQt4.QtGui import QImage, qRgb
import numpy as np


#      ___800___
#     |         |
# 600 |         |
#     |_________|
#
# len(np_img[0]) == 800
# len(np_img[0][0]) == 4


def bgra2rgba(np_img):
    n, m, _ = np_img.shape
    np_img.shape = (-1, 4)
    np_img = np_img[:, [2, 1, 0, 3]]
    np_img = np_img.reshape(n, m, 4)
    return np_img.copy()


def np_to_qimage(np_img, copy=False):
    gray_color_table = [qRgb(i, i, i) for i in range(256)]
    if np_img is None:
        return QImage()

    if np_img.dtype != np.uint8:
        print np_img.dtype
        np.clip(np_img, 0, 255, out=np_img)
        np_img = np_img.astype('uint8')

    if len(np_img.shape) == 2:
        qimg = QImage(np_img.data, np_img.shape[1], np_img.shape[0], np_img.strides[0], QImage.Format_Indexed8)
        qimg.setColorTable(gray_color_table)
        return qimg.copy() if copy else qimg
    elif len(np_img.shape) == 3:
        if np_img.shape[2] == 3:
            qimg = QImage(np_img.data, np_img.shape[1], np_img.shape[0], np_img.strides[0], QImage.Format_RGB888)
            return qimg.copy() if copy else qimg
        elif np_img.shape[2] == 4:
            qimg = QImage(np_img.data, np_img.shape[1], np_img.shape[0], np_img.strides[0], QImage.Format_ARGB32)
            return qimg.copy() if copy else qimg

    raise NotImplementedError


class NumpyCImageConverter(object):
    def __init__(self):
        self.np_img = None
        self.c_img = None
        self.stored_alpha_channel = None
        self.np_shape = None

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


