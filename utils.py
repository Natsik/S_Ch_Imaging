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