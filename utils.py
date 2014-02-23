__author__ = 'aynroot'

from PyQt4.QtGui import QImage, qRgb
from PIL import Image
import numpy as np


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
            # raise NotImplementedError
            qimg = QImage(np_img.data, np_img.shape[1], np_img.shape[0], np_img.strides[0], QImage.Format_ARGB32)
            return qimg.copy() if copy else qimg

    raise NotImplementedError


def PIL_to_np(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def np_to_PIL(np_img, size):
    mode = 'RGBA'
    np_img = np_img.reshape(np_img.shape[0]*np_img.shape[1], np_img.shape[2])
    if len(np_img[0]) == 3:
        np_img = np.c_[np_img, 255 * np.ones((len(np_img), 1), np.uint8)]
    return Image.frombuffer(mode, size, np_img.tostring(), 'raw', mode, 0, 1)