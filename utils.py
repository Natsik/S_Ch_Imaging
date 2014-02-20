__author__ = 'aynroot'

from PyQt4 import QtGui
import numpy as np


def qimage_to_np_array(img):
    """  converts a QImage into an numpy array """
    # TODO: not tested
    # TODO: check format
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    width = img.width()
    height = img.height()

    ptr = img.constBits()
    array = np.array(ptr).reshape(height, width, 4)
    return array


def np_array_to_qimage(bgra_array):
    """ converts numpy array (bgra format) into a QImage """
    # TODO: not tested
    h, w = bgra_array.shape
    img = QtGui.QImage(bgra_array.data, w, h, QtGui.QImage.Format_RGB32)
    img.ndarray = bgra_array
    return img

    '''
    # some suggestions to make grayscale image
    COLORTABLE = [~((i + (i<<8) + (i<<16))) for i in range(255,-1,-1)]

    image = QtGui.QImage(x.data,w,h,QtGui.QImage.Format_Indexed8)
    image.setColorTable(COLORTABLE)
    '''

