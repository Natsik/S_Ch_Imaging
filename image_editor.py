__author__ = 'aynroot'

import scipy.misc
import utils


class ImageEditor(object):

    def __init__(self):
        self.np_img = None

    def update_image(self, np_img):
        self.np_img = np_img

    def grayscale(self):
        np_img = scipy.misc.imread('lena.jpg')
        return utils.np_to_qimage(np_img)
