__author__ = 'aynroot'

import scipy.misc
from PyQt4 import QtGui, QtCore
import utils

ext_mappings = {
    'jpeg': 'JPEG',
    'jpg': 'JPEG',
    'bmp': 'BMP',
    'png': 'PNG'
}

filter_mappings = {
    'JPEG': 'jpg',
    'BMP': 'bmp',
    'PNG': 'png'
}


class ImageOpener(object):

    # TODO: cancel open dialog protection
    def __init__(self, main_window):
        self.main_window = main_window
        self.filename = None

    def open_file(self):
        self.filename = str(QtGui.QFileDialog.getOpenFileName(self.main_window, 'Open File...', '.',
                                                             'Image File(*.png *.jpg *.jpeg *.bmp);; All Files (*)'))
        if self.filename:
            return self._load(self.filename)

    def _load(self, filename):
        if not QtCore.QFile.exists(filename):
            raise IOError

        fh = QtCore.QFile(filename)
        if not fh.open(QtCore.QFile.ReadOnly):
            raise IOError
        np_img = scipy.misc.imread(filename)
        return np_img


class ImageSaver(object):
    def __init__(self, main_window):
        self.main_window = main_window
        self.filename = None

    def save_as_file(self):
        filename, filter = QtGui.QFileDialog().getSaveFileNameAndFilter(self.main_window, 'Save File...', '.',
                                                                        'JPEG (*jpeg *jpg);; BMP (*bmp);; PNG (*png)')
        extension = filter_mappings[str(filter).split()[0]]
        filename = '.'.join((str(filename), extension))
        if filename:
            self.filename = filename
            self._save()

    def save_file(self):
        if self.filename:
            self._save()
        else:
            self.save_as_file()

    def _save(self):
        img = utils.np_to_qimage(self.main_window.np_img)
        img.save(self.filename, ext_mappings[self.filename.lower().split('.')[-1]])

    @staticmethod
    def save_any(np_img, filename):
        img = utils.np_to_qimage(np_img)
        img.save(filename, ext_mappings[filename.lower().split('.')[-1]])