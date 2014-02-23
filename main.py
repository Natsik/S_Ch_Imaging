__author__ = 'aynroot'

import sys
import scipy.misc

from PyQt4 import QtGui, QtCore
from ui_main import Ui_MainWindow

from image_editor import ImageEditor
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

    def __init__(self, main_window):
        self.main_window = main_window

    def open_file(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(self.main_window, 'Open File...', '.',
                                                         'Image File(*.png *.jpg *.jpeg *.bmp);; All Files (*)'))
        if filename:
            return self._load(filename)

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
        self.np_img = None

    def save_as_file(self):
        filename, filter = QtGui.QFileDialog().getSaveFileNameAndFilter(self.main_window, 'Save File...', '.',
                                                                        'JPEG (*jpeg *jpg);; BMP (*bmp);; PNG (*png)')
        extension = filter_mappings[str(filter).split()[0]]
        filename = '.'.join((str(filename), extension))
        if filename:
            self.filename = filename
            self._save()

    def save_file(self):
        # TODO: twisted logic?
        if self.filename:
            self._save()
        else:
            self.save_as_file()

    def _save(self):
        img = utils.np_to_qimage(self.np_img)
        img.save(self.filename, ext_mappings[self.filename.lower().split('.')[-1]])


class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.np_img = None
        self.init_ui()

        self.view = self.ui.imageLabel
        self.image_opener = ImageOpener(self)
        self.image_saver = ImageSaver(self)
        self.image_editor = ImageEditor()
        self.init_file_actions()
        self.init_edit_actions()

    def init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.ui.imageLabel)

    def init_file_actions(self):
        self.ui.actionExit.triggered.connect(QtGui.qApp.quit)
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionSave.triggered.connect(self.save_file)
        self.ui.actionSave_As.triggered.connect(self.image_saver.save_as_file)

    def init_edit_actions(self):
        self.ui.actionGrayscale.triggered.connect(lambda: self.image_editor_wrapper(self.image_editor.grayscale))

    def image_editor_wrapper(self, editor_func):
        self.image_editor.np_img = self.np_img
        pixmap = QtGui.QPixmap.fromImage(editor_func())
        pixmap = self.scale_pixmap(pixmap)
        self.view.setPixmap(pixmap)

    def open_file(self):
        self.np_img = self.image_opener.open_file()

        # in case of png files swap channels
        if self.np_img.shape[2] == 4:
            self.np_img = utils.bgra2rgba(self.np_img)

        self.show_np_image()

    def save_file(self):
        self.image_saver.np_img = self.np_img
        self.image_saver.save_file()

    def show_np_image(self):
        img = utils.np_to_qimage(self.np_img)
        self.show_image(img)

    def show_image(self, img):
        pixmap = QtGui.QPixmap.fromImage(img)
        pixmap = self.scale_pixmap(pixmap)
        self.view.setPixmap(pixmap)

    def scale_pixmap(self, pixmap):
        dest_height, dest_width = self.view.size().height(), self.view.size().width()
        if dest_height < pixmap.size().height():
            pixmap = pixmap.scaledToHeight(self.view.size().height())
        elif dest_width < pixmap.size().width():
            pixmap = pixmap.scaledToWidth(self.view.size().width())
        return pixmap

    def show_window(self):
        self.show()            


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mw = MainWindow()
    mw.show_window()
    sys.exit(app.exec_())