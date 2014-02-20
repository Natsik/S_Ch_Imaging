__author__ = 'aynroot'

import sys
from PyQt4 import QtGui, QtCore
from ui_main import Ui_MainWindow

# from pylab import imread, mean

def describe(obj):
    ''' helper function that prints info about obj attributes '''
    for key in dir(obj):
        try:
            val = getattr(obj, key)
        except AttributeError:
            continue
        if callable(val):
            help(val)
        else:
            print('{k} => {v}'.format(k = key, v = val))
        print('-'*80)


class ImageOpener(object):

    def __init__(self, main_window, view):
        self.main_window = main_window
        self.view = view

    def open_file(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(self.main_window, 'Open File...', '.',
                                                         'Image File(*.png *.jpg *.jpeg *.bmp);; All Files (*)'))
        if filename:
            self._load(filename)

    def _load(self, filename):
        if not QtCore.QFile.exists(filename):
            return False

        fh = QtCore.QFile(filename)
        if not fh.open(QtCore.QFile.ReadOnly):
            return False

        pixmap = QtGui.QPixmap(filename)
        dest_height, dest_width = self.view.size().height(), self.view.size().width()
        if dest_height < pixmap.size().height():
            pixmap = pixmap.scaledToHeight(self.view.size().height())
        elif dest_width < pixmap.size().width():
            pixmap = pixmap.scaledToWidth(self.view.size().width())
        self.view.setPixmap(pixmap)
        return True


class ImageSaver(object):

    def __init__(self, main_window, view):
        self.main_window = main_window
        self.view = view
        self.filename = None

    def save_as_file(self):
        filename = str(QtGui.QFileDialog().getSaveFileName(self.main_window, 'Save File...', '.',
                                                           'JPEG (*jpeg *jpg);; BMP(*bmp);; PNG(*png)'))

        if filename:
            if not filename.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
                filename += '.jpg'
            self.filename = filename
            self._save()

    def save_file(self):
        # TODO: twisted logic?
        if self.filename:
            self._save()
        else:
            self.save_as_file()

    def _save(self):
        pixmap = self.view.pixmap()
        img = pixmap.toImage()
        # TODO: maybe make work with extensions prettier
        ext_mappings = {
            'jpeg': 'JPEG',
            'jpg': 'JPEG',
            'bmp': 'BMP',
            'png': 'PNG'
        }
        img.save(self.filename, ext_mappings[self.filename.lower().split('.')[-1]])


class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()              
        self.init_ui()        

        self.image_opener = ImageOpener(self, self.ui.imageLabel)
        self.image_saver = ImageSaver(self, self.ui.imageLabel)
        self.init_file_actions()

    def init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.ui.imageLabel)

    def init_file_actions(self):
        self.ui.actionExit.triggered.connect(QtGui.qApp.quit)
        self.ui.actionOpen.triggered.connect(self.image_opener.open_file)
        self.ui.actionSave.triggered.connect(self.image_saver.save_file)
        self.ui.actionSave_As.triggered.connect(self.image_saver.save_as_file)

    def show_window(self):
        self.show()            


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mw = MainWindow()
    mw.show_window()
    sys.exit(app.exec_())