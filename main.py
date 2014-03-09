__author__ = 'aynroot'

import sys

from PyQt4 import QtGui, QtCore
from ui_main import Ui_MainWindow

from image_open_close import ImageOpener, ImageSaver
from image_editor import ImageEditor
from history import History
import utils


class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.np_img = None
        self.image_actions = []
        self._init_ui()

        self.view = self.ui.imageLabel
        self.image_opener = ImageOpener(self)
        self.image_saver = ImageSaver(self)
        self.image_editor = ImageEditor()
        self.image_history = History()

        self._init_file_actions()
        self._init_M2_actions()
        self._init_M3_actions()
        self._enable_menu_items(False)

    def _init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.ui.imageLabel)

    def _init_file_actions(self):
        # TODO: undo / redo enable/disable funcs
        self.ui.actionExit.triggered.connect(QtGui.qApp.quit)
        self.ui.actionOpen.triggered.connect(self._open_file)
        self.ui.actionSave.triggered.connect(self._save_file)
        self.ui.actionSave_As.triggered.connect(self.image_saver.save_as_file)
        self.ui.actionUndo.triggered.connect(self._undo)
        self.ui.actionRedo.triggered.connect(self._redo)

        self.image_actions.append(self.ui.actionSave)
        self.ui.actionSave_As.setEnabled(False)

    def _init_M2_actions(self):
        self.ui.actionErosion.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.erosion))
        self.ui.actionDilatation.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.dilatation))
        self.ui.actionInversion.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.inversion))
        self.image_actions.extend([self.ui.actionErosion, self.ui.actionDilatation, self.ui.actionInversion])

    def _init_M3_actions(self):
        # TODO: write triggers
        self.image_actions.extend([])

    def _enable_menu_items(self, mode):
        for action in self.image_actions:
            action.setEnabled(mode)

    def _image_editor_wrapper(self, editor_func):
        self.image_editor.update_image(self.np_img)
        self.np_img = editor_func()
        img = utils.np_to_qimage(self.np_img)
        pixmap = QtGui.QPixmap.fromImage(img)
        pixmap = self.scale_pixmap(pixmap)
        self.view.setPixmap(pixmap)
        self._enable_menu_items(True)

    def _open_file(self):
        self.np_img = self.image_opener.open_file()
        self.image_saver.filename = self.image_opener.filename

        # in case of png files swap channels
        if self.np_img.shape[2] == 4:
            self.np_img = utils.bgra2rgba(self.np_img)
        self._show_np_image()

        self._enable_menu_items(True)
        self.ui.actionSave_As.setEnabled(True)

    def _save_file(self):
        self.image_saver.np_img = self.np_img
        self.image_saver.save_file()
        self.ui.actionSave.setEnabled(False)

    def _undo(self):
        if self.image_history.can_undo():
            self.np_img = self.image_history.undo()
        if not self.image_history.can_undo():
            self.ui.actionUndo.setEnabled(False)
            self.ui.actionRedo.setEnabled(True)

    def _redo(self):
        if self.image_history.can_redo():
            self.np_img = self.image_history.redo()
        if not self.image_history.can_redo():
            self.ui.actionUndo.setEnabled(True)
            self.ui.actionRedo.setEnabled(False)

    def _show_np_image(self):
        img = utils.np_to_qimage(self.np_img)
        self._show_image(img)

    def _show_image(self, img):
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