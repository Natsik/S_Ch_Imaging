__author__ = 'aynroot'

import os
import sys

from PyQt4 import QtGui, QtCore
import scipy.misc

from ui_main import Ui_MainWindow
from make_new_custom_linear_filter_dialog import MakeNewCustomLinearFilterDialog
from base_two_params_dialog import BaseTwoParamsDialog

from image_open_close import ImageOpener, ImageSaver
from image_editor import ImageEditor
from history import History
from user_filters_dump_n_loader import UserSettingsDumpNLoader
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
        self.user_settings_dump_n_loader = UserSettingsDumpNLoader()
        self.diff_images_dir = self.user_settings_dump_n_loader.get_diff_images_dir()
        self.golden_images_dir = self.user_settings_dump_n_loader.get_golden_images_dir()

        self._init_file_actions()
        self._init_M2_actions()
        self._init_M3_actions()
        self._init_M4_actions()
        self._init_M5_actions()
        self._init_M6_actions()
        self._enable_menu_items(False)

    def _init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.ui.imageLabel)

    def _init_file_actions(self):
        self.ui.actionExit.triggered.connect(QtGui.qApp.quit)
        self.ui.actionOpen.triggered.connect(self._open_file)
        self.ui.actionSave.triggered.connect(self._save_file)
        self.ui.actionSave_As.triggered.connect(self.image_saver.save_as_file)
        self.ui.actionUndo.triggered.connect(self._undo)
        self.ui.actionRedo.triggered.connect(self._redo)

        self.image_actions.append(self.ui.actionSave)
        self.ui.actionSave_As.setEnabled(False)
        self._enable_undo_redo()

    def _init_M2_actions(self):
        self.ui.actionErosion.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.erosion))
        self.ui.actionDilatation.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.dilatation))
        self.ui.actionInversion.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.inversion))
        self.image_actions.extend([self.ui.actionErosion, self.ui.actionDilatation, self.ui.actionInversion])

    def _init_M3_actions(self):
        self.ui.actionIntegrating_filter.triggered.connect(
            lambda: self._image_editor_wrapper(self.image_editor.linear_filter, ImageEditor.integration_filter_matrix,
                                               ImageEditor.integration_filter_divisor)
        )
        self.ui.actionBlur.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.linear_filter,
                                             ImageEditor.blur_matrix, ImageEditor.blur_divisor))
        self.ui.actionSharpen.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.linear_filter,
                                                ImageEditor.sharpen_matrix, ImageEditor.sharpen_divisor))
        self.ui.actionMake_new_custom_filter.triggered.connect(self._make_new_custom_filter)
        for filter_name, (matrix, divisor) in self.user_settings_dump_n_loader.get_filters().iteritems():
            action = QtGui.QAction(self)
            action.setText(filter_name)
            action.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.linear_filter,
                                     matrix, divisor))
            self.ui.menuCustom_filters.addAction(action)
        self.image_actions.extend([self.ui.actionIntegrating_filter, self.ui.actionBlur,
                                   self.ui.actionSharpen, self.ui.actionMake_new_custom_filter])

    def _init_M4_actions(self):
        self.ui.actionWhite_noise.triggered.connect(lambda: self._get_two_params_and_edit("Probability (%)", "Range",
                                                                                          self.image_editor.white_noise))
        self.ui.actionDust.triggered.connect(lambda: self._get_two_params_and_edit("Probability (%)", "Min value",
                                                                                   self.image_editor.dust))
        self.ui.actionGrid.triggered.connect(lambda: self._get_two_params_and_edit("Width", "Height",
                                                                                   self.image_editor.grid))
        self.image_actions.extend([self.ui.actionWhite_noise, self.ui.actionDust, self.ui.actionGrid])

    def _init_M5_actions(self):
        self.ui.actionDifference.triggered.connect(lambda: self._diff_images())
        self.ui.actionSet_diff_images_path.triggered.connect(lambda: self._set_diff_images_dir(self._choose_directory()))
        self.ui.actionSet_golden_images_path.triggered.connect(lambda: self._set_golden_images_dir(self._choose_directory()))
        self.image_actions.extend([self.ui.actionDifference,
                                   self.ui.actionSet_diff_images_path, self.ui.actionSet_golden_images_path])

    def _init_M6_actions(self):
        self.ui.actionMedian_filter_r_1.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.median_filter, 1))
        self.ui.actionMedian_filter_r_2.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.median_filter, 2))
        self.ui.actionMedian_filter_r_3.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.median_filter, 3))
        self.image_actions.extend([self.ui.actionMedian_filter_r_1, self.ui.actionMedian_filter_r_2,
                                   self.ui.actionMedian_filter_r_3])

    def _enable_menu_items(self, mode):
        for action in self.image_actions:
            action.setEnabled(mode)

    def _image_editor_wrapper(self, editor_func, *args):
        self.image_history.add_new_state(self.np_img)
        self.image_editor.update_image(self.np_img)
        self.np_img = editor_func(*args)
        self._show_np_image()
        self._enable_undo_redo()
        self._enable_menu_items(True)

    def _make_new_custom_filter(self):
        dialog = MakeNewCustomLinearFilterDialog(self.user_settings_dump_n_loader.get_filters().keys())
        if dialog.exec_():
            matrix, divisor, name = dialog.get_values()
            self._image_editor_wrapper(self.image_editor.linear_filter, matrix, divisor)
            self.user_settings_dump_n_loader.save_filter(matrix, divisor, name)
            action = QtGui.QAction(self)
            action.setText(name)
            self.ui.menuCustom_filters.addAction(action)
            action.triggered.connect(lambda: self._image_editor_wrapper(self.image_editor.linear_filter,
                                     matrix, divisor))

    def _get_two_params_and_edit(self, param1, param2, editor_func):
        dialog = BaseTwoParamsDialog(param1, param2)
        if dialog.exec_():
            param1_value, param2_value = dialog.get_values()
            self._image_editor_wrapper(editor_func, param1_value, param2_value)

    @staticmethod
    def _choose_directory():
        return str(QtGui.QFileDialog.getExistingDirectory(None, 'Choose directory', '.', QtGui.QFileDialog.ShowDirsOnly))

    def _set_golden_images_dir(self, dirname):
        if dirname:
            self.golden_images_dir = dirname
            self.user_settings_dump_n_loader.set_golden_images_dir(dirname)

    def _set_diff_images_dir(self, dirname):
        if dirname:
            self.golden_images_dir = dirname
            self.user_settings_dump_n_loader.set_diff_images_dir(dirname)

    def _diff_images(self):
        self.image_editor.update_image(self.np_img)
        basename = os.path.basename(str(self.image_saver.filename))
        try:
            golden_np_img = scipy.misc.imread(os.path.join(str(self.golden_images_dir), basename))
            is_ok, diff_img, percentage = self.image_editor.diff_images(golden_np_img)
            if not is_ok:
                QtGui.QMessageBox.about(self, 'Error', 'Images have different size.s')
            else:
                QtGui.QMessageBox.about(self, 'Diff percentage', 'diff: %.1f %%' % percentage)
                diff_filename = os.path.join(str(self.diff_images_dir), basename)
                if percentage:
                    self.image_saver.save_any(diff_img, diff_filename)
        except IOError:
            QtGui.QMessageBox.about(self, 'Error', 'There is no golden image with corresponding name.')

    def _open_file(self):
        self.np_img = self.image_opener.open_file()
        self.image_history.reset()
        self.image_history.add_new_state(self.np_img)
        self.image_saver.filename = self.image_opener.filename

        # in case of png files swap channels
        if self.np_img.shape[2] == 4:
            self.np_img = utils.bgra2rgba(self.np_img)
        self._show_np_image()

        self._enable_menu_items(True)
        self._enable_undo_redo()
        self.ui.actionSave_As.setEnabled(True)

    def _save_file(self):
        self.image_saver.np_img = self.np_img
        self.image_saver.save_file()
        self.ui.actionSave.setEnabled(False)

    def _undo(self):
        if self.image_history.can_undo():
            self.np_img = self.image_history.undo()
            self._show_np_image()
            self._enable_undo_redo()

    def _redo(self):
        if self.image_history.can_redo():
            self.np_img = self.image_history.redo()
            self._show_np_image()
            self._enable_undo_redo()

    def _enable_undo_redo(self):
        undo_state, redo_state = True, True
        if not self.image_history.can_redo():
            redo_state = False
        if not self.image_history.can_undo():
            undo_state = False
        self.ui.actionRedo.setEnabled(redo_state)
        self.ui.actionUndo.setEnabled(undo_state)

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