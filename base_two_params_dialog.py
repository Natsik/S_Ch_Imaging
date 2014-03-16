__author__ = 'aynroot'
from PyQt4 import QtGui
from ui_two_params_dialog import Ui_Dialog


class BaseTwoParamsDialog(QtGui.QDialog):
    def __init__(self, param1, param2):
        super(BaseTwoParamsDialog, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self._set_names(param1, param2)

        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)

    def _set_names(self, param1, param2):
        self.ui.labelParam1.setText(param1)
        self.ui.labelParam2.setText(param2)

    def get_values(self):
        param1_value = self.ui.lineEditParam1.text()
        param2_value = self.ui.lineEditParam2.text()

        return param1_value, param2_value