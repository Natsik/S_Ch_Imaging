__author__ = 'aynroot'

import numpy as np

from PyQt4 import QtGui
from ui_make_new_custom_linear_filter_dialog import Ui_Dialog


class MakeNewCustomLinearFilterDialog(QtGui.QDialog):
    def __init__(self):
        super(MakeNewCustomLinearFilterDialog, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self._init_triggers()

    def _init_triggers(self):
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)
        self.ui.dimensionSpinner.valueChanged.connect(lambda: self.ui.change_matrix_layout(self))

    def get_values(self):
        dimension = len(self.ui.matrix_widget.elems)
        elem_values = []
        for column_elems in self.ui.matrix_widget.elems:
            for elem in column_elems:
                elem_values.append(int(elem.text()))
        matrix = np.array(elem_values).reshape(dimension, dimension).transpose()

        divisor = int(self.ui.divisorSpinner.value())
        name = self.ui.nameEdit.text()
        return matrix, divisor, name