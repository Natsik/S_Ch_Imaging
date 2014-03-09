__author__ = 'aynroot'

import numpy as np

from PyQt4 import QtGui
from ui_make_new_custom_linear_filter_dialog import Ui_Dialog


class MakeNewCustomLinearFilterDialog(QtGui.QDialog):
    def __init__(self, existing_names):
        super(MakeNewCustomLinearFilterDialog, self).__init__()
        self.ui = Ui_Dialog()
        self.existing_names = existing_names
        self.ui.setupUi(self, self._get_new_filter_name())
        self._init_triggers()
        self.update_existing_names = []

    def _get_new_filter_name(self):
        name = "New Filter"
        while name in self.existing_names:
            try:
                parts = name.split('_')
                n = int(parts[-1]) + 1
            except:
                n = 1
            name += '_%d' % n
        return name

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
        self.existing_names.append(name)
        return matrix, divisor, name