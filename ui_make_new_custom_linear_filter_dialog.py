# -*- coding: utf-8 -*-
from PyQt4 import QtCore, QtGui

# TODO: refactor magic numbers

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


class MatrixWidget(object):
    def __init__(self, widget, elems, dialog_height):
        self.widget = widget
        self.elems = elems
        self.dialog_height_param = dialog_height


class Ui_Dialog(object):

    def setupUi(self, Dialog):
        Dialog.resize(380, 220)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(260, 20, 91, 61))
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBox.sizePolicy().hasHeightForWidth())
        self.buttonBox.setSizePolicy(sizePolicy)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.layoutWidget = QtGui.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 189, 74))
        self.formLayout = QtGui.QFormLayout(self.layoutWidget)
        self.formLayout.setMargin(0)
        self.nameLabel = QtGui.QLabel(self.layoutWidget)
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.nameLabel)
        self.nameEdit = QtGui.QLineEdit(self.layoutWidget)
        self.nameEdit.setText("New Filter")
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.nameEdit)
        self.dimensionLabel = QtGui.QLabel(self.layoutWidget)
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.dimensionLabel)
        self.dimensionSpinner = QtGui.QSpinBox(self.layoutWidget)
        self.dimensionSpinner.setMinimum(3)
        self.dimensionSpinner.setMaximum(5)
        self.dimensionSpinner.setSingleStep(2)
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.dimensionSpinner)
        self.divisorLabel = QtGui.QLabel(self.layoutWidget)
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.divisorLabel)
        self.divisorSpinner = QtGui.QSpinBox(self.layoutWidget)
        self.divisorSpinner.setMinimum(1)
        self.divisorSpinner.setMaximum(100)
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.divisorSpinner)

        self.matrix5GridLayoutWidget = self.generate_matrix_layout(Dialog, 5)
        self.matrix5GridLayoutWidget.widget.hide()
        self.matrix3GridLayoutWidget = self.generate_matrix_layout(Dialog, 3)
        self.matrix_widget = self.matrix3GridLayoutWidget

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def change_matrix_layout(self, Dialog):
        dimension = int(self.dimensionSpinner.value())
        if dimension != 3 and dimension != 5:
            dimension = min(5, max(3, dimension))
            self.dimensionSpinner.setValue(dimension)
        if dimension == 3:
            self.matrix_widget = self.matrix3GridLayoutWidget
            self.matrix3GridLayoutWidget.widget.setVisible(True)
            self.matrix5GridLayoutWidget.widget.setVisible(False)
            Dialog.resize(Dialog.width(), self.matrix3GridLayoutWidget.dialog_height_param)
        elif dimension == 5:
            self.matrix_widget = self.matrix5GridLayoutWidget
            self.matrix5GridLayoutWidget.widget.setVisible(True)
            self.matrix3GridLayoutWidget.widget.setVisible(False)
            Dialog.resize(Dialog.width(), self.matrix5GridLayoutWidget.dialog_height_param)

    def generate_matrix_layout(self, Dialog, dimension):
        w = dimension * 60 + 5 * (dimension + 1) + 1
        h = 40 * dimension + 1
        matrixGridLayoutWidget = QtGui.QWidget(Dialog)
        matrixGridLayout = QtGui.QGridLayout(matrixGridLayoutWidget)
        matrixGridLayout.setMargin(0)
        matrixLabel = QtGui.QLabel(matrixGridLayoutWidget)
        matrixGridLayout.addWidget(matrixLabel, 0, 0, 1, 1)
        matrixGridLayoutWidget.setGeometry(QtCore.QRect(20, 90, w, h))

        matrixHorizontalLayout = QtGui.QHBoxLayout()
        matrix_elem_edits = []
        for i in xrange(dimension):
            v_layout = QtGui.QVBoxLayout()
            elems = []
            for j in xrange(dimension):
                elem = QtGui.QLineEdit(matrixGridLayoutWidget)
                elem.setText("0")
                v_layout.addWidget(elem)
                elems.append(elem)
            matrix_elem_edits.append(elems)
            matrixHorizontalLayout.addLayout(v_layout)
        matrixGridLayout.addLayout(matrixHorizontalLayout, 0, 2, 1, 1)
        matrixLabel.setText(_translate("Dialog", "Matrix" + " " * (90 / dimension), None))

        return MatrixWidget(matrixGridLayoutWidget, matrix_elem_edits, 220 + (dimension - 3) * 20)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.nameLabel.setText(_translate("Dialog", "Name", None))
        self.dimensionLabel.setText(_translate("Dialog", "Dimension", None))
        self.divisorLabel.setText(_translate("Dialog", "Divisor", None))

