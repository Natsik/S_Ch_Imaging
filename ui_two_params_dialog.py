# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'two_params_dialog.ui'
#
# Created: Sun Mar 16 21:28:26 2014
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

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

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(332, 88)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(230, 20, 81, 241))
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.widget = QtGui.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(20, 20, 187, 48))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.formLayout = QtGui.QFormLayout(self.widget)
        self.formLayout.setMargin(0)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.labelParam1 = QtGui.QLabel(self.widget)
        self.labelParam1.setObjectName(_fromUtf8("labelParam1"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.labelParam1)
        self.lineEditParam1 = QtGui.QLineEdit(self.widget)
        self.lineEditParam1.setObjectName(_fromUtf8("lineEditParam1"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.lineEditParam1)
        self.labelParam2 = QtGui.QLabel(self.widget)
        self.labelParam2.setObjectName(_fromUtf8("labelParam2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.labelParam2)
        self.lineEditParam2 = QtGui.QLineEdit(self.widget)
        self.lineEditParam2.setObjectName(_fromUtf8("lineEditParam2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.lineEditParam2)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.labelParam1.setText(_translate("Dialog", "TextLabel", None))
        self.labelParam2.setText(_translate("Dialog", "TextLabel", None))

