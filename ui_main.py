# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created: Mon Mar 10 12:38:18 2014
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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(863, 557)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.imageLabel = QtGui.QLabel(self.centralwidget)
        self.imageLabel.setText(_fromUtf8(""))
        self.imageLabel.setObjectName(_fromUtf8("imageLabel"))
        self.gridLayout.addWidget(self.imageLabel, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 863, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))
        self.menuFilters = QtGui.QMenu(self.menubar)
        self.menuFilters.setObjectName(_fromUtf8("menuFilters"))
        self.menuM3 = QtGui.QMenu(self.menubar)
        self.menuM3.setObjectName(_fromUtf8("menuM3"))
        self.menuCustom_filters = QtGui.QMenu(self.menuM3)
        self.menuCustom_filters.setObjectName(_fromUtf8("menuCustom_filters"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtGui.QAction(MainWindow)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.actionSave = QtGui.QAction(MainWindow)
        self.actionSave.setObjectName(_fromUtf8("actionSave"))
        self.actionSave_As = QtGui.QAction(MainWindow)
        self.actionSave_As.setObjectName(_fromUtf8("actionSave_As"))
        self.actionAbout = QtGui.QAction(MainWindow)
        self.actionAbout.setObjectName(_fromUtf8("actionAbout"))
        self.actionErosion = QtGui.QAction(MainWindow)
        self.actionErosion.setObjectName(_fromUtf8("actionErosion"))
        self.actionDilatation = QtGui.QAction(MainWindow)
        self.actionDilatation.setObjectName(_fromUtf8("actionDilatation"))
        self.actionInversion = QtGui.QAction(MainWindow)
        self.actionInversion.setObjectName(_fromUtf8("actionInversion"))
        self.actionTest = QtGui.QAction(MainWindow)
        self.actionTest.setObjectName(_fromUtf8("actionTest"))
        self.actionAny_standard_filter = QtGui.QAction(MainWindow)
        self.actionAny_standard_filter.setObjectName(_fromUtf8("actionAny_standard_filter"))
        self.actionMake_new_custom_filter = QtGui.QAction(MainWindow)
        self.actionMake_new_custom_filter.setObjectName(_fromUtf8("actionMake_new_custom_filter"))
        self.actionLinear_filter = QtGui.QAction(MainWindow)
        self.actionLinear_filter.setObjectName(_fromUtf8("actionLinear_filter"))
        self.actionNoize_filter = QtGui.QAction(MainWindow)
        self.actionNoize_filter.setObjectName(_fromUtf8("actionNoize_filter"))
        self.actionWhite_noize = QtGui.QAction(MainWindow)
        self.actionWhite_noize.setObjectName(_fromUtf8("actionWhite_noize"))
        self.actionMedian_filter = QtGui.QAction(MainWindow)
        self.actionMedian_filter.setObjectName(_fromUtf8("actionMedian_filter"))
        self.actionGauss_filter = QtGui.QAction(MainWindow)
        self.actionGauss_filter.setObjectName(_fromUtf8("actionGauss_filter"))
        self.actionLow_pass_filter = QtGui.QAction(MainWindow)
        self.actionLow_pass_filter.setObjectName(_fromUtf8("actionLow_pass_filter"))
        self.actionIntegrating_filter = QtGui.QAction(MainWindow)
        self.actionIntegrating_filter.setObjectName(_fromUtf8("actionIntegrating_filter"))
        self.actionBlur = QtGui.QAction(MainWindow)
        self.actionBlur.setObjectName(_fromUtf8("actionBlur"))
        self.actionSharpen = QtGui.QAction(MainWindow)
        self.actionSharpen.setObjectName(_fromUtf8("actionSharpen"))
        self.actionUndo = QtGui.QAction(MainWindow)
        self.actionUndo.setObjectName(_fromUtf8("actionUndo"))
        self.actionRedo = QtGui.QAction(MainWindow)
        self.actionRedo.setObjectName(_fromUtf8("actionRedo"))
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionUndo)
        self.menuFile.addAction(self.actionRedo)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionAbout)
        self.menuFilters.addAction(self.actionErosion)
        self.menuFilters.addAction(self.actionDilatation)
        self.menuFilters.addAction(self.actionInversion)
        self.menuCustom_filters.addSeparator()
        self.menuCustom_filters.addAction(self.actionMake_new_custom_filter)
        self.menuM3.addAction(self.menuCustom_filters.menuAction())
        self.menuM3.addAction(self.actionIntegrating_filter)
        self.menuM3.addAction(self.actionBlur)
        self.menuM3.addAction(self.actionSharpen)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuFilters.menuAction())
        self.menubar.addAction(self.menuM3.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Viewer", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuHelp.setTitle(_translate("MainWindow", "Help", None))
        self.menuFilters.setTitle(_translate("MainWindow", "M2", None))
        self.menuM3.setTitle(_translate("MainWindow", "M3", None))
        self.menuCustom_filters.setTitle(_translate("MainWindow", "Custom filters", None))
        self.actionOpen.setText(_translate("MainWindow", "Open...", None))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        self.actionExit.setShortcut(_translate("MainWindow", "Esc", None))
        self.actionSave.setText(_translate("MainWindow", "Save", None))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S", None))
        self.actionSave_As.setText(_translate("MainWindow", "Save As...", None))
        self.actionSave_As.setShortcut(_translate("MainWindow", "Ctrl+Shift+S", None))
        self.actionAbout.setText(_translate("MainWindow", "About", None))
        self.actionAbout.setShortcut(_translate("MainWindow", "F1", None))
        self.actionErosion.setText(_translate("MainWindow", "Erosion", None))
        self.actionErosion.setShortcut(_translate("MainWindow", "Ctrl+E", None))
        self.actionDilatation.setText(_translate("MainWindow", "Dilatation", None))
        self.actionDilatation.setShortcut(_translate("MainWindow", "Ctrl+D", None))
        self.actionInversion.setText(_translate("MainWindow", "Inversion", None))
        self.actionInversion.setShortcut(_translate("MainWindow", "Ctrl+I", None))
        self.actionTest.setText(_translate("MainWindow", "test", None))
        self.actionTest.setShortcut(_translate("MainWindow", "Ctrl+T", None))
        self.actionAny_standard_filter.setText(_translate("MainWindow", "Any standard filter", None))
        self.actionMake_new_custom_filter.setText(_translate("MainWindow", "Make new custom filter...", None))
        self.actionLinear_filter.setText(_translate("MainWindow", "Linear filter", None))
        self.actionNoize_filter.setText(_translate("MainWindow", "Noize filter", None))
        self.actionWhite_noize.setText(_translate("MainWindow", "White noize", None))
        self.actionMedian_filter.setText(_translate("MainWindow", "Median filter", None))
        self.actionGauss_filter.setText(_translate("MainWindow", "Gauss filter", None))
        self.actionLow_pass_filter.setText(_translate("MainWindow", "Low-pass filter", None))
        self.actionIntegrating_filter.setText(_translate("MainWindow", "Integrating filter", None))
        self.actionBlur.setText(_translate("MainWindow", "Blur", None))
        self.actionSharpen.setText(_translate("MainWindow", "Sharpen", None))
        self.actionUndo.setText(_translate("MainWindow", "Undo", None))
        self.actionUndo.setShortcut(_translate("MainWindow", "Ctrl+Z", None))
        self.actionRedo.setText(_translate("MainWindow", "Redo", None))
        self.actionRedo.setShortcut(_translate("MainWindow", "Ctrl+Shift+Z", None))

