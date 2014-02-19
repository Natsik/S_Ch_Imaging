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

    def fileOpen(self):
        filename = QtGui.QFileDialog.getOpenFileName(self.main_window, 'Open File...', '.', 'Image File(*.png *.jpg *.jpeg *.bmp);; All Files (*)')        
        if filename:
            self.load(filename)

    def load(self, filename):
        if not QtCore.QFile.exists(filename):
            return False

        fh = QtCore.QFile(filename)
        if not fh.open(QtCore.QFile.ReadOnly):
            return False

        pixmap = QtGui.QPixmap(filename)
        dest_height, dest_width = self.view.size().height(), self.view.size().width()
        # TODO: debug this
        if dest_height < pixmap.size().height():
            pixmap = pixmap.scaledToHeight(self.view.size().height())
        elif dest_width < pixmap.size().width():
            pixmap = pixmap.scaledToWidth(self.view.size().width())
        self.view.setPixmap(pixmap)        

        # img = mean(imread(filename), 2)
        return True


class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()              
        self.init_ui()        

        self.io = ImageOpener(self, self.ui.imageLabel)
        self.init_file_actions()

    def init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)      
        self.ui.imageLabel.setAlignment(QtCore.Qt.AlignCenter)  

    def init_file_actions(self):        
        self.ui.actionExit.setShortcut('Esc')
        self.ui.actionExit.triggered.connect(QtGui.qApp.quit)

        self.ui.actionOpen.setShortcut('Ctrl+O')        
        self.ui.actionOpen.triggered.connect(self.io.fileOpen)        

    def show_window(self):
        self.show()            


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mw = MainWindow()
    mw.show_window()
    sys.exit(app.exec_())