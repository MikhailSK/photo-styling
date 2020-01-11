import sys

from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow


class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Main menu"
        self.cams = -1
        self.InitUI()

    def InitUI(self):
        uic.loadUi('ui/main.ui', self)
        self.btn_exit.clicked.connect(
            self.btn_exit_onClick)
        self.btn_gan.clicked.connect(
            self.btn_gan_onClick)
        self.btn_open.clicked.connect(
            self.btn_open_onClick)
        self.show()

    @pyqtSlot()
    def btn_gan_onClick(self):
        from qt_gan import GANWidget

        self.cams = GANWidget()
        self.cams.show()
        self.close()

    def btn_open_onClick(self):
        from qt_workplace import WorkspaceWidget

        self.cams = WorkspaceWidget()
        self.cams.show()
        self.close()

    @staticmethod
    def btn_exit_onClick():
        sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWidget()
    sys.exit(app.exec_())
