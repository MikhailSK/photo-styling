import sys

from PIL import Image
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QDir, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QMainWindow


class StyleThread(QThread):

    def run(self):
        from style import Style
        style = Style()
        style.train()


class WorkspaceWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pixmap = -1
        self.thread = None
        self.filter_style = 0
        self.title = "Workplace"
        self.InitUI()

    def InitUI(self):
        uic.loadUi('ui/workplace.ui', self)
        self.btn_filter_style.clicked.connect(
            self.btn_filter_style_onClick)
        self.btn_apply.clicked.connect(self.apply)
        self.actionOpen.setShortcut('Ctrl+O')
        self.actionOpen.triggered.connect(self.act_open)
        self.actionSave.setShortcut('Ctrl+S')
        self.actionSave.triggered.connect(self.act_save)
        self.actionSave_as.setShortcut('Ctrl+Shift+S')
        self.actionSave_as.triggered.connect(self.act_save_as)
        self.show()

    @pyqtSlot()
    def btn_filter_style_onClick(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image",
                                                  QDir.homePath())
        i = Image.open(fileName)
        i.save("images/images/style.png")
        self.thread = StyleThread()

    def apply(self):
        if self.thread is not None:
            self.thread.start()

    def act_open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image",
                                                  QDir.homePath())
        if fileName != '':
            self.pixmap = QPixmap(fileName)
            self.img.setPixmap(self.pixmap)
            i = Image.open(fileName)
            i.save("images/images/input1.png")

    def act_save(self):
        pass

    def act_save_as(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WorkspaceWidget()
    sys.exit(app.exec_())
