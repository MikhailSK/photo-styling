import sys

from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from gan import GAN


class Loader(QThread):
    change_value = pyqtSignal(int)

    def __init__(self, par):
        super().__init__()
        self.generator = "generator_4500.h5"
        self.discriminator = "discriminator_4500.h5"
        self.par = par
        self.gan = None

    def get_gan(self):
        return self.gan

    def run(self):
        if self.par == 0:
            self.gan = GAN()
            self.gan.load_model(self.generator, self.discriminator)
            self.par += 1
        i = 0
        self.gan.save_images(i)
        self.change_value.emit(i)


class GANWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "GAN"
        self.i = 0
        self.load = Loader(0)
        self.load.start()
        self.InitUI()

    def InitUI(self):
        uic.loadUi('ui/gan.ui', self)
        self.btn_generate.clicked.connect(
            self.btn_generate_onClick)
        self.put_image()
        self.show()

    def btn_generate_onClick(self):
        self.load = Loader(0)
        self.load.start()
        self.put_image()

    def put_image(self):
        self.pixmap = QPixmap('images/image_' + str(self.i) + '.png')
        self.img.setPixmap(self.pixmap)

    @staticmethod
    def btn_exit_onClick():
        sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GANWidget()
    sys.exit(app.exec_())
