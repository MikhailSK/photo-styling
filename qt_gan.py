import sys

from PyQt5 import uic
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from gan import GAN


class Loader(QThread):
    def __init__(self):
        super().__init__()
        self.generator = "generator_4500.h5"
        self.discriminator = "discriminator_4500.h5"

    def run(self):
        gan = GAN()
        gan.load_model(self.generator, self.discriminator)
        i = 0
        gan.save_images(i)


class GANWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "GAN"
        self.i = 0
        self.InitUI()
        load = Loader()
        load.change_value.connect(self.put_image())
        load.start()
        self.pixmap = QPixmap('images/image_' + str(self.i) + '.png')
        self.InitUI()

    def InitUI(self):
        uic.loadUi('ui/gan.ui', self)
        self.btn_generate.clicked.connect(
            self.btn_generate_onClick)
        self.show()

    def btn_generate_onClick(self):
        self.i += 1
        self.gan.save_images(self.i)
        self.pixmap = QPixmap('images/image_' + str(self.i) + '.png')
        self.put_image()

    def put_image(self):
        self.img.setPixmap(self.pixmap)

    @staticmethod
    def btn_exit_onClick():
        sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GANWidget()
    sys.exit(app.exec_())
