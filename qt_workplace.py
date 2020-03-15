import sys

from PIL import Image
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QDir, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QMainWindow

import cv2
import numpy as np
from matplotlib import pyplot


class StyleThread(QThread):

    def __init__(self, iterations=10, channels=3, image_size=500,
                 content_weight=0.02, style_weight=4.5, total_variation_weight=0.995,
                 total_variation_loss_factor=1.25):
        super().__init__()
        self.iterations = iterations
        self.channels = channels
        self.image_size = image_size
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight
        self.total_variation_loss_factor = total_variation_loss_factor

    def run(self):
        from style import Style
        style = Style()
        style.set_parameters(self.iterations, self.channels, self.image_size,
                             self.content_weight, self.style_weight, self.total_variation_weight,
                             self.total_variation_loss_factor)
        style.train()


class WorkspaceWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.i = 1
        self.pixmap = -1
        self.image = None
        self.thread = None
        self.iterations = 4
        self.filter_style = 0
        self.title = "Workplace"
        self.mode = "none"
        self.processed = 0
        self.InitUI()

    def InitUI(self):
        uic.loadUi('ui/workplace.ui', self)
        self.btn_filter_style.clicked.connect(
            self.btn_ftr_style)
        self.btn_filter_blur.clicked.connect(
            self.btn_ftr_blur)
        self.btn_filter_noise.clicked.connect(
            self.btn_ftr_noise)
        self.btn_filter_grayscale.clicked.connect(
            self.btn_ftr_grayscale)
        self.btn_filter_threshold.clicked.connect(
            self.btn_ftr_threshold)
        self.btn_sh.clicked.connect(self.show_img)
        self.btn_apply.clicked.connect(self.apply)
        self.actionOpen.setShortcut('Ctrl+O')
        self.actionOpen.triggered.connect(self.act_open)
        self.actionSave.setShortcut('Ctrl+S')
        self.actionSave.triggered.connect(self.act_save)
        self.actionSave_as.setShortcut('Ctrl+Shift+S')
        self.actionSave_as.triggered.connect(self.act_save_as)
        self.show()

    @pyqtSlot()
    def btn_ftr_style(self):
        self.mode = "style"
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image",
                                                  QDir.homePath())
        i = Image.open(fileName)
        i.save("images/images/style.png")
        self.thread = StyleThread(self.iterations)

    def btn_ftr_blur(self):
        self.mode = "blur"

    def btn_ftr_noise(self):
        self.mode = "noise"

    def btn_ftr_grayscale(self):
        self.mode = "grayscale"

    def btn_ftr_threshold(self):
        self.mode = "threshold"

    def apply(self):
        if self.mode == "style":
            self.ftr_style()
        elif self.mode == "blur":
            self.ftr_blur()
        elif self.mode == "noise":
            self.ftr_noise()
        elif self.mode == "grayscale":
            self.ftr_grayscale()
        elif self.mode == "threshold":
            self.ftr_threshold()

    def show_img(self):
        try:
            self.image = cv2.imread("images/images/stylized/stylized{}.png".format(self.iterations))
            self.i += 1
            cv2.imwrite("images/images/input{}.png".format(self.i), self.image)
            self.put_image()
            self.processed = 0
        except Exception as e:
            print("WAIT", e)

    def act_open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image",
                                                  QDir.homePath())
        if fileName != '':
            self.pixmap = QPixmap(fileName)
            self.img.setPixmap(self.pixmap)
            i = Image.open(fileName)
            i.save("images/images/input{}.png".format(self.i))
            self.image = cv2.imread("images/images/input{}.png".format(self.i))
            self.hist_bgr()
            self.hist_pixmap = QPixmap("images/images/hist{}.jpg".format(self.i))
            self.img_hist.setPixmap(self.hist_pixmap)

    def act_save(self):
        pass

    def act_save_as(self):
        pass

    def cv_img_open(self):
        if self.image is None:
            self.image = cv2.imread("images/images/input{}.png".format(self.iterations))
            self.hist_bgr()
            self.hist_pixmap = QPixmap("images/images/hist{}.jpg".format(self.i))
            self.img_hist.setPixmap(self.hist_pixmap)

    def put_image(self):
        self.hist_bgr()
        self.pixmap = QPixmap("images/images/input{}.png".format(self.i))
        self.hist_pixmap = QPixmap("images/images/hist{}.jpg".format(self.i))
        self.img.setPixmap(self.pixmap)
        self.img_hist.setPixmap(self.hist_pixmap)

    def ftr_style(self):
        if self.thread is not None:
            self.thread.start()
            self.processed = 1

    def ftr_blur(self, par=3):
        self.cv_img_open()
        self.i += 1
        self.image = cv2.GaussianBlur(self.image, (par, par), 0)
        cv2.imwrite("images/images/input{}.png".format(self.i), self.image)
        self.put_image()

    def ftr_noise(self, par=0):
        self.cv_img_open()
        self.i += 1
        dst = np.empty_like(self.image)
        noise = cv2.randn(dst, (0, 0, 0), (20, 20, 20))
        self.image = cv2.addWeighted(self.image, 0.5, noise, 0.5, 30)
        cv2.imwrite("images/images/input{}.png".format(self.i), self.image)
        self.put_image()

    def ftr_grayscale(self, par=0):
        self.cv_img_open()
        self.i += 1
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("images/images/input{}.png".format(self.i), self.image)
        self.put_image()

    def ftr_threshold(self, par=0):
        self.cv_img_open()
        self.i += 1
        self.image = cv2.threshold(self.image, 127, 255, par)[1]
        cv2.imwrite("images/images/input{}.png".format(self.i), self.image)
        self.put_image()

    def hist_plot(self, plot_1, plot_2, plot_3):
        fig, ax = pyplot.subplots()
        ax.plot(plot_1, color='b')
        ax.plot(plot_2, color='g')
        ax.plot(plot_3, color='r')
        pyplot.savefig("images/images/hist{}.jpg".format(self.i))

    def hist_bgr(self, par=0):
        self.cv_img_open()
        bgr = ('b', 'g', 'r')
        a = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        b = cv2.calcHist([self.image], [1], None, [256], [0, 256])
        c = cv2.calcHist([self.image], [2], None, [256], [0, 256])
        self.hist_plot(a, b, c)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WorkspaceWidget()
    sys.exit(app.exec_())
