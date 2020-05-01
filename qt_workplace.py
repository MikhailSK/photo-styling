import sys

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import pyqtSlot, QDir, Qt
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QGridLayout, QPushButton, QMenuBar, QAction, \
    QWidget, QFrame, QGraphicsScene, QGraphicsView, QShortcut
from matplotlib import pyplot
from qtconsole.qt import QtGui

from widgets.BlurParWidget import BlurParWidget
from widgets.GrayscaleParWidget import GrayscaleParWidget
from widgets.NoiseParWidget import NoiseParWidget
from widgets.StyleParWidget import StyleParWidget
from widgets.StyleThread import StyleThread
from widgets.ThresholdParWidget import ThresholdParWidget
from widgets.HSBParWidget import HSBParWidget
from widgets.HQParWidget import HQParWidget
from widgets.QualityThread import QualityThread


class WorkspaceWidget(QWidget):
    def __init__(self):
        super().__init__()
        print("init")

        self.par = 0
        self.height = 1016
        self.width = 1465
        self.left = 200
        self.top = 50
        self.image_height = 500
        self.image_width = 500
        self.height_append = 20
        self.width_append = 20
        self.sh_height = 400
        self.sh_width = 400
        self.wp = 9

        self.bar_menu = QMenuBar()
        self.actionOpen = QAction(QIcon("icons/ic_open.png"), "&Open", self)
        self.actionSave = QAction(QIcon("icons/ic_save.png"), "&Save", self)
        self.actionSave_as = QAction(QIcon("icons/ic_save_as.png"), "&Save As", self)
        self.m_open = self.bar_menu.addMenu("&Open")
        self.m_save = self.bar_menu.addMenu("&Save")
        self.m_save_as = self.bar_menu.addMenu("&Save As")

        self.font_1 = QFont("Monaco")

        self.frame = QFrame()
        self.grid = QGridLayout()
        self.img = QGraphicsScene()
        self.view = QGraphicsView()
        self.img_hist = QLabel(self)
        self.img_hist.setObjectName("img_hist")

        self.btn_filter_style = QPushButton("Style")
        self.btn_filter_blur = QPushButton("Blur")
        self.btn_filter_noise = QPushButton("Noise")
        self.btn_filter_grayscale = QPushButton("Grayscale")
        self.btn_filter_threshold = QPushButton("Threshold")
        self.btn_filter_hsb = QPushButton("HSB")
        self.btn_filter_HQ = QPushButton("HQ (Demo)")
        self.btn_sh = QPushButton("Show")
        self.btn_apply = QPushButton("Apply")

        self.style_par_widget = StyleParWidget()
        self.blur_par_widget = BlurParWidget()
        self.noise_par_widget = NoiseParWidget()
        self.grayscale_par_widget = GrayscaleParWidget()
        self.threshold_par_widget = ThresholdParWidget()
        self.hsb_par_widget = HSBParWidget()
        self.HQ_par_widget = HQParWidget()

        self.resize_up = QShortcut(QtGui.QKeySequence('Ctrl+='), self)
        self.resize_up.activated.connect(self.act_resize_up)

        self.resize_down = QShortcut(QtGui.QKeySequence('Ctrl+-'), self)
        self.resize_down.activated.connect(self.act_resize_down)

        self.cancel = QShortcut(QtGui.QKeySequence('Ctrl+z'), self)
        self.cancel.activated.connect(self.act_cancel)

        self.enter = QShortcut(QtGui.QKeySequence(Qt.Key_Space), self)
        self.enter.activated.connect(self.apply)

        self.i = 1
        self.pixmap = -1
        self.image = None
        self.thread = None
        self.iterations = 7
        self.filter_style = 0
        self.title = "ArtRunBox"
        self.mode = "none"
        self.mode_prev = "none"
        self.processed = 0
        self.save_path = "none"
        self.save_name = "/image.png"

        self.InitUI()

    def InitUI(self):
        print("initUI")
        self.setWindowIcon(QtGui.QIcon("resources/icons/icon.jpg"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet(open("stylesheets/default.qss", "r").read())

        self.img_hist.setScaledContents(True)
        self.img_hist.setContentsMargins(0, 0, 0, 5)

        self.view.setAlignment(Qt.AlignCenter)
        self.img_hist.setAlignment(Qt.AlignCenter)

        self.m_open.addAction(self.actionOpen)
        self.m_save.addAction(self.actionSave)
        self.m_save_as.addAction(self.actionSave_as)

        self.grid.addWidget(self.view, 0, 0, 16, 10)
        self.grid.addWidget(self.img_hist, 0, 11, 4, 5)

        self.grid.addWidget(self.btn_filter_style, 4, 11, 1, 2)
        self.grid.addWidget(self.btn_filter_blur, 5, 11, 1, 2)
        self.grid.addWidget(self.btn_filter_noise, 6, 11, 1, 2)
        self.grid.addWidget(self.btn_filter_HQ, 7, 11, 1, 2)
        self.grid.addWidget(self.btn_filter_grayscale, 4, 14, 1, 2)
        self.grid.addWidget(self.btn_filter_threshold, 5, 14, 1, 2)
        self.grid.addWidget(self.btn_filter_hsb, 6, 14, 1, 2)
        self.grid.addWidget(self.btn_sh, 14, 11, 1, 5)
        self.grid.addWidget(self.btn_apply, 15, 11, 1, 5)

        self.grid.setSpacing(1)
        self.grid.setContentsMargins(1, 1, 1, 1)

        self.grid.setMenuBar(self.bar_menu)
        self.setLayout(self.grid)

        self.font_1.setPixelSize(23)

        self.btn_filter_style.setFont(self.font_1)
        self.btn_filter_blur.setFont(self.font_1)
        self.btn_filter_noise.setFont(self.font_1)
        self.btn_filter_threshold.setFont(self.font_1)
        self.btn_filter_grayscale.setFont(self.font_1)
        self.btn_filter_hsb.setFont(self.font_1)
        self.btn_filter_HQ.setFont(self.font_1)
        self.btn_sh.setFont(self.font_1)
        self.btn_apply.setFont(self.font_1)

        self.btn_sh.setVisible(False)

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
        self.btn_filter_hsb.clicked.connect(
            self.btn_ftr_hsb)
        self.btn_filter_HQ.clicked.connect(self.btn_ftr_HQ)
        self.btn_sh.clicked.connect(self.show_img)
        self.btn_apply.clicked.connect(self.apply)

        self.actionOpen.setShortcut('Ctrl+O')
        self.actionOpen.triggered.connect(self.act_open)
        self.actionSave.setShortcut('Ctrl+S')
        self.actionSave.triggered.connect(self.act_save)
        self.actionSave_as.setShortcut('Ctrl+Shift+S')
        self.actionSave_as.triggered.connect(self.act_save_as)

        self.show()

    def cv_img_open(self):
        print("cv_img_open")
        if self.image is None or self.mode == "grayscale" or self.par == 1:
            self.par = 0
            self.image = cv2.imread("images/images/input{}.png".format(self.i-1))
            self.hist_bgr()
            self.hist_pixmap = QPixmap("images/images/hist.jpg")
            self.img_hist.setPixmap(self.hist_pixmap)

    def put_image(self):
        print("put_image")
        if self.mode != "grayscale" or self.par == 0:
            self.hist_bgr()
        self.put_resized_image()
        self.hist_pixmap = QPixmap("images/images/hist.jpg")
        self.img_hist.setPixmap(self.hist_pixmap)

    def clear_space(self):
        print("clear_space")
        print(self.mode_prev, self.mode)
        if self.mode != "style":
            try:
                self.btn_sh.setVisible(False)
                self.style_par_widget.deleteLater()
                self.frame.deleteLater()
            except Exception as e:
                print(e)
        if self.mode != "blur":
            try:
                self.blur_par_widget.deleteLater()
                self.frame.deleteLater()
            except Exception as e:
                print(e)
        if self.mode != "noise":
            try:
                self.noise_par_widget.deleteLater()
                self.frame.deleteLater()
            except Exception as e:
                print(e)
        if self.mode != "grayscale":
            try:
                self.grayscale_par_widget.deleteLater()
                self.frame.deleteLater()
            except Exception as e:
                print(e)
        if self.mode != "threshold":
            try:
                self.threshold_par_widget.deleteLater()
                self.frame.deleteLater()
            except Exception as e:
                print(e)
        if self.mode != "hsb":
            try:
                self.hsb_par_widget.deleteLater()
                self.frame.deleteLater()
            except Exception as e:
                print(e)

        self.style_par_widget = StyleParWidget()
        self.blur_par_widget = BlurParWidget()
        self.noise_par_widget = NoiseParWidget()
        self.grayscale_par_widget = GrayscaleParWidget()
        self.threshold_par_widget = ThresholdParWidget()
        self.hsb_par_widget = HSBParWidget()
        self.frame = QFrame()

    def apply(self):
        print("apply")
        if self.processed == 0:
            if self.mode != "none":
                self.i += 1
                if self.mode == "style":
                    self.btn_sh.setVisible(True)
                    self.iterations = self.style_par_widget.iterations
                    self.ftr_style(self.style_par_widget.iterations, self.image_height, self.image_width,
                                   self.style_par_widget.content_weight,
                                   self.style_par_widget.style_weight, self.style_par_widget.total_variation_weight,
                                   self.style_par_widget.total_variation_loss_factor,
                                   "images/images/input{}.png".format(self.i - 1), "images/images/style.png")
                elif self.mode == "HQ":
                    self.btn_sh.setVisible(True)
                    self.ftr_HQ()
                else:
                    self.cv_img_open()
                    if self.mode == "blur":
                        self.ftr_blur(self.blur_par_widget.kernel)
                    elif self.mode == "noise":
                        self.ftr_noise(self.noise_par_widget.r1, self.noise_par_widget.g1, self.noise_par_widget.b1,
                                       self.noise_par_widget.r2, self.noise_par_widget.g2, self.noise_par_widget.b2,
                                       self.noise_par_widget.alpha, self.noise_par_widget.beta,
                                       self.noise_par_widget.gamma)
                    elif self.mode == "grayscale":
                        self.ftr_grayscale()
                    elif self.mode == "threshold":
                        self.ftr_threshold(self.threshold_par_widget.lower, self.threshold_par_widget.upper,
                                           self.threshold_par_widget.par)
                    elif self.mode == "hsb":
                        self.ftr_hsb(self.hsb_par_widget.hue, self.hsb_par_widget.saturation,
                                     self.hsb_par_widget.brightness)
                    self.put_image()
        else:
            print("PROCESSED")

    @pyqtSlot()
    def btn_ftr_style(self):
        print("btn_ftr_style")
        if self.processed == 0:
            try:
                fileName, _ = QFileDialog.getOpenFileName(self, "Open Image",
                                                          QDir.homePath())
                self.mode_prev = self.mode
                self.mode = "style"
                i = Image.open(fileName)
                i.save("images/images/style.png")
                if self.mode_prev != "style" or self.mode_prev == "none":
                    self.clear_space()
                    self.grid.addWidget(self.frame, self.wp, 11, 4, 5)
                    self.grid.addWidget(self.style_par_widget, self.wp, 11, 4, 5)
            except Exception as e:
                print(e)
        else:
            print("Processed")

    def btn_ftr_blur(self):
        print("btn_ftr_blur")
        if self.processed == 0:
            self.mode_prev = self.mode
            self.mode = "blur"
            if self.mode_prev != "blur" or self.mode_prev == "none":
                self.clear_space()
                self.grid.addWidget(self.frame, self.wp, 11, 1, 5)
                self.grid.addWidget(self.blur_par_widget, self.wp, 11, 1, 5)
        else:
            print("Processed")

    def btn_ftr_noise(self):
        print("btn_ftr_noise")
        if self.processed == 0:
            self.mode_prev = self.mode
            self.mode = "noise"
            if self.mode_prev != "noise" or self.mode_prev == "none":
                self.clear_space()
                self.grid.addWidget(self.frame, self.wp, 11, 4, 5)
                self.grid.addWidget(self.noise_par_widget, self.wp, 11, 4, 5)
        else:
            print("Processed")

    def btn_ftr_grayscale(self):
        print("btn_ftr_grayscale")
        if self.processed == 0:
            self.mode_prev = self.mode
            self.mode = "grayscale"
            if self.mode_prev != "grayscale" or self.mode_prev == "none":
                self.clear_space()
        else:
            print("Processed")

    def btn_ftr_threshold(self):
        print("btn_ftr_threshold")
        if self.processed == 0:
            self.mode_prev = self.mode
            self.mode = "threshold"
            if self.mode_prev != "threshold" or self.mode_prev == "none":
                self.clear_space()
                self.grid.addWidget(self.frame, self.wp, 11, 2, 5)
                self.grid.addWidget(self.threshold_par_widget, self.wp, 11, 2, 5)
        else:
            print("Processed")

    def btn_ftr_hsb(self):
        print("btn_ftr_hsb")
        if self.processed == 0:
            self.mode_prev = self.mode
            self.mode = "hsb"
            if self.mode_prev != "hsb" or self.mode_prev == "none":
                self.clear_space()
                self.grid.addWidget(self.frame, self.wp, 11, 3, 5)
                self.grid.addWidget(self.hsb_par_widget, self.wp, 11, 3, 5)
        else:
            print("Processed")

    def btn_ftr_HQ(self):
        print("btn_ftr_HQ")
        if self.processed == 0:
            self.mode_prev = self.mode
            self.mode = "HQ"
            if self.mode_prev != "hq" or self.mode_prev == "none":
                self.clear_space()
        else:
            print("Processed")

    def show_img(self):
        print("show_img")
        try:
            if self.mode == "style":
                print("-----style")
                self.image = cv2.imread("images/images/stylized/stylized{}.png".format(self.iterations))
            elif self.mode == "HQ":
                print("------HQ")
                self.image = cv2.imread("images/images/hq_image.png")
            cv2.imwrite("images/images/input{}.png".format(self.i), self.image)
            self.pixmap = QPixmap("images/images/input{}.png".format(self.i))

            self.image_height = self.pixmap.size().height()
            self.image_width = self.pixmap.size().width()

            self.sh_height = self.image_height
            self.sh_width = self.image_width

            self.height_append = int(self.image_height * 0.10)
            self.width_append = self.sh_width - ((self.sh_height - self.height_append) * self.sh_width) / self.sh_height

            self.put_image()
            self.processed = 0
        except Exception as e:
            print("WAIT", e)

    def change_image(self):
        print("change_image")
        self.img = QGraphicsScene()

        self.view = QGraphicsView()
        self.view.setAlignment(Qt.AlignCenter)

        self.grid.addWidget(self.view, 0, 0, 16, 10)

    def act_open(self):
        print("act_open")
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", QDir.homePath())
        if fileName != '':
            self.i = 1
            self.pixmap = QPixmap(fileName)

            self.image_height = self.pixmap.size().height()
            self.image_width = self.pixmap.size().width()

            self.sh_height = self.image_height
            self.sh_width = self.image_width

            self.height_append = int(self.image_height * 0.10)
            self.width_append = self.sh_width - ((self.sh_height - self.height_append) * self.sh_width) / self.sh_height

            self.change_image()

            self.img.addPixmap(self.pixmap)
            self.view.setScene(self.img)
            i = Image.open(fileName)
            i.save("images/images/input{}.png".format(self.i))
            self.image = cv2.imread("images/images/input{}.png".format(self.i))
            self.hist_bgr()
            self.hist_pixmap = QPixmap("images/images/hist.jpg")
            self.img_hist.setPixmap(self.hist_pixmap)

    def act_save(self):
        print("act_save")
        if self.processed == 0:
            try:
                if self.save_path == "none":
                    options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly
                    dir_cur = QDir.homePath()
                    directory = QFileDialog.getExistingDirectory(None, "Select Folder", dir_cur, options)
                    self.save_path = directory + self.save_name

                i = Image.open("images/images/input{}.png".format(self.i))
                try:
                    i.save(self.save_path)
                except Exception as e:
                    print(e)
                    self.save_path = "none"
                    self.act_save()
            except Exception as e:
                print(e)
        else:
            print("Wait")

    def act_save_as(self):
        print("act_save_as")
        if self.processed == 0:
            try:
                options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly
                dir_cur = QDir.homePath()
                directory = QFileDialog.getExistingDirectory(None, "Select Folder", dir_cur, options)
                self.save_path = directory + self.save_name
                i = Image.open("images/images/input{}.png".format(self.i))
                try:
                    i.save(self.save_path)
                except Exception as e:
                    print(e)
                    self.save_path = "none"
            except Exception as e:
                print(e)
        else:
            print("Wait")

    def act_resize_up(self):
        print("act_resize_up")
        self.sh_width = self.sh_width + int(self.width_append)
        self.sh_height = self.sh_height + self.height_append
        self.put_resized_image()

    def act_resize_down(self):
        print("act_resize_down")
        self.sh_width = self.sh_width - int(self.width_append)
        self.sh_height = self.sh_height - self.height_append
        self.put_resized_image()

    def act_cancel(self):
        print("act_cancel")
        if self.processed == 0:
            if self.i > 1:
                self.i -= 1
                self.mode = self.mode_prev
                self.mode_prev = "none"

                self.pixmap = QPixmap("images/images/input{}.png".format(self.i))

                self.image_height = self.pixmap.size().height()
                self.image_width = self.pixmap.size().width()

                self.sh_height = self.image_height
                self.sh_width = self.image_width

                self.height_append = int(self.image_height * 0.10)
                self.width_append = self.sh_width - (
                            (self.sh_height - self.height_append) * self.sh_width) / self.sh_height

                self.image = cv2.imread("images/images/input{}.png".format(self.i))
                self.put_image()

    def put_resized_image(self):
        print("put_resized_image")
        self.change_image()
        image = cv2.imread("images/images/input{}.png".format(self.i))
        print(self.sh_width, self.height)
        image = cv2.resize(image, (self.sh_width, self.sh_height))
        cv2.imwrite("images/images/1input.png", image)
        self.pixmap = QPixmap("images/images/1input.png")
        self.img.clear()
        self.img.addPixmap(self.pixmap)
        self.view.setScene(self.img)
        self.img_hist.setPixmap(self.hist_pixmap)

    def ftr_style(self, iterations, height, width, content_weight, style_weight, total_variation_weight,
                  total_variation_loss_factor,
                  name1, name2):
        print("ftr_style")

        self.thread = StyleThread(iterations, content_weight, style_weight, total_variation_weight,
                                  total_variation_loss_factor, height, width, name1, name2)
        self.thread.start()
        self.processed = 1

    def ftr_blur(self, par=3):
        print("ftr_blur")
        self.image = cv2.GaussianBlur(self.image, (par, par), 0)
        cv2.imwrite("images/images/input{}.png".format(self.i), self.image)

    def ftr_noise(self, r1=0, g1=0, b1=0, r2=20, g2=20, b2=20, alpha=0.5, beta=0.5, gamma=30):
        print("ftr_noise")
        dst = np.empty_like(self.image)
        noise = cv2.randn(dst, (r1, g1, b1), (r2, g2, b2))
        self.image = cv2.addWeighted(self.image, alpha, noise, beta, gamma)
        cv2.imwrite("images/images/input{}.png".format(self.i), self.image)

    def ftr_grayscale(self):
        print("ftr_grayscale")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("images/images/input{}.png".format(self.i), self.image)
        self.image = cv2.imread("images/images/input{}.png".format(self.i))

    def ftr_threshold(self, lower=127, upper=255, par=0):
        print("ftr_threshold")
        self.image = cv2.threshold(self.image, lower, upper, par)[1]
        cv2.imwrite("images/images/input{}.png".format(self.i), self.image)

    def ftr_hsb(self, hue=0, saturation=0, brightness=0):
        print("ftr_hsb")
        hsb = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        hsb[:, :, 0] += hue
        hsb[:, :, 1] += saturation
        hsb[:, :, 2] += brightness
        self.image = cv2.cvtColor(hsb, cv2.COLOR_HSV2RGB)
        cv2.imwrite("images/images/input{}.png".format(self.i), self.image)

    def ftr_HQ(self):
        print("ftr_HQ")
        img = cv2.imread("images/images/input{}.png".format(self.i-1))
        cv2.imwrite("images/inputs/input.png", img)
        self.thread = QualityThread()
        self.thread.start()
        self.processed = 1

    def hist_plot(self, plot_1, plot_2, plot_3):
        print("hist_plot")
        fig, ax = pyplot.subplots(1, 1, sharey=True, figsize=(5, 2))
        ax.plot(plot_1, color='b')
        ax.plot(plot_2, color='g')
        ax.plot(plot_3, color='r')
        fig.set_facecolor('#357E76')
        ax.set_facecolor('#92BCC4')
        pyplot.savefig("images/images/hist.jpg", facecolor=fig.get_facecolor())
        pyplot.close('all')

    def hist_bgr(self):
        print("hist_bgr")
        bgr = ('b', 'g', 'r')
        a = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        b = cv2.calcHist([self.image], [1], None, [256], [0, 256])
        c = cv2.calcHist([self.image], [2], None, [256], [0, 256])
        self.hist_plot(a, b, c)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WorkspaceWidget()
    sys.exit(app.exec_())
