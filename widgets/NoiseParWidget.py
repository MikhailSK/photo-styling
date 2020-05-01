from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QSlider


class NoiseParWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.r1 = 0
        self.g1 = 0
        self.b1 = 0
        self.r2 = 20
        self.g2 = 20
        self.b2 = 20
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 30

        self.font = QFont("Monaco")

        self.txt_rgb1 = QLabel()
        self.txt_rgb2 = QLabel()
        self.txt_alpha = QLabel()
        self.txt_beta = QLabel()
        self.txt_gamma = QLabel()

        self.input_r1 = QSlider(Qt.Horizontal)
        self.input_g1 = QSlider(Qt.Horizontal)
        self.input_b1 = QSlider(Qt.Horizontal)
        self.input_r2 = QSlider(Qt.Horizontal)
        self.input_g2 = QSlider(Qt.Horizontal)
        self.input_b2 = QSlider(Qt.Horizontal)
        self.input_alpha = QSlider(Qt.Horizontal)
        self.input_beta = QSlider(Qt.Horizontal)
        self.input_gamma = QSlider(Qt.Horizontal)

        self.txt_val_rgb1 = QLabel()
        self.txt_val_rgb2 = QLabel()
        self.txt_val_alpha = QLabel()
        self.txt_val_beta = QLabel()
        self.txt_val_gamma = QLabel()

        self.grid = QGridLayout()

        self.initUI()

    def initUI(self):
        self.font.setPixelSize(16)

        self.txt_rgb1.setText("rgb first")
        self.txt_rgb2.setText("rgb second")
        self.txt_alpha.setText("alpha")
        self.txt_beta.setText("beta")
        self.txt_gamma.setText("gamma")

        self.txt_val_rgb1.setText("({};{};{})".format(self.r1, self.g1, self.b1))
        self.txt_val_rgb2.setText("({};{};{})".format(self.r2, self.g2, self.b2))
        self.txt_val_alpha.setText(str(self.alpha))
        self.txt_val_beta.setText(str(self.beta))
        self.txt_val_gamma.setText(str(self.gamma))

        self.txt_val_rgb1.setAlignment(Qt.AlignRight)
        self.txt_val_rgb2.setAlignment(Qt.AlignRight)
        self.txt_val_alpha.setAlignment(Qt.AlignRight)
        self.txt_val_beta.setAlignment(Qt.AlignRight)
        self.txt_val_gamma.setAlignment(Qt.AlignRight)

        self.txt_rgb1.setFont(self.font)
        self.txt_rgb2.setFont(self.font)
        self.txt_alpha.setFont(self.font)
        self.txt_beta.setFont(self.font)
        self.txt_gamma.setFont(self.font)

        self.txt_val_rgb1.setFont(self.font)
        self.txt_val_rgb2.setFont(self.font)
        self.txt_val_alpha.setFont(self.font)
        self.txt_val_beta.setFont(self.font)
        self.txt_val_gamma.setFont(self.font)

        self.txt_val_rgb1.setFont(self.font)
        self.txt_val_rgb2.setFont(self.font)
        self.txt_val_alpha.setFont(self.font)
        self.txt_val_beta.setFont(self.font)
        self.txt_val_gamma.setFont(self.font)

        self.input_r1.valueChanged[int].connect(self.txt_r1_change)
        self.input_g1.valueChanged[int].connect(self.txt_g1_change)
        self.input_b1.valueChanged[int].connect(self.txt_b1_change)
        self.input_r2.valueChanged[int].connect(self.txt_r2_change)
        self.input_g2.valueChanged[int].connect(self.txt_g2_change)
        self.input_b2.valueChanged[int].connect(self.txt_b2_change)
        self.input_alpha.valueChanged[int].connect(self.txt_alpha_change)
        self.input_beta.valueChanged[int].connect(self.txt_beta_change)
        self.input_gamma.valueChanged[int].connect(self.txt_gamma_change)

        self.input_r1.setSliderPosition(self.r1)
        self.input_r1.setPageStep(3)
        self.input_r1.setMaximum(256)

        self.input_g1.setSliderPosition(self.g1)
        self.input_g1.setPageStep(3)
        self.input_g1.setMaximum(256)

        self.input_b1.setSliderPosition(self.b1)
        self.input_b1.setPageStep(3)
        self.input_b1.setMaximum(256)

        self.input_r2.setSliderPosition(self.r2)
        self.input_r2.setPageStep(3)
        self.input_r2.setMaximum(256)

        self.input_g2.setSliderPosition(self.g2)
        self.input_g2.setPageStep(3)
        self.input_g2.setMaximum(256)

        self.input_b2.setSliderPosition(self.b2)
        self.input_b2.setPageStep(3)
        self.input_b2.setMaximum(256)

        self.input_alpha.setSliderPosition(5)
        self.input_alpha.setPageStep(3)
        self.input_alpha.setMaximum(100)

        self.input_beta.setSliderPosition(5)
        self.input_beta.setPageStep(3)
        self.input_beta.setMaximum(100)

        self.input_gamma.setSliderPosition(30)
        self.input_gamma.setPageStep(3)
        self.input_gamma.setMaximum(100)

        self.grid.addWidget(self.txt_rgb1, 0, 0, 1, 5)
        self.grid.addWidget(self.txt_val_rgb1, 0, 6, 1, 3)
        self.grid.addWidget(self.input_r1, 1, 0, 1, 3)
        self.grid.addWidget(self.input_g1, 1, 3, 1, 3)
        self.grid.addWidget(self.input_b1, 1, 6, 1, 3)
        self.grid.addWidget(self.txt_rgb2, 2, 0, 1, 5)
        self.grid.addWidget(self.txt_val_rgb2, 2, 6, 1, 3)
        self.grid.addWidget(self.input_r2, 3, 0, 1, 3)
        self.grid.addWidget(self.input_g2, 3, 3, 1, 3)
        self.grid.addWidget(self.input_b2, 3, 6, 1, 3)
        self.grid.addWidget(self.txt_alpha, 4, 0, 1, 5)
        self.grid.addWidget(self.txt_val_alpha, 4, 6, 1, 3)
        self.grid.addWidget(self.input_alpha, 5, 0, 1, 9)
        self.grid.addWidget(self.txt_beta, 6, 0, 1, 5)
        self.grid.addWidget(self.txt_val_beta, 6, 6, 1, 3)
        self.grid.addWidget(self.input_beta, 7, 0, 1, 9)
        self.grid.addWidget(self.txt_gamma, 8, 0, 1, 5)
        self.grid.addWidget(self.txt_val_gamma, 8, 6, 1, 3)
        self.grid.addWidget(self.input_gamma, 9, 0, 1, 9)

        self.setLayout(self.grid)

    def txt_r1_change(self, value):
        self.r1 = value
        self.txt_val_rgb1.setText("({};{};{})".format(self.r1, self.g1, self.b1))

    def txt_g1_change(self, value):
        self.g1 = value
        self.txt_val_rgb1.setText("({};{};{})".format(self.r1, self.g1, self.b1))

    def txt_b1_change(self, value):
        self.b1 = value
        self.txt_val_rgb1.setText("({};{};{})".format(self.r1, self.g1, self.b1))

    def txt_r2_change(self, value):
        self.r2 = value
        self.txt_val_rgb2.setText("({};{};{})".format(self.r2, self.g2, self.b2))

    def txt_g2_change(self, value):
        self.g2 = value
        self.txt_val_rgb2.setText("({};{};{})".format(self.r2, self.g2, self.b2))

    def txt_b2_change(self, value):
        self.b2 = value
        self.txt_val_rgb2.setText("({};{};{})".format(self.r2, self.g2, self.b2))

    def txt_alpha_change(self, value):
        self.alpha = value / 10
        self.txt_val_alpha.setText(str(self.alpha))

    def txt_beta_change(self, value):
        self.beta = value / 10
        self.txt_val_beta.setText(str(self.beta))

    def txt_gamma_change(self, value):
        self.gamma = value
        self.txt_val_gamma.setText(str(self.gamma))
