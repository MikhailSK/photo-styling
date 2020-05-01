from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QSlider


class BlurParWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.kernel = 1

        self.txt_kernel = QLabel()

        self.input_kernel = QSlider(Qt.Horizontal)

        self.txt_val_kernel = QLabel()

        self.grid = QGridLayout()

        self.font = QFont("Monaco")

        self.initUI()

    def initUI(self):
        self.font.setPixelSize(16)

        self.txt_kernel.setText("kernel size")

        self.txt_val_kernel.setText("3")

        self.txt_val_kernel.setAlignment(Qt.AlignRight)

        self.txt_kernel.setFont(self.font)

        self.txt_val_kernel.setFont(self.font)

        self.txt_val_kernel.setFont(self.font)

        self.input_kernel.valueChanged[int].connect(self.txt_kernel_change)

        self.input_kernel.setSliderPosition(self.kernel)
        self.input_kernel.setPageStep(2)
        self.input_kernel.setMinimum(1)
        self.input_kernel.setMaximum(20)
        self.input_kernel.setSingleStep(2)
        self.input_kernel.setSliderPosition(3)

        self.grid.addWidget(self.txt_kernel, 0, 0, 1, 5)
        self.grid.addWidget(self.txt_val_kernel, 0, 6, 1, 3)
        self.grid.addWidget(self.input_kernel, 1, 0, 1, 9)

        self.setLayout(self.grid)

    def txt_kernel_change(self, value):
        self.kernel = value * 2 + 1
        self.txt_val_kernel.setText(str(self.kernel))
