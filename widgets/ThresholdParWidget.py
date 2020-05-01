from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QSlider


class ThresholdParWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.lower = 127
        self.upper = 256
        self.par = 0

        self.font = QFont("Monaco")

        self.txt_lower = QLabel()
        self.txt_upper = QLabel()
        self.txt_par = QLabel()

        self.input_lower = QSlider(Qt.Horizontal)
        self.input_upper = QSlider(Qt.Horizontal)
        self.input_par = QSlider(Qt.Horizontal)

        self.txt_val_lower = QLabel()
        self.txt_val_upper = QLabel()
        self.txt_val_par = QLabel()

        self.grid = QGridLayout()

        self.initUI()

    def initUI(self):
        self.font.setPixelSize(16)

        self.txt_lower.setText("lower")
        self.txt_upper.setText("upper")

        self.txt_val_lower.setText(str(self.lower))
        self.txt_val_upper.setText(str(self.upper))

        self.txt_val_lower.setAlignment(Qt.AlignRight)
        self.txt_val_upper.setAlignment(Qt.AlignRight)

        self.txt_lower.setFont(self.font)
        self.txt_upper.setFont(self.font)

        self.txt_val_lower.setFont(self.font)
        self.txt_val_upper.setFont(self.font)

        self.txt_val_lower.setFont(self.font)
        self.txt_val_upper.setFont(self.font)

        self.input_lower.valueChanged[int].connect(self.txt_lower_change)
        self.input_upper.valueChanged[int].connect(self.txt_upper_change)

        self.input_lower.setSliderPosition(self.lower)
        self.input_lower.setPageStep(3)
        self.input_lower.setMinimum(1)
        self.input_lower.setMaximum(254)

        self.input_upper.setSliderPosition(self.upper)
        self.input_upper.setPageStep(3)
        self.input_upper.setMinimum(self.lower)
        self.input_upper.setMaximum(256)

        self.grid.addWidget(self.txt_lower, 0, 0, 1, 5)
        self.grid.addWidget(self.txt_val_lower, 0, 6, 1, 3)
        self.grid.addWidget(self.input_lower, 1, 0, 1, 9)

        self.grid.addWidget(self.txt_upper, 2, 0, 1, 5)
        self.grid.addWidget(self.txt_val_upper, 2, 6, 1, 3)
        self.grid.addWidget(self.input_upper, 3, 0, 1, 9)

        self.setLayout(self.grid)

    def txt_lower_change(self, value):
        self.lower = value
        self.input_upper.setMinimum(self.lower + 1)
        self.txt_val_lower.setText(str(self.lower))

    def txt_upper_change(self, value):
        self.upper = value
        self.txt_val_upper.setText(str(self.upper))

