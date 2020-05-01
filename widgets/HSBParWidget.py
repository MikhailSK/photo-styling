from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QLabel, QSlider, QGridLayout


class HSBParWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.hue = 0
        self.saturation = 0
        self.brightness = 0

        self.font = QFont("Monaco")

        self.txt_hue = QLabel()
        self.txt_saturation = QLabel()
        self.txt_brightness = QLabel()

        self.input_hue = QSlider(Qt.Horizontal)
        self.input_saturation = QSlider(Qt.Horizontal)
        self.input_brightness = QSlider(Qt.Horizontal)

        self.txt_val_hue = QLabel()
        self.txt_val_saturation = QLabel()
        self.txt_val_brightness = QLabel()

        self.grid = QGridLayout()

        self.initUI()

    def initUI(self):
        self.font.setPixelSize(16)

        self.txt_hue.setText("hue")
        self.txt_saturation.setText("saturation")
        self.txt_brightness.setText("brightness")

        self.txt_val_hue.setText("0")
        self.txt_val_saturation.setText("0")
        self.txt_val_brightness.setText("0")

        self.txt_val_hue.setAlignment(Qt.AlignRight)
        self.txt_val_saturation.setAlignment(Qt.AlignRight)
        self.txt_val_brightness.setAlignment(Qt.AlignRight)

        self.txt_hue.setFont(self.font)
        self.txt_saturation.setFont(self.font)
        self.txt_brightness.setFont(self.font)

        self.txt_val_hue.setFont(self.font)
        self.txt_val_saturation.setFont(self.font)
        self.txt_val_brightness.setFont(self.font)

        self.txt_val_hue.setFont(self.font)
        self.txt_val_saturation.setFont(self.font)
        self.txt_val_brightness.setFont(self.font)

        self.input_hue.valueChanged[int].connect(self.txt_hue_change)
        self.input_saturation.valueChanged[int].connect(self.txt_saturation_change)
        self.input_brightness.valueChanged[int].connect(self.txt_brightness_change)

        self.input_hue.setPageStep(2)
        self.input_hue.setMinimum(0)
        self.input_hue.setMaximum(100)
        self.input_hue.setSingleStep(2)
        self.input_hue.setSliderPosition(self.hue)

        self.input_saturation.setPageStep(2)
        self.input_saturation.setMinimum(0)
        self.input_saturation.setMaximum(100)
        self.input_saturation.setSingleStep(2)
        self.input_saturation.setSliderPosition(self.saturation)

        self.input_brightness.setPageStep(2)
        self.input_brightness.setMinimum(0)
        self.input_brightness.setMaximum(100)
        self.input_brightness.setSingleStep(2)
        self.input_brightness.setSliderPosition(self.brightness)

        self.grid.addWidget(self.txt_hue, 0, 0, 1, 5)
        self.grid.addWidget(self.txt_val_hue, 0, 6, 1, 3)
        self.grid.addWidget(self.input_hue, 1, 0, 1, 9)

        self.grid.addWidget(self.txt_saturation, 2, 0, 1, 5)
        self.grid.addWidget(self.txt_val_saturation, 2, 6, 1, 3)
        self.grid.addWidget(self.input_saturation, 3, 0, 1, 9)

        self.grid.addWidget(self.txt_brightness, 4, 0, 1, 5)
        self.grid.addWidget(self.txt_val_brightness, 4, 6, 1, 3)
        self.grid.addWidget(self.input_brightness, 5, 0, 1, 9)

        self.setLayout(self.grid)

    def txt_hue_change(self, value):
        self.hue = value
        self.txt_val_hue.setText(str(self.hue))

    def txt_saturation_change(self, value):
        self.saturation = value
        self.txt_val_saturation.setText(str(self.saturation))

    def txt_brightness_change(self, value):
        self.brightness = value
        self.txt_val_brightness.setText(str(self.brightness))
