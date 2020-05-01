from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QSlider, QGridLayout


class StyleParWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.iterations = 10
        self.content_weight = 0.02
        self.style_weight = 4.5
        self.total_variation_weight = 0.995
        self.total_variation_loss_factor = 1.25

        self.hundred_style = 0
        self.val_style = 45
        self.hundred_content = 0
        self.val_content = 2
        self.hundred_weight = 9
        self.val_weight = 95
        self.hundred_loss = 1
        self.val_loss = 25

        self.font = QFont("Monaco")

        self.txt_iterations = QLabel()
        self.txt_content_weight = QLabel()
        self.txt_style_weight = QLabel()
        self.txt_total_variation_weight = QLabel()
        self.txt_total_variation_loss_factor = QLabel()

        self.txt_val_content_weight = QLabel()
        self.txt_val_style_weight = QLabel()
        self.txt_val_total_variation_weight = QLabel()
        self.txt_val_total_variation_loss_factor = QLabel()

        self.input_iterations = QLineEdit()
        self.input_content_weight = QSlider(Qt.Horizontal)
        self.input_content_weight_dop = QSlider(Qt.Horizontal)
        self.input_style_weight = QSlider(Qt.Horizontal)
        self.input_style_weight_dop = QSlider(Qt.Horizontal)
        self.input_total_variation_weight = QSlider(Qt.Horizontal)
        self.input_total_variation_weight_dop = QSlider(Qt.Horizontal)
        self.input_total_variation_loss_factor = QSlider(Qt.Horizontal)
        self.input_total_variation_loss_factor_dop = QSlider(Qt.Horizontal)

        self.grid = QGridLayout()

        self.initUI()

    def initUI(self):
        self.font.setPixelSize(16)

        self.txt_iterations.setText("iterations")
        self.txt_content_weight.setText("content_weight")
        self.txt_style_weight.setText("style_weight")
        self.txt_total_variation_weight.setText("total_variation_weight")
        self.txt_total_variation_loss_factor.setText("total_variation_loss_factor")

        self.txt_val_content_weight.setText("0.02")
        self.txt_val_style_weight.setText("5.0")
        self.txt_val_total_variation_weight.setText("0.995")
        self.txt_val_total_variation_loss_factor.setText("1.25")

        self.txt_val_content_weight.setAlignment(Qt.AlignRight)
        self.txt_val_style_weight.setAlignment(Qt.AlignRight)
        self.txt_val_total_variation_weight.setAlignment(Qt.AlignRight)
        self.txt_val_total_variation_loss_factor.setAlignment(Qt.AlignRight)

        self.txt_iterations.setFont(self.font)
        self.txt_content_weight.setFont(self.font)
        self.txt_style_weight.setFont(self.font)
        self.txt_total_variation_weight.setFont(self.font)
        self.txt_total_variation_loss_factor.setFont(self.font)

        self.txt_val_content_weight.setFont(self.font)
        self.txt_val_style_weight.setFont(self.font)
        self.txt_val_total_variation_weight.setFont(self.font)
        self.txt_val_total_variation_loss_factor.setFont(self.font)

        self.txt_val_content_weight.setFont(self.font)
        self.txt_val_style_weight.setFont(self.font)
        self.txt_val_total_variation_weight.setFont(self.font)
        self.txt_val_total_variation_loss_factor.setFont(self.font)

        self.input_iterations.textChanged[str].connect(self.txt_iterations_change)
        self.input_content_weight.valueChanged[int].connect(self.txt_content_weight_change)
        self.input_content_weight_dop.valueChanged[int].connect(self.txt_content_weight_change_dop)
        self.input_style_weight.valueChanged[int].connect(self.txt_style_weight_change)
        self.input_style_weight_dop.valueChanged[int].connect(self.txt_style_weight_change_dop)
        self.input_total_variation_weight.valueChanged[int].connect(self.txt_total_variation_weight_change)
        self.input_total_variation_weight_dop.valueChanged[int].connect(self.txt_total_variation_weight_change_dop)
        self.input_total_variation_loss_factor.valueChanged[int].connect(self.txt_total_variation_loss_factor_change)
        self.input_total_variation_loss_factor_dop.valueChanged[int].connect(
            self.txt_total_variation_loss_factor_change_dop)

        self.input_iterations.setText("10")

        self.input_content_weight.setSliderPosition(self.val_content)
        self.input_content_weight.setPageStep(5)
        self.input_content_weight.setMinimum(1)

        self.input_content_weight_dop.setSliderPosition(self.hundred_content)
        self.input_content_weight_dop.setPageStep(3)
        self.input_content_weight_dop.setMaximum(29)

        self.input_style_weight.setSliderPosition(self.val_style)
        self.input_style_weight.setPageStep(5)
        self.input_style_weight.setMinimum(1)

        self.input_style_weight_dop.setSliderPosition(self.hundred_style)
        self.input_style_weight_dop.setPageStep(3)
        self.input_style_weight_dop.setMaximum(29)

        self.input_total_variation_weight.setSliderPosition(self.val_weight)
        self.input_total_variation_weight.setPageStep(5)
        self.input_total_variation_weight_dop.setMinimum(1)

        self.input_total_variation_weight_dop.setSliderPosition(9)
        self.input_total_variation_weight_dop.setPageStep(3)
        self.input_total_variation_weight_dop.setMaximum(29)

        self.input_total_variation_loss_factor.setSliderPosition(self.val_loss)
        self.input_total_variation_loss_factor.setPageStep(3)
        self.input_total_variation_loss_factor.setMinimum(1)

        self.input_total_variation_loss_factor_dop.setSliderPosition(self.hundred_loss)
        self.input_total_variation_loss_factor_dop.setPageStep(5)
        self.input_total_variation_loss_factor_dop.setMaximum(29)

        self.grid.addWidget(self.txt_iterations, 0, 0, 1, 6)
        self.grid.addWidget(self.input_iterations, 0, 7, 1, 2)

        self.grid.addWidget(self.txt_content_weight, 1, 0, 1, 5)
        self.grid.addWidget(self.txt_val_content_weight, 1, 6, 1, 3)
        self.grid.addWidget(self.input_content_weight_dop, 2, 0, 1, 9)
        self.grid.addWidget(self.input_content_weight, 3, 0, 1, 9)

        self.grid.addWidget(self.txt_style_weight, 4, 0, 1, 5)
        self.grid.addWidget(self.txt_val_style_weight, 4, 6, 1, 3)
        self.grid.addWidget(self.input_style_weight_dop, 5, 0, 1, 9)
        self.grid.addWidget(self.input_style_weight, 6, 0, 1, 9)

        self.grid.addWidget(self.txt_total_variation_weight, 7, 0, 1, 5)
        self.grid.addWidget(self.txt_val_total_variation_weight, 7, 6, 1, 3)
        self.grid.addWidget(self.input_total_variation_weight_dop, 8, 0, 1, 9)
        self.grid.addWidget(self.input_total_variation_weight, 9, 0, 1, 9)

        self.grid.addWidget(self.txt_total_variation_loss_factor, 10, 0, 1, 5)
        self.grid.addWidget(self.txt_val_total_variation_loss_factor, 10, 6, 1, 3)
        self.grid.addWidget(self.input_total_variation_loss_factor_dop, 11, 0, 1, 9)
        self.grid.addWidget(self.input_total_variation_loss_factor, 12, 0, 1, 9)

        self.setLayout(self.grid)

    def txt_iterations_change(self, text):
        try:
            self.iterations = int(self.input_iterations.text())
        except Exception as e:
            print(e)
            self.input_iterations.setPlaceholderText("error INT")
            self.input_iterations.setText("")

    def txt_style_weight_change(self, value):
        self.val_style = value
        self.put_style_weight()

    def txt_style_weight_change_dop(self, value):
        self.hundred_style = value * 100
        self.put_style_weight()

    def put_style_weight(self):
        self.style_weight = float(self.hundred_style + self.val_style) / 10
        self.txt_val_style_weight.setText(str(self.style_weight))

    def txt_content_weight_change(self, value):
        self.val_content = value
        self.put_content_weight()

    def txt_content_weight_change_dop(self, value):
        self.hundred_content = value * 100
        self.put_content_weight()

    def put_content_weight(self):
        self.content_weight = float(self.hundred_content + self.val_content) / 100
        self.txt_val_content_weight.setText(str(self.content_weight))

    def txt_total_variation_weight_change(self, value):
        self.val_weight = value
        self.put_total_variation_weight()

    def txt_total_variation_weight_change_dop(self, value):
        self.hundred_weight = value * 100
        self.put_total_variation_weight()

    def put_total_variation_weight(self):
        self.total_variation_weight = float(self.hundred_weight + self.val_weight) / 1000
        self.txt_val_total_variation_weight.setText(str(self.total_variation_weight))

    def txt_total_variation_loss_factor_change(self, value):
        self.val_loss = value
        self.put_total_variation_loss_factor()

    def txt_total_variation_loss_factor_change_dop(self, value):
        self.hundred_loss = value * 100
        self.put_total_variation_loss_factor()

    def put_total_variation_loss_factor(self):
        self.total_variation_loss_factor = float(self.hundred_loss + self.val_loss) / 100
        self.txt_val_total_variation_loss_factor.setText(str(self.total_variation_loss_factor))
