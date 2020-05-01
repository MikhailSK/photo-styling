from qt_workplace import *


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.title = "ArtRunBox"
        self.height = 540
        self.width = 960
        self.left = 200
        self.top = 50
        self.cams = -1

        self.btn_exit = QPushButton()
        self.btn_gan = QPushButton()
        self.btn_open = QPushButton()
        self.frame = QFrame()
        self.frame1 = QFrame()
        self.main_frame = QFrame()
        self.esc = QShortcut(QtGui.QKeySequence('Esc'), self)

        self.text = QLabel()
        self.img = QLabel()

        self.grid = QGridLayout()

        self.font_1 = QFont("Monaco")
        self.font_btn = QFont("Monaco")

        self.InitUI()

    def InitUI(self):
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QtGui.QIcon("resources/icons/icon.jpg"))
        self.setWindowTitle(self.title)
        self.setStyleSheet(open("stylesheets/default.qss", "r").read())

        self.font_1.setPixelSize(40)
        self.font_btn.setPixelSize(20)

        self.img.setPixmap(QPixmap('resources/image.jpg'))
        self.img.setScaledContents(True)

        self.main_frame.setObjectName("background")
        self.main_frame.setVisible(False)

        self.text.setText(self.title)
        self.text.setFont(self.font_1)
        self.text.setAlignment(Qt.AlignCenter)
        self.text.setObjectName("title")

        self.btn_gan.setFont(self.font_btn)
        self.btn_open.setFont(self.font_btn)
        self.btn_exit.setFont(self.font_btn)

        self.btn_exit.clicked.connect(
            self.btn_exit_onClick)
        self.btn_gan.clicked.connect(
            self.btn_gan_onClick)
        self.btn_open.clicked.connect(
            self.btn_open_onClick)

        self.btn_gan.setText("GAN")
        self.btn_gan.setVisible(False)
        self.btn_open.setText("Open")
        self.btn_exit.setText("Exit")
        self.esc.activated.connect(self.act_esc)

        self.grid.addWidget(self.frame1, 0, 0, 16, 14)
        self.grid.addWidget(self.frame, 0, 14, 16, 16)
        self.grid.addWidget(self.text, 3, 2, 2, 10)
        self.grid.addWidget(self.btn_open, 5, 2, 2, 10)
        self.grid.addWidget(self.btn_gan, 7, 2, 2, 10)
        self.grid.addWidget(self.btn_exit, 10, 2, 2, 10)
        self.grid.addWidget(self.img, 1, 15, 14, 14)
        self.grid.addWidget(self.main_frame, 0, 0, 16, 30)

        self.grid.setSpacing(1)
        self.grid.setContentsMargins(1, 1, 1, 1)

        self.setLayout(self.grid)

        self.show()

    @pyqtSlot()
    def btn_gan_onClick(self):
        from qt_gan import GANWidget

        self.cams = GANWidget()
        self.cams.show()
        self.close()

    def btn_open_onClick(self):
        self.main_frame.setVisible(True)
        self.workspace = WorkspaceWidget()
        self.setGeometry(self.workspace.left, self.workspace.top,
                         self.workspace.width, self.workspace.height)
        self.grid.addWidget(self.workspace, 0, 0, 16, 30)

    @staticmethod
    def btn_exit_onClick():
        sys.exit(app.exec_())

    def act_esc(self):
        self.workspace.deleteLater()
        self.main_frame.setVisible(False)
        self.setGeometry(self.left, self.top,
                         self.width, self.height)
        self.workspace = WorkspaceWidget


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWidget()
    sys.exit(app.exec_())
