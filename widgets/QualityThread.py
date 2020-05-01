from PyQt5.QtCore import QThread
from widgets.ESRGAN import SRGAN


class QualityThread(QThread):
    def __init__(self):
        super().__init__()
        pass

    def run(self):
        RDDB = SRGAN(training_mode=False)
        RDDB.generator.load_weights(r'./models/DIV2K_gan.h5')

        print(">> Creating the ESRGAN network")
        gan = SRGAN(training_mode=False, refer_model=RDDB.generator)
        gan.generator.load_weights(r'./models/DIV2K_generator_4X_epoch65000.h5')
        try:
            gan.test(datapath_test='./images/inputs', log_test_path="./images/images")
        except Exception as e:
            print(e)
