from PyQt5.QtCore import QThread


class StyleThread(QThread):
    def __init__(self, iterations=10, content_weight=0.02, style_weight=4.5, total_variation_weight=0.995,
                 total_variation_loss_factor=1.25, height=500, weight=500, name1="images/images/input1.png",
                 name2="images/images/style.png"):
        super().__init__()

        self.iterations = iterations
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight
        self.total_variation_loss_factor = total_variation_loss_factor
        self.height = height
        self.weight = weight
        self.name1 = name1
        self.name2 = name2

    def run(self):
        from widgets.style import Style
        style = Style()
        style.set_parameters(self.iterations, self.height, self.weight,
                             self.content_weight, self.style_weight, self.total_variation_weight,
                             self.total_variation_loss_factor, self.name1, self.name2)
        style.train()
