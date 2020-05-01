import warnings

import cv2
import numpy as np
from PIL import Image
from keras import backend
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b

warnings.filterwarnings('ignore')


def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


class Style:
    def __init__(self):
        print("INIT")
        self.ITERATIONS = 10
        self.CHANNELS = 3
        self.IMAGE_WIDTH = 500
        self.IMAGE_HEIGHT = 500
        self.IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
        self.CONTENT_WEIGHT = 0.02
        self.STYLE_WEIGHT = 4.5
        self.TOTAL_VARIATION_WEIGHT = 0.995
        self.TOTAL_VARIATION_LOSS_FACTOR = 1.25
        self.name1 = "images/images/input1.png"
        self.name2 = "images/images/style.png"

    def set_parameters(self, iterations=10, height=500, weight=500,
                       content_weight=0.02, style_weight=4.5, total_variation_weight=0.995,
                       total_variation_loss_factor=1.25, name1="images/images/input1.png",
                       name2="images/images/style.png"):
        print("SET_PARAMETERS")

        self.ITERATIONS = iterations
        self.IMAGE_WIDTH = weight
        self.IMAGE_HEIGHT = height
        self.CONTENT_WEIGHT = content_weight
        self.STYLE_WEIGHT = style_weight
        self.TOTAL_VARIATION_WEIGHT = total_variation_weight
        self.TOTAL_VARIATION_LOSS_FACTOR = total_variation_loss_factor
        self.name1 = name1
        self.name2 = name2

    def open_image(self, name1, name2):

        self.input_image_path = name1
        self.style_image_path = name2

        self.output_image_path = "images/images/stylized/stylized{}.png".format(self.ITERATIONS)

        self.image = cv2.imread(name1)
        cv2.imwrite(name1, self.image)

        self.input_image = Image.open(self.input_image_path)
        self.input_image = self.input_image.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        self.input_image.save("images/images/input.png")

        self.style_image = Image.open(self.style_image_path)
        self.style_image = self.style_image.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        print(1)

        self.input_image_array = np.asarray(self.input_image, dtype="float32")
        self.input_image_array = np.expand_dims(self.input_image_array, axis=0)
        self.input_image_array[:, :, :, 0] -= self.IMAGENET_MEAN_RGB_VALUES[2]
        self.input_image_array[:, :, :, 1] -= self.IMAGENET_MEAN_RGB_VALUES[1]
        self.input_image_array[:, :, :, 2] -= self.IMAGENET_MEAN_RGB_VALUES[0]
        self.input_image_array = self.input_image_array[:, :, :, ::-1]

        self.style_image_array = np.asarray(self.style_image, dtype="float32")
        self.style_image_array = np.expand_dims(self.style_image_array, axis=0)
        self.style_image_array[:, :, :, 0] -= self.IMAGENET_MEAN_RGB_VALUES[2]
        self.style_image_array[:, :, :, 1] -= self.IMAGENET_MEAN_RGB_VALUES[1]
        self.style_image_array[:, :, :, 2] -= self.IMAGENET_MEAN_RGB_VALUES[0]
        self.style_image_array = self.style_image_array[:, :, :, ::-1]

        self.input_image = backend.variable(self.input_image_array)
        self.style_image = backend.variable(self.style_image_array)
        self.combination_image = backend.placeholder((1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3))

        print(self.input_image.shape)
        print(self.style_image.shape)
        print(self.combination_image.shape)

        print(2)
        self.input_tensor = backend.concatenate([self.input_image, self.style_image, self.combination_image], axis=0)
        self.model = VGG16(input_tensor=self.input_tensor, include_top=False)
        print(3)

    def compute_style_loss(self, style, combination):
        style = gram_matrix(style)
        combination = gram_matrix(combination)
        size = self.IMAGE_HEIGHT * self.IMAGE_WIDTH
        return backend.sum(backend.square(style - combination)) / (4. * (self.CHANNELS ** 2) * (size ** 2))

    def evaluate_loss_and_gradients(self, x):
        x = x.reshape((1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS))
        outs = backend.function([self.combination_image], self.outputs)([x])
        loss = outs[0]
        gradients = outs[1].flatten().astype("float64")
        return loss, gradients

    def total_variation_loss(self, x):
        a = backend.square(x[:, :self.IMAGE_HEIGHT - 1, :self.IMAGE_WIDTH - 1, :] - x[:, 1:, :self.IMAGE_WIDTH - 1, :])
        b = backend.square(x[:, :self.IMAGE_HEIGHT - 1, :self.IMAGE_WIDTH - 1, :] - x[:, :self.IMAGE_HEIGHT - 1, 1:, :])
        return backend.sum(backend.pow(a + b, self.TOTAL_VARIATION_LOSS_FACTOR))

    def loss(self, x):
        loss, gradients = self.evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients

    def train(self):

        print(4)
        print(self.name1)
        print(self.name2)

        self.open_image(self.name1, self.name2)
        print(5)

        layers = dict([(layer.name, layer.output) for layer in self.model.layers])
        print(6)

        content_layer = 'block2_conv2'
        layer_features = layers[content_layer]
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]

        loss = backend.variable(0.)
        loss = loss + self.CONTENT_WEIGHT * content_loss(content_image_features,
                                                         combination_features)

        style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
        for layer_name in style_layers:
            layer_features = layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            style_loss = self.compute_style_loss(style_features, combination_features)
            loss += (self.STYLE_WEIGHT / len(style_layers)) * style_loss

        loss += self.TOTAL_VARIATION_WEIGHT * self.total_variation_loss(self.combination_image)

        self.outputs = [loss]
        self.outputs += backend.gradients(loss, self.combination_image)

        x = np.random.uniform(0, 255, (1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)) - 128.

        print("starting really train for ", self.ITERATIONS)

        for i in range(self.ITERATIONS):
            x, loss, info = fmin_l_bfgs_b(self.loss, x.flatten(), fprime=self.gradients, maxfun=20)
            print("Iteration %d completed with loss %d" % (i, loss))

        x = x.reshape((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS))
        x = x[:, :, ::-1]
        x[:, :, 0] += self.IMAGENET_MEAN_RGB_VALUES[2]
        x[:, :, 1] += self.IMAGENET_MEAN_RGB_VALUES[1]
        x[:, :, 2] += self.IMAGENET_MEAN_RGB_VALUES[0]
        x = np.clip(x, 0, 255).astype("uint8")
        output_image = Image.fromarray(x)
        output_image.save(self.output_image_path)
