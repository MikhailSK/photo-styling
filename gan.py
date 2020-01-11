import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

import warnings
import keras
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Conv2DTranspose

warnings.filterwarnings('ignore')


class GAN:
    def __init__(self):
        self.img_shape = (64, 64, 3)

        self.noise_size = 100

        optimizer = Adam(0.0001, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined = Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.discriminator)

        self.discriminator.trainable = False

        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined.summary()

    # Creating the generator, the large kernels in the convolutional layers allow the network to create complex
    # structures.
    def build_generator(self):
        epsilon = 0.00001  # Small float added to variance to avoid dividing by zero in the BatchNorm layers.
        noise_shape = (self.noise_size,)

        model = Sequential()

        model.add(Dense(4 * 4 * 512, activation='linear', input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 512)))

        model.add(Conv2DTranspose(512, kernel_size=[5, 5], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(256, kernel_size=[5, 5], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=[5, 5], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(64, kernel_size=[5, 5], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3, kernel_size=[5, 5], strides=[1, 1], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))

        # Standard activation for the generator of a GAN
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_images=10, save_model=2000):

        X_train = np.array(images)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)

        mean_d_loss = [0, 0]
        mean_g_loss = 0

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.noise_size))
            gen_imgs = self.generator.predict(noise)

            # Training the discriminator

            # The loss of the discriminator is the mean of the losses while training on authentic and fake images
            d_loss = 0.5 * np.add(self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1))),
                                  self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1))))

            # Training the generator

            noise = np.random.normal(0, 1, (batch_size, self.noise_size))

            valid_y = np.array([1] * batch_size)
            g_loss = self.combined.train_on_batch(noise, valid_y)

            mean_d_loss[0] += d_loss[0]
            mean_d_loss[1] += d_loss[1]
            mean_g_loss += g_loss

            # We print the losses and accuracy of the networks every 10
            # batches mainly to make surethe accuracy of the discriminator
            # is not stable at around 50% or 100% (which would mean
            # the discriminator performs not well enough or too well)
            if epoch % 1 == 0:
                print("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" %
                      (epoch, mean_d_loss[0] / 200, mean_d_loss[1] / 2, mean_g_loss / 200))

                mean_d_loss = [0, 0]
                mean_g_loss = 0

            if epoch % save_images == 0:
                self.save_images(epoch)

            # We save the architecture of the model, the weights and the state of the optimizer
            # This way we can restart the training exactly where we stopped
            if epoch % save_model == 0:
                self.save_model(epoch)

    # Saving n generated images to have a representation of the spectrum of images created by the generator
    def save_images(self, epoch):
        noise = np.random.normal(0, 1, (25, self.noise_size))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(5, 5, figsize=(8, 8))

        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs[5 * i + j])
                axs[i, j].axis('off')

        fig.savefig("images/image_%d.png" % epoch)

    def save_model(self, epoch):
        self.generator.save("generator_%d.h5" % epoch)
        self.discriminator.save("discriminator_%d.h5" % epoch)

    def load_model(self, name1, name2):
        self.generator = load_model("models/" + name1)
        self.discriminator = load_model("models/" + name2)
