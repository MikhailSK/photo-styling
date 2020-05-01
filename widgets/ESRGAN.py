import datetime
import os
import sys
import numpy as np

# Import keras + tensorflow without the "Using XX Backend" message
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Add, Concatenate
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense
from keras.layers import Lambda, Dropout
from keras.optimizers import Adam
from keras.applications.vgg19 import preprocess_input
from keras.utils.data_utils import OrderedEnqueuer
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

sys.stderr = stderr
from widgets.vgg19_noAct import VGG19
from widgets.util import DataLoader, plot_test_images, plot_bigger_images


class SRGAN():
    def __init__(self, height_lr=32, width_lr=32, channels=3, upscaling_factor=4, gen_lr=1e-4,
                 dis_lr=1e-4, training_mode=True, refer_model=None, ):

        # :param int height_lr: Height of low-resolution images
        # :param int width_lr: Width of low-resolution images
        # :param int channels: Image channels
        # :param int upscaling_factor: Up-scaling factor
        # :param int gen_lr: Learning rate of generator
        # :param int dis_lr: Learning rate of discriminator

        self.height_lr = height_lr
        self.width_lr = width_lr

        if upscaling_factor not in [2, 4, 8]:
            raise ValueError('Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        self.channels = channels
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        self.gen_lr = gen_lr
        self.dis_lr = dis_lr

        self.loss_weights = {'percept': 1e-3, 'gen': 5e-3, 'pixel': 1e-2}

        self.gan_loss = 'mse'
        self.dis_loss = 'binary_crossentropy'

        self.generator = self.build_generator()
        self.compile_generator(self.generator)
        self.refer_model = refer_model

        if training_mode:
            self.vgg = self.build_vgg()
            self.discriminator = self.build_discriminator()
            self.RaGAN = self.build_RaGAN()
            self.srgan = self.build_srgan()
            self.compile_vgg(self.vgg)

    def save_weights(self, filepath, e=None):
        self.generator.save_weights("{}_generator_{}X_epoch{}.h5".format(filepath, self.upscaling_factor, e))
        self.discriminator.save_weights("{}_discriminator_{}X_epoch{}.h5".format(filepath, self.upscaling_factor, e))

    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)

    def SubpixelConv2D(self, name, scale=2):
        def subpixel_shape(input_shape):
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            output_shape = tuple(dims)
            return output_shape

        def subpixel(x):
            return tf.depth_to_space(x, scale)

        return Lambda(subpixel, output_shape=subpixel_shape, name=name)

    def build_vgg(self):
        img = Input(shape=self.shape_hr)

        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[20].output]

        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        return model

    def preprocess_vgg(self, x):
        plot_test_images("preprocess_vgg")
        if isinstance(x, np.ndarray):
            return preprocess_input((x + 1) * 127.5)
        else:
            return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)

    def build_generator(self, ):
        def dense_block(input):
            x1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])

            x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = dense_block(input)
            x = dense_block(x)
            x = dense_block(x)
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out

        def upsample(x, number):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_' + str(number))(x)
            x = self.SubpixelConv2D('upSampleSubPixel_' + str(number), 2)(x)
            x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
            return x

        lr_input = Input(shape=(None, None, 3))

        x_start = Conv2D(64, kernel_size=3, strides=1, padding='same')(lr_input)
        x_start = LeakyReLU(0.2)(x_start)

        x = RRDB(x_start)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])

        x = upsample(x, 1)
        if self.upscaling_factor > 2:
            x = upsample(x, 2)
        if self.upscaling_factor > 4:
            x = upsample(x, 3)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        model = Model(inputs=lr_input, outputs=hr_output)
        return model

    def build_discriminator(self, filters=64):
        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img = Input(shape=self.shape_hr)
        x = conv2d_block(img, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters * 2)
        x = conv2d_block(x, filters * 2, strides=2)
        x = conv2d_block(x, filters * 4)
        x = conv2d_block(x, filters * 4, strides=2)
        x = conv2d_block(x, filters * 8)
        x = conv2d_block(x, filters * 8, strides=2)
        x = Dense(filters * 16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(1)(x)

        model = Model(inputs=img, outputs=x)
        return model

    def build_srgan(self):
        def comput_loss(x):
            img_hr, generated_hr = x
            gen_feature = self.vgg(self.preprocess_vgg(generated_hr))
            ori_feature = self.vgg(self.preprocess_vgg(img_hr))
            percept_loss = tf.losses.mean_squared_error(gen_feature, ori_feature)
            fake_logit, real_logit = self.RaGAN([img_hr, generated_hr])
            gen_loss = K.mean(
                K.binary_crossentropy(K.zeros_like(real_logit), real_logit) +
                K.binary_crossentropy(K.ones_like(fake_logit), fake_logit))
            return [percept_loss, gen_loss]

        img_lr = Input(self.shape_lr)
        img_hr = Input(self.shape_hr)
        generated_hr = self.generator(img_lr)

        self.discriminator.trainable = False
        self.RaGAN.trainable = False

        total_loss = Lambda(comput_loss, name='comput_loss')([img_hr, generated_hr])
        percept_loss = Lambda(lambda x: self.loss_weights['percept'] * x, name='percept_loss')(total_loss[0])
        gen_loss = Lambda(lambda x: self.loss_weights['gen'] * x, name='gen_loss')(total_loss[1])

        model = Model(inputs=[img_lr, img_hr], outputs=[percept_loss, gen_loss])

        model.add_loss(percept_loss)
        model.add_loss(gen_loss)
        model.compile(optimizer=Adam(self.gen_lr))

        model.metrics_names.append('percept_loss')
        model.metrics_tensors.append(percept_loss)
        model.metrics_names.append('gen_loss')
        model.metrics_tensors.append(gen_loss)
        return model

    def build_RaGAN(self):
        def interpolating(x):
            u = K.random_uniform((K.shape(x[0])[0],) + (1,) * (K.ndim(x[0]) - 1))
            return x[0] * u + x[1] * (1 - u)

        def comput_loss(x):
            real, fake = x
            fake_logit = K.sigmoid(fake - K.mean(real))
            real_logit = K.sigmoid(real - K.mean(fake))
            return [fake_logit, real_logit]

        imgs_hr = Input(self.shape_hr)
        generated_hr = Input(self.shape_hr)
        real_discriminator_logits = self.discriminator(imgs_hr)
        fake_discriminator_logits = self.discriminator(generated_hr)

        total_loss = Lambda(comput_loss, name='comput_loss')([real_discriminator_logits, fake_discriminator_logits])
        fake_logit = Lambda(lambda x: x, name='fake_logit')(total_loss[0])
        real_logit = Lambda(lambda x: x, name='real_logit')(total_loss[1])

        dis_loss = K.mean(K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit) +
                          K.binary_crossentropy(K.ones_like(real_logit), real_logit))

        model = Model(inputs=[imgs_hr, generated_hr], outputs=[fake_logit, real_logit])

        model.add_loss(dis_loss)
        model.compile(optimizer=Adam(self.dis_lr))

        model.metrics_names.append('dis_loss')
        model.metrics_tensors.append(dis_loss)
        return model

    def compile_vgg(self, model):
        model.compile(loss='mse', optimizer=Adam(self.gen_lr, 0.9), metrics=['accuracy'])

    def compile_generator(self, model):
        model.compile(loss=self.gan_loss, optimizer=Adam(self.gen_lr, 0.9), metrics=['mae', self.PSNR])

    def compile_discriminator(self, model):
        model.compile(loss=None, optimizer=Adam(self.dis_lr, 0.9, 0.999))

    def compile_srgan(self, model):
        model.compile(loss=None, optimizer=Adam(self.gen_lr, 0.9, 0.999))

    def PSNR(self, y_true, y_pred):
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

    def train_generator(self, epochs, batch_size, workers=1, dataname='doctor',
                        datapath_train='./images/train_dir', datapath_validation='./images/val_dir',
                        datapath_test='./images/val_dir', steps_per_epoch=1000, steps_per_validation=1000,
                        crops_per_image=2, log_weight_path='../models/', log_tensorboard_path='./data/logs/',
                        log_tensorboard_name='SR-RRDB-D', log_tensorboard_update_freq=1,
                        log_test_path="./images/samples-d/"):

        train_loader = DataLoader(datapath_train, batch_size, self.height_hr,
                                  self.width_hr, self.upscaling_factor, crops_per_image)
        test_loader = None
        if datapath_validation is not None:
            test_loader = DataLoader(datapath_validation, batch_size, self.height_hr, self.width_hr,
                                     self.upscaling_factor, crops_per_image)

        self.gen_lr = 3.2e-5
        for step in range(epochs // 10):
            self.compile_generator(self.generator)
            callbacks = []
            if log_tensorboard_path:
                tensorboard = TensorBoard(log_dir=os.path.join(log_tensorboard_path, log_tensorboard_name),
                                          histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False,
                                          update_freq=log_tensorboard_update_freq)
                callbacks.append(tensorboard)
            else:
                print(">> Not logging to tensorboard since no log_tensorboard_path is set")

            modelcheckpoint = ModelCheckpoint(
                os.path.join(log_weight_path, dataname + '_{}X.h5'.format(self.upscaling_factor)),
                monitor='PSNR', save_best_only=True, save_weights_only=True)
            callbacks.append(modelcheckpoint)

            if datapath_test is not None:
                testplotting = LambdaCallback(
                    on_epoch_end=lambda epoch, logs: plot_test_images(
                        self, test_loader, datapath_test, log_test_path,
                        epoch + step * 10, name='RRDB-D'))
                callbacks.append(testplotting)

            self.generator.fit_generator(
                train_loader, steps_per_epoch=steps_per_epoch,
                epochs=10, validation_data=test_loader, validation_steps=steps_per_validation,
                callbacks=callbacks, use_multiprocessing=workers > 1, workers=workers)
            self.generator.save('./models/Doctor_gan(Step %dK).h5' % (step * 10 + 10))
            self.gen_lr /= 1.149
            print(step, self.gen_lr)

    def train_srgan(self, epochs, batch_size, dataname, datapath_train, datapath_validation=None,
                    datapath_test=None, workers=40, max_queue_size=100, first_epoch=0,
                    print_frequency=2, crops_per_image=2, log_weight_frequency=1000,
                    log_weight_path='./models/', log_test_frequency=500, log_test_path="./images/samples/"):

        # :param int epochs: how many epochs to train the network for
        # :param str dataname: name to use for storing model weights etc.
        # :param str datapath_train: path for te image files to use for training
        # :param str datapath_test: path for the image files to use for testing / plotting
        # :param int print_frequency: how often (in epochs) to print progress to
        #        terminal. Warning: will run validation inference!
        # :param int log_weight_frequency: how often (in epochs) should network weights be saved. None for never
        # :param int log_weight_path: where should network weights be saved
        # :param int log_test_frequency: how often (in epochs) should testing & validation be performed
        # :param str log_test_path: where should test results be saved
        # :param str log_tensorboard_path: where should tensorflow logs be sent
        # :param str log_tensorboard_name: what folder should tf logs be saved under

        loader = DataLoader(datapath_train, batch_size, self.height_hr,
                            self.width_hr, self.upscaling_factor, crops_per_image)

        if datapath_validation is not None:
            validation_loader = DataLoader(datapath_validation, batch_size, self.height_hr, self.width_hr,
                self.upscaling_factor, crops_per_image)
        print("Picture Loaders has been ready.")

        enqueuer = OrderedEnqueuer(loader, use_multiprocessing=False, shuffle=True)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
        print("Data Enqueuer has been ready.")

        print_losses = {"G": [], "D": []}
        start_epoch = datetime.datetime.now()

        idxs = np.random.randint(0, len(loader), epochs)

        for epoch in range(first_epoch, epochs + first_epoch):
            if epoch % print_frequency == 1:
                start_epoch = datetime.datetime.now()

            imgs_lr, imgs_hr = next(output_generator)
            generated_hr = self.generator.predict(imgs_lr)

            discriminator_loss = self.RaGAN.train_on_batch([imgs_hr, generated_hr], None)

            generator_loss = self.srgan.train_on_batch([imgs_lr, imgs_hr], None)

            print_losses['G'].append(generator_loss)
            print_losses['D'].append(discriminator_loss)

            if epoch % print_frequency == 0:
                g_avg_loss = np.array(print_losses['G']).mean(axis=0)
                d_avg_loss = np.array(print_losses['D']).mean(axis=0)
                print(self.srgan.metrics_names, g_avg_loss)
                print(self.RaGAN.metrics_names, d_avg_loss)
                print("\nEpoch {}/{} | Time: {}s\n>> Generator/GAN: {}\n>> Discriminator: {}".format(
                    epoch, epochs + first_epoch,
                    (datetime.datetime.now() - start_epoch).seconds,
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.srgan.metrics_names, g_avg_loss)]),
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.RaGAN.metrics_names, d_avg_loss)])
                ))
                print_losses = {"G": [], "D": []}

            if datapath_test and epoch % log_test_frequency == 0:
                print(">> Ploting test images")
                plot_test_images(self, loader, datapath_test, log_test_path, epoch, refer_model=self.refer_model)

            if log_weight_frequency and epoch % log_weight_frequency == 0:
                print(">> Saving the network weights")
                self.save_weights(os.path.join(log_weight_path, dataname), epoch)

    def test(self, refer_model=None, batch_size=1, datapath_test='./images/val_dir', crops_per_image=1,
             log_test_path="./images/test/", model_name=''):

        loader = DataLoader(datapath_test, batch_size, self.height_hr,
                            self.width_hr, self.upscaling_factor, crops_per_image)
        print(">> Ploting test images")
        if self.refer_model is not None:
            refer_model = self.refer_model
        e = -1
        if len(model_name) > 27:
            e = int(model_name[24:-3])
            print(e)
        plot_bigger_images(self, loader, datapath_test, log_test_path)
