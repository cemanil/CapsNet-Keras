import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


class MnistLoader(object):

    def __init__(self, args):
        # Obtain args.
        self.args = args

        # Load MNIST data.
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_mnist()

        # Preprocess MNIST - reshaping and casting.
        self._reshape_mnist()

    @staticmethod
    def load_mnist():
        # The data, shuffled and split between train and test sets.
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        return (x_train, y_train), (x_test, y_test)

    def train_generator(self, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # Shift pixels of the image.
        generator = train_datagen.flow(self.x_train, self.y_train, batch_size=batch_size)
        while True:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    def valid_generator(self, batch_size):
        valid_datagen = ImageDataGenerator()
        generator = valid_datagen.flow(self.x_test, self.y_test, batch_size=batch_size)
        while True:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    def test_generator(self, batch_size):
        valid_datagen = ImageDataGenerator()
        generator = valid_datagen.flow(self.x_test, self.y_test, batch_size=batch_size)
        while True:
            x_batch, y_batch = generator.next()
            yield [x_batch, y_batch]

    def get_mnist(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def _reshape_mnist(self):
        assert isinstance(self.x_train, np.ndarray)
        assert isinstance(self.x_test, np.ndarray)

        # Scale to [0,1].
        self.x_train = self.x_train.reshape(-1, self.args.im_width, self.args.im_height, self.args.im_chn)
        self.x_train = self.x_train.astype('float32')
        self.x_train = (self.x_train - self.x_train.min()) / (self.x_train.max() - self.x_train.min())

        self.x_test = self.x_test.reshape(-1, self.args.im_width, self.args.im_height, self.args.im_chn)
        self.x_test = self.x_test.astype('float32')
        self.x_test = (self.x_test - self.x_test.min()) / (self.x_test.max() - self.x_test.min())

        # Cast to float32.
        self.y_train = to_categorical(self.y_train.astype('float32'))
        self.y_test = to_categorical(self.y_test.astype('float32'))
