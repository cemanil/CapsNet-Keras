from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from itertools import cycle


class MnistLoader(object):

    def __init__(self):
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
        """
        # Training without data augmentation:
        model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
                  validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
        """

        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST.
        generator = train_datagen.flow(self.x_train, self.y_train, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    def valid_generator(self):
        valid_set = [[self.x_test, self.y_test], [self.y_test, self.x_test]]
        cycle_valid = cycle([valid_set])

        return cycle_valid

    def get_mnist(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def _reshape_mnist(self):
        # Scale to [0,1].
        self.x_train = self.x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
        self.x_test = self.x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

        # Cast to float32.
        self.y_train = to_categorical(self.y_train.astype('float32'))
        self.y_test = to_categorical(self.y_test.astype('float32'))
