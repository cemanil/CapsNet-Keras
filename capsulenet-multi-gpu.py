"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       python capsulenet-multi-gpu.py
       python capsulenet-multi-gpu.py --gpus 2
       ... ...

Result:
    About 55 seconds per epoch on two GTX1080Ti GPU cards

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

from keras import optimizers
from keras import backend as kb
from capsulenet import caps_net, margin_loss, load_mnist, manipulate_latent, test

kb.set_image_data_format('channels_last')


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # Unpacking the data.
    (x_train, y_train), (x_test, y_test) = data

    # Callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # Compile the model.
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon])

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.utils.vis_utils import plot_model
    from keras.utils import multi_gpu_model

    # Setting the hyper parameters.
    import argparse

    Parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    Parser.add_argument('--epochs', default=50, type=int)
    Parser.add_argument('--batch_size', default=300, type=int)
    Parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    Parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    Parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    Parser.add_argument('--debug', default=0, type=int,
                        help="Save weights by TensorBoard")
    Parser.add_argument('--save_dir', default='./result')
    Parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    Parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    Parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    Parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    Parser.add_argument('--gpus', default=2, type=int)
    Args = Parser.parse_args()
    print(Args)
    if not os.path.exists(Args.save_dir):
        os.makedirs(Args.save_dir)

    # Load data.
    (XTrain, YTrain), (XTest, YTest) = load_mnist()

    # Define model.
    with tf.device('/cpu:0'):
        Model, EvalModel, ManipulateModel = caps_net(input_shape=XTrain.shape[1:],
                                                     n_class=len(np.unique(np.argmax(YTrain, 1))),
                                                     routings=Args.routings)
    Model.summary()
    plot_model(Model, to_file=Args.save_dir + '/model.png', show_shapes=True)

    # Train or test.
    if Args.weights is not None:  # Init the model weights with provided one.
        Model.load_weights(Args.weights)
    if not Args.testing:
        # Define muti-gpu model.
        MultiModel = multi_gpu_model(Model, gpus=Args.gpus)
        train(model=MultiModel, data=((XTrain, YTrain), (XTest, YTest)), args=Args)
        Model.save_weights(Args.save_dir + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % Args.save_dir)
        test(model=EvalModel, data=(XTest, YTest), args=Args)
    else:  # As long as weights are given, will run testing.
        if Args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(ManipulateModel, (XTest, YTest), Args)
        test(model=EvalModel, data=(XTest, YTest), args=Args)
