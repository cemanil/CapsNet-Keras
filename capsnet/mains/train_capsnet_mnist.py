import os
import argparse
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model

from capsnet.data_loaders.mnist import MnistLoader
from capsnet.models.capsnet import caps_net
from capsnet.trainers.capsnet_trainer import train, test, manipulate_latent
from capsnet.utils.utils import plot_log  # TODO: Understand why this is unused.


def main():
    # Setting the hyper parameters.
    Parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    Parser.add_argument('--epochs', default=50, type=int)
    Parser.add_argument('--batch_size', default=300, type=int)
    Parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    Parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
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
    Parser.add_argument('--gpus', default=1, type=int)
    Args = Parser.parse_args()
    print(Args)
    if not os.path.exists(Args.save_dir):
        os.makedirs(Args.save_dir)

    # Load data.
    mnist_loader = MnistLoader()
    (XTrain, YTrain), (XTest, YTest) = mnist_loader.get_mnist()

    # Define model.
    with tf.device('/cpu:0'):
        Model, EvalModel, ManipulateModel = caps_net(input_shape=XTrain.shape[1:],
                                                     n_class=len(np.unique(np.argmax(YTrain, 1))),
                                                     routings=Args.routings)
    Model.summary()
    # plot_model(Model, to_file=Args.save_dir + '/model.png', show_shapes=True)  # TODO: UNCOMMENT THIS.

    # Train or test.
    if Args.weights is not None:  # Init the model weights with provided one.
        Model.load_weights(Args.weights)

    if not Args.testing:
        if Args.gpus < 2:  # If cpu or single GPU training.
            train(model=Model, data_generator=mnist_loader, args=Args)
        else:
            # Define multi-gpu model.
            MultiModel = multi_gpu_model(Model, gpus=Args.gpus)
            train(model=MultiModel, data_generator=mnist_loader, args=Args)

            # Save weights.
            Model.save_weights(Args.save_dir + '/trained_model.h5')
            print('Trained model saved to \'%s/trained_model.h5\'' % Args.save_dir)

            # Test the model.
            test(model=EvalModel, data=(XTest, YTest), args=Args)

    else:  # As long as weights are given, will run testing.
        if Args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(ManipulateModel, (XTest, YTest), Args)
        test(model=EvalModel, data=(XTest, YTest), args=Args)


if __name__ == "__main__":
    main()