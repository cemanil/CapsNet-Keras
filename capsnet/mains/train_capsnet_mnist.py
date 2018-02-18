import os
import argparse
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model
from keras import callbacks

from capsnet.data_loaders.mnist import MnistLoader
from capsnet.models.capsnet import caps_net
from capsnet.trainers.capsnet_trainer import train, test, manipulate_latent
from capsnet.utils.config import process_config, merge_configs


def main():
    # Setting the hyper parameters.
    Parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    Parser.add_argument('--config_path', default='../configs/default_mnist.json',
                        help="Path of the .json file that contains the hyperparameters of the network. ")
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
    if not os.path.exists(Args.save_dir):
        os.makedirs(Args.save_dir)

    # Get hyperparameters and put them in the arguments namespace.
    hparams = process_config(Args.config_path)
    Args = merge_configs(dict(hparams), vars(Args))
    print(Args)

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
        # Callbacks.
        log = callbacks.CSVLogger(Args.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=Args.save_dir + '/tensorboard-logs',
                                   batch_size=Args.batch_size, histogram_freq=Args.debug)
        checkpoint = callbacks.ModelCheckpoint(Args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: Args.learning_rate * (Args.lr_decay ** epoch))
        training_callbacks = [log, tb, checkpoint, lr_decay]

        if Args.gpus < 2:  # If cpu or single GPU training.
            train(model=Model, data_generator=mnist_loader, args=Args, training_callbacks=training_callbacks)
        else:
            # Define multi-gpu model.
            MultiModel = multi_gpu_model(Model, gpus=Args.gpus)
            train(model=MultiModel, data_generator=mnist_loader, args=Args, training_callbacks=training_callbacks)

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
