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
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--config_path', default='../configs/default_mnist.json',
                        help="Path of the .json file that contains the hyperparameters of the network. ")
    parser.add_argument('--debug', default=0, type=int,
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--gpus', default=1, type=int)

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Get hyperparameters from config json and merge them with the Arguments namespace.
    hparams = process_config(args.config_path)
    args = merge_configs(dict(hparams), vars(args))
    print(args)

    # Load data.
    mnist_loader = MnistLoader(args)
    (XTrain, YTrain), (XTest, YTest) = mnist_loader.get_mnist()

    # Define model.
    with tf.device('/cpu:0'):
        model, eval_model, manipulate_model = caps_net(args=args)
    model.summary()
    # plot_model(Model, to_file=Args.save_dir + '/model.png', show_shapes=True)  # TODO: UNCOMMENT THIS.

    # Train or test.
    if args.weights is not None:  # Init the model weights with provided one.
        model.load_weights(args.weights)

    if not args.testing:
        # Callbacks.
        log = callbacks.CSVLogger(args.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                                   batch_size=args.tr_batch_size, histogram_freq=args.debug)
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (args.lr_decay ** epoch))
        training_callbacks = [log, tb, checkpoint, lr_decay]

        if args.gpus < 2:  # If CPU or single GPU training.
            train(model=model, data_generator=mnist_loader, args=args, training_callbacks=training_callbacks)
        else:
            # Define multi-gpu model.
            multi_model = multi_gpu_model(model, gpus=args.gpus)
            train(model=multi_model, data_generator=mnist_loader, args=args, training_callbacks=training_callbacks)

            # Save weights.
            model.save_weights(args.save_dir + '/trained_model.h5')
            print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

            # Test the model.
            test(model=eval_model, data=(XTest, YTest), args=args)

    else:  # As long as weights are given, will run testing.
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (XTest, YTest), args)
        test(model=eval_model, data=(XTest, YTest), args=args)


if __name__ == "__main__":
    main()
