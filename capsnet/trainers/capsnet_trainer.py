import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras import optimizers
from capsnet.utils.utils import margin_loss, combine_images, plot_log


def train(model, data_generator, args, training_callbacks):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data_generator: a class whose static methods are generators for training and validation data.
    :param args: arguments
    :param training_callbacks: a list of keras callbacks.
    :return: The trained model
    """
    # Compile the model.
    model.compile(optimizer=optimizers.Adam(lr=args.learning_rate),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=data_generator.train_generator(args.tr_batch_size,
                                                                 shift_fraction=args.shift_fraction),
                        steps_per_epoch=args.no_tr_ex / args.tr_batch_size,
                        epochs=args.epochs,
                        validation_data=data_generator.valid_generator(args.val_batch_size),
                        validation_steps=args.no_test_ex / args.val_batch_size,
                        callbacks=training_callbacks)

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/log.csv', show=False)

    return model


def test(model, test_generator, args):
    x_test, y_test = next(test_generator)
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    assert isinstance(img, np.ndarray)

    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, test_generator, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = test_generator.next()
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    assert isinstance(img, np.ndarray)

    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)
