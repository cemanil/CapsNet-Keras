import numpy as np
from keras import layers, models
from capsnet.models.layers.capsnet_layers import CapsuleLayer, primary_cap, Length, Mask


def caps_net(args):
    """
    A Capsule Network on MNIST.

    :return: Three Keras Models, the first one used for training, the second one for evaluation, third one for
        inspecting when we manipulate calsule activations.
            `eval_model` can also be used for training.
    """
    input_shape = (args.im_width, args.im_height, args.im_chn)
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer.
    conv1 = layers.Conv2D(filters=args.lyr_1_filters, kernel_size=args.lyr_1_kernel_size, strides=args.lyr_1_strides,
                          padding=args.lyr_1_padding, activation=args.lyr_1_activation, name=args.lyr_1_name)(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule].
    primarycaps = primary_cap(conv1, dim_capsule=args.pcaps_dim_capsule, n_channels=args.pcaps_n_channels,
                              kernel_size=args.pcaps_kernel_size, strides=args.pcaps_strides, padding=args.pcaps_padding)

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=args.output_cls, dim_capsule=args.dcaps_dim_capsule,
                             routings=args.dcaps_routings, name=args.dcaps_name)(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name=args.out_caps_name)(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(args.output_cls,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training.
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction.

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name=args.dcdr_name)
    decoder.add(layers.Dense(args.dcdr_lyr_1_hdim, activation=args.dcdr_lyr_1_activation,
                             input_dim=args.dcaps_dim_capsule * args.output_cls))
    decoder.add(layers.Dense(args.dcdr_lyr_2_hdim, activation=args.dcdr_lyr_2_activation))
    decoder.add(layers.Dense(np.prod(input_shape), activation=args.dcdr_lyr_3_activation))
    decoder.add(layers.Reshape(target_shape=input_shape, name=args.dcdr_output_name))

    # Models for training and evaluation (prediction).
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # Model for manipulating units in digitcaps.
    noise = layers.Input(shape=(args.output_cls, args.dcaps_dim_capsule))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model
