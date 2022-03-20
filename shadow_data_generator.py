import tensorflow as tf
from tensorflow import initializers
from tensorflow_core import transpose
from tensorflow_core.contrib import slim


def _shadowdata_generator_model_simple(netinput, is_training=True):
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose, slim.convolution1d],
            trainable=is_training,
            data_format="NHWC"
    ):
        band_size = netinput.get_shape()[3].value
        net = tf.expand_dims(tf.squeeze(netinput, axis=[1, 2]), axis=2)
        net = slim.convolution1d(net, 1, band_size, padding='SAME',
                                 normalizer_fn=None,
                                 normalizer_params=None,
                                 weights_regularizer=None,
                                 activation_fn=None)
    return tf.expand_dims(tf.expand_dims(slim.flatten(net), axis=1), axis=1)


def _shadowdata_discriminator_model_simple(generated_data, generator_input, is_training=True):
    with slim.arg_scope([slim.fully_connected, slim.separable_conv2d, slim.convolution1d],
                        weights_initializer=initializers.variance_scaling(scale=2.0),
                        activation_fn=(lambda inp: slim.nn.leaky_relu(inp, alpha=0.01))):
        band_size = generated_data.get_shape()[3].value

        net = tf.concat(axis=3, values=[generated_data, generator_input])
        net = tf.squeeze(net, axis=[1, 2])
        net = tf.expand_dims(net, axis=2)
        size = band_size * 2
        net = slim.convolution1d(net, size, size, padding='VALID',
                                 normalizer_fn=None,
                                 normalizer_params=None,
                                 activation_fn=None)
        net = tf.expand_dims(tf.expand_dims(slim.flatten(net), axis=1), axis=1)
    return net


def _shadowdata_generator_model(netinput, is_training=True):
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose, slim.convolution1d],
            # weights_initializer=initializers.variance_scaling(scale=2.0),
            weights_initializer=initializers.zeros(),
            # weights_regularizer=slim.l1_l2_regularizer(),
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training, 'decay': 0.95},
            # normalizer_fn=slim.instance_norm,
            # normalizer_params={'center': True, 'scale': True, 'epsilon': 0.001},
            activation_fn=(lambda inp: slim.nn.leaky_relu(inp, alpha=0.1)),
            trainable=is_training,
            data_format="NHWC"
    ):
        num_filters = 1
        band_size = netinput.get_shape()[3].value
        kernel_size = band_size

        net0 = tf.expand_dims(tf.squeeze(netinput, axis=[1, 2]), axis=2)
        net1 = slim.convolution1d(net0, num_filters, kernel_size, padding='SAME')
        net1 = net1 + net0

        net2 = slim.convolution1d(net1, num_filters, kernel_size // 2, padding='SAME')
        net2 = net2 + net1 + net0

        net3 = slim.convolution1d(net2, num_filters, kernel_size // 4, padding='SAME')
        net3 = net3 + net2 + net1

        net4 = slim.convolution1d(net3, num_filters, kernel_size // 8, padding='SAME')
        net4 = net4 + net3 + net2

        net5 = slim.convolution1d(net4, num_filters, kernel_size // 4, padding='SAME')
        net5 = net5 + net4 + net3

        net6 = slim.convolution1d(net5, num_filters, kernel_size // 2, padding='SAME')
        net6 = net6 + net5 + net4

        net7 = slim.convolution1d(net6, num_filters, kernel_size, padding='SAME',
                                  normalizer_fn=None,
                                  normalizer_params=None,
                                  weights_regularizer=None,
                                  activation_fn=None)
        flatten = slim.flatten(net7)
        # net9 = slim.fully_connected(flatten, band_size, activation_fn=None)
    return tf.expand_dims(tf.expand_dims(flatten, axis=1), axis=1)


def _shadowdata_discriminator_model(generated_data, generator_input, is_training=True):
    with slim.arg_scope([slim.fully_connected, slim.separable_conv2d, slim.convolution1d],
                        weights_initializer=initializers.variance_scaling(scale=2.0),
                        weights_regularizer=slim.l2_regularizer(0.001),
                        # normalizer_fn=slim.batch_norm,
                        # normalizer_params={'is_training': is_training, 'decay': 0.999},
                        # normalizer_fn=slim.instance_norm,
                        # normalizer_params={'center': True, 'scale': True, 'epsilon': 0.001},
                        activation_fn=(lambda inp: slim.nn.leaky_relu(inp, alpha=0.1))):
        band_size = generated_data.get_shape()[3].value

        net = generated_data
        net = tf.squeeze(net, axis=[1, 2])
        net = tf.expand_dims(net, axis=2)

        net1 = slim.convolution1d(net, band_size, band_size, padding='VALID')

        net2 = slim.convolution1d(transpose(net1, perm=[0, 2, 1]), band_size, band_size, padding='VALID',
                                  normalizer_fn=None,
                                  normalizer_params=None,
                                  activation_fn=None)

    return tf.expand_dims(tf.expand_dims(slim.flatten(net2), axis=1), axis=1)


def _shadowdata_feature_discriminator_model(generated_data, is_training=True):
    with slim.arg_scope([slim.fully_connected, slim.separable_conv2d, slim.convolution1d],
                        weights_initializer=initializers.variance_scaling(scale=2.0),
                        weights_regularizer=slim.l2_regularizer(0.001),
                        # normalizer_fn=slim.batch_norm,
                        # normalizer_params={'is_training': is_training, 'decay': 0.999},
                        # normalizer_fn=slim.instance_norm,
                        # normalizer_params={'center': True, 'scale': True, 'epsilon': 0.001},
                        activation_fn=(lambda inp: slim.nn.leaky_relu(inp, alpha=0.1))):
        band_size = generated_data.get_shape()[3].value

        net = generated_data
        net = slim.flatten(net)

        net1 = slim.fully_connected(net, band_size // 2)
        net2 = slim.fully_connected(net1, band_size // 4)
        net3 = slim.fully_connected(net2, band_size // 8)
    return net3
