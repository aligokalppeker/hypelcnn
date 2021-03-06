import numpy
import tensorflow as tf
from tensorflow import initializers, expand_dims
from tensorflow.contrib import slim

model_forward_generator_name = 'ModelX2Y'
model_backward_generator_name = 'ModelY2X'


def extract_common_normalizer(hsi_grss2013, hsi_grss2018):
    grss2013_band_count = hsi_grss2013.shape[2]
    grss2018_band_count = hsi_grss2018.shape[2]
    scale_down_indices = numpy.rint(numpy.linspace(0, grss2013_band_count - 1, grss2018_band_count)).astype(int)

    grss2013_min = numpy.min(hsi_grss2013, axis=(0, 1))
    grss2018_min = numpy.min(hsi_grss2018, axis=(0, 1))

    hsi2013_global_minimum = numpy.minimum(grss2013_min, numpy.repeat(grss2018_min, 3))
    hsi2018_global_minimum = numpy.take(hsi2013_global_minimum, scale_down_indices)
    hsi_grss2013_zero_centered = hsi_grss2013 - hsi2013_global_minimum
    hsi_grss2018_zero_centered = hsi_grss2018 - hsi2018_global_minimum

    grss2013_max = numpy.max(hsi_grss2013_zero_centered, axis=(0, 1))
    grss2018_max = numpy.max(hsi_grss2018_zero_centered, axis=(0, 1))

    hsi2013_global_maximum = numpy.maximum(grss2013_max, numpy.repeat(grss2018_max, 3))
    hsi2018_global_maximum = numpy.take(hsi2013_global_maximum, scale_down_indices)

    return hsi2013_global_minimum, hsi2018_global_minimum, hsi2013_global_maximum, hsi2018_global_maximum


def _srdata_generator_model(netinput, is_training=True):
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            weights_initializer=initializers.variance_scaling(scale=2.0),
            # weights_regularizer=slim.l2_regularizer(0.00001),
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training, 'decay': 0.95},
            # normalizer_fn=slim.instance_norm,
            # normalizer_params={'center': True, 'scale': True, 'epsilon': 0.001},
            # activation_fn=(lambda inp: slim.nn.leaky_relu(inp)),
            trainable=is_training,
            data_format="NHWC"):
        net = gen_net_method3(netinput)
        return net


def gen_net_method4(netinput):
    return slim.conv2d(netinput, 144, [10, 10],
                       activation_fn=(lambda inp: slim.nn.leaky_relu(inp)))


def gen_net_method3(netinput):
    net_hats = []
    for index in range(0, 144):
        level0 = expand_dims(netinput[:, :, :, index], axis=3)
        level1 = slim.conv2d(level0, 1, [1, 1], activation_fn=(lambda inp: slim.nn.leaky_relu(inp)))
        # level1 = level1 + level0
        level2 = slim.conv2d(level1, 1, [3, 3], activation_fn=(lambda inp: slim.nn.leaky_relu(inp)))
        # level2 = level2 + level1
        level3 = slim.conv2d(level2, 1, [5, 5], activation_fn=(lambda inp: slim.nn.leaky_relu(inp)))
        # level3 = level3 + level2
        level4 = slim.conv2d(level3, 1, [10, 10], activation_fn=(lambda inp: slim.nn.leaky_relu(inp)),
                             normalizer_fn=None,
                             normalizer_params=None,
                             weights_regularizer=None)
        net_hats.append(level4)

    net = tf.concat(net_hats, axis=3)
    return net


def gen_net_method2(netinput):
    net = tf.reshape(netinput, [-1, 100, 144])
    net_hats = []
    for index in range(0, 144):
        net_hats.append(expand_dims(slim.fully_connected(net[:, :, index], 10 * 10), axis=2))
    net = tf.concat(net_hats, axis=2)
    net = tf.reshape(net, [-1, 10, 10, 144])
    return net


def gen_net_method1(netinput):
    net_hats = []
    for first_index in range(0, 10):
        net_internal_hats = []
        for second_index in range(0, 10):
            net_internal_hats.append(
                slim.conv2d(expand_dims(expand_dims(netinput[:, first_index, second_index, :], axis=1), axis=1),
                            144, [1, 1], activation_fn=None))
        net_hats.append(tf.concat(net_internal_hats, axis=2))
    net = tf.concat(net_hats, axis=1)
    return net


def _srdata_discriminator_model(generated_data, generator_input, is_training=True):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=initializers.variance_scaling(scale=2.0),
                        weights_regularizer=slim.l2_regularizer(0.001),
                        # normalizer_fn=slim.batch_norm,
                        # normalizer_params={'is_training': is_training, 'decay': 0.95},
                        normalizer_fn=slim.instance_norm,
                        normalizer_params={'center': True, 'scale': True, 'epsilon': 0.001},
                        activation_fn=(lambda inp: slim.nn.leaky_relu(inp))):
        # net = tf.concat(axis=3, values=[generated_data, generator_input])
        #
        # net = slim.flatten(net)
        # net = slim.fully_connected(net, 768 * 2, scope='fc2')
        # net = slim.fully_connected(net, 384 * 2, scope='fc2')
        # net = slim.fully_connected(net, 192 * 2, scope='fc3')
        # net = slim.fully_connected(net, 128, scope='fc4')
        # net = slim.fully_connected(net, 96, scope='fc5')
        # net = slim.fully_connected(net, 64, scope='fc6')
        net_hats = []
        for index in range(0, 144):
            level0 = expand_dims(generated_data[:, :, :, index], axis=3)
            level1 = slim.conv2d(level0, 1, [5, 5], padding='VALID')
            level2 = slim.conv2d(level1, 1, [3, 3], padding='VALID')
            level3 = slim.conv2d(level2, 1, [1, 1], padding='VALID',
                                 activation_fn=None, normalizer_fn=None, normalizer_params=None)
            net_hats.append(level3)
        net = tf.concat(net_hats, axis=3)
        # net = slim.flatten(net)
        # net = slim.fully_connected(net, 4 * 4 * 144, scope='fc1')
        # net = slim.dropout(net, is_training=is_training)
        # net = slim.fully_connected(net, 4 * 4 * 144, scope='fc2',
        #                            activation_fn=None, normalizer_fn=None, normalizer_params=None)
        # net = tf.reshape(net, [-1, 2, 2, 144])
    return net


def construct_inference_graph(input_tensor, model_name):
    with tf.variable_scope(model_name):
        with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
            generated_tensor = _srdata_generator_model(input_tensor, False)
    return generated_tensor


def create_generator_restorer():
    # Restore all the variables that were saved in the checkpoint.
    cyclegan_restorer = tf.train.Saver(
        slim.get_variables_to_restore(include=[model_forward_generator_name]) +
        slim.get_variables_to_restore(include=[model_backward_generator_name]), name='GeneratorRestoreHandler'
    )
    return cyclegan_restorer
