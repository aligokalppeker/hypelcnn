import tensorflow as tf
from tensorflow import initializers, reduce_mean
from tensorflow.contrib import slim

model_forward_generator_name = 'ModelX2Y'
model_backward_generator_name = 'ModelY2X'


def _shadowdata_generator_model(netinput, is_training=True):
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose, slim.convolution1d],
            # weights_initializer=initializers.variance_scaling(scale=2.0),
            # weights_regularizer=slim.l1_l2_regularizer(),
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training, 'decay': 0.95},
            # normalizer_fn=slim.instance_norm,
            # normalizer_params={'center': True, 'scale': True, 'epsilon': 0.001},
            activation_fn=(lambda inp: slim.nn.leaky_relu(inp, alpha=0.2)),
            trainable=is_training,
            data_format="NHWC"
    ):
        band_size = netinput.get_shape()[3].value

        net0 = tf.expand_dims(tf.squeeze(netinput, axis=[1, 2]), axis=2)
        net1 = slim.convolution1d(net0, 1, band_size, padding='SAME',
                                  normalizer_fn=None,
                                  normalizer_params=None,
                                  weights_regularizer=None)
        net2 = slim.convolution1d(net1, 1, band_size * 2, padding='SAME',
                                  normalizer_fn=None,
                                  normalizer_params=None,
                                  weights_regularizer=None)

        # net3 = slim.convolution1d(net2, 1, band_size * 4, padding='SAME',
        #                           normalizer_fn=None,
        #                           normalizer_params=None,
        #                           weights_regularizer=None)
        # net4 = slim.convolution1d(net3, 1, band_size * 2, padding='SAME',
        #                           normalizer_fn=None,
        #                           normalizer_params=None,
        #                           weights_regularizer=None)
        net5 = slim.convolution1d(net2, 1, band_size, padding='SAME',
                                  normalizer_fn=None,
                                  normalizer_params=None,
                                  weights_regularizer=None,
                                  activation_fn=None)
    return tf.expand_dims(tf.expand_dims(slim.flatten(net5), axis=1), axis=1)


def _shadowdata_discriminator_model(generated_data, generator_input, is_training=True):
    with slim.arg_scope([slim.fully_connected, slim.separable_conv2d, slim.convolution1d],
                        weights_initializer=initializers.variance_scaling(scale=2.0),
                        # weights_regularizer=slim.l2_regularizer(0.001),
                        # normalizer_fn=slim.batch_norm,
                        # normalizer_params={'is_training': is_training, 'decay': 0.999},
                        # normalizer_fn=slim.instance_norm,
                        # normalizer_params={'center': True, 'scale': True, 'epsilon': 0.001},
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


def construct_inference_graph(input_tensor, model_name, clip_invalid_values=True):
    shp = input_tensor.get_shape()

    output_tensor_in_col = []
    for first_dim in range(shp[0]):
        output_tensor_in_row = []
        for second_dim in range(shp[1]):
            input_cell = tf.expand_dims(tf.expand_dims(tf.expand_dims(input_tensor[first_dim][second_dim], 0), 0), 0)
            with tf.variable_scope(model_name):
                with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
                    generated_tensor = _shadowdata_generator_model(input_cell, False)
                    if clip_invalid_values:
                        input_mean = reduce_mean(input_cell)
                        generated_mean = reduce_mean(generated_tensor)

            if clip_invalid_values:
                result_tensor = tf.cond(tf.less(generated_mean, input_mean),
                                        lambda: generated_tensor,
                                        lambda: input_cell)
            else:
                result_tensor = generated_tensor

            output_tensor_in_row.append(tf.squeeze(result_tensor, [0, 1]))
        image_output_row = tf.concat(output_tensor_in_row, axis=0)
        output_tensor_in_col.append(image_output_row)

    image_output_row = tf.stack(output_tensor_in_col)

    return image_output_row


def create_generator_restorer():
    # Restore all the variables that were saved in the checkpoint.
    cyclegan_restorer = tf.train.Saver(
        slim.get_variables_to_restore(include=[model_forward_generator_name]) +
        slim.get_variables_to_restore(include=[model_backward_generator_name]), name='GeneratorRestoreHandler'
    )
    return cyclegan_restorer


def construct_cyclegan_inference_graph(input_data, model_name):
    with tf.device('/cpu:0'):
        axis_id = 2
        band_size = input_data.get_shape()[axis_id].value
        hs_lidar_groups = tf.split(axis=axis_id, num_or_size_splits=[band_size - 1, 1],
                                   value=input_data)
        hs_converted = construct_inference_graph(hs_lidar_groups[0], model_name, clip_invalid_values=False)
    return tf.concat(axis=axis_id, values=[hs_converted, hs_lidar_groups[1]])


def construct_cyclegan_inference_graph_randomized(input_data):
    # coin = tf.less(tf.random_uniform([1], 0, 1.0)[0], 0.5)
    # images = tf.cond(coin,
    #                  lambda: GRSS2013DataLoader.construct_cyclegan_inference_graph(input_data,
    #                                                                                model_forward_generator_name),
    #                  lambda: GRSS2013DataLoader.construct_cyclegan_inference_graph(input_data,
    #                                                                                model_backward_generator_name))
    images = construct_cyclegan_inference_graph(input_data, model_forward_generator_name)
    return images


def construct_simple_shadow_inference_graph(input_data, shadow_ratio):
    # coin = tf.less(tf.random_uniform([1], 0, 1.0)[0], 0.5)
    # images = tf.cond(coin, lambda: input_data / shadow_ratio, lambda: input_data * shadow_ratio)
    images = input_data / shadow_ratio
    return images
