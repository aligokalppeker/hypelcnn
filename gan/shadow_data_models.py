import tensorflow as tf
from tensorflow import transpose
from tensorflow.python.keras.activations import tanh
from tensorflow.python.ops.gen_nn_ops import leaky_relu
from tensorflow.python.ops.initializers_ns import variance_scaling
from tf_slim import conv2d, flatten, fully_connected, arg_scope, l2_regularizer, separable_conv2d, \
    convolution1d, conv2d_transpose, batch_norm


def _shadowdata_generator_model_simple(netinput, is_training=True):
    with arg_scope(
            [conv2d, conv2d_transpose, convolution1d],
            trainable=is_training,
            data_format="NHWC"
    ):
        band_size = netinput.get_shape()[3].value
        net = tf.expand_dims(tf.squeeze(netinput, axis=[1, 2]), axis=2)
        net = convolution1d(net, 1, band_size, padding="SAME",
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_regularizer=None,
                            activation_fn=None)
    return tf.expand_dims(tf.expand_dims(flatten(net), axis=1), axis=1)


def _shadowdata_discriminator_model_simple(generated_data, generator_input, is_training=True):
    with arg_scope([fully_connected, separable_conv2d, convolution1d],
                   weights_initializer=tf.compat.v1.initializers.variance_scaling(scale=2.0),
                   activation_fn=(lambda inp: leaky_relu(inp, alpha=0.01))):
        band_size = generated_data.get_shape()[3].value

        net = tf.concat(axis=3, values=[generated_data, generator_input])
        net = tf.squeeze(net, axis=[1, 2])
        net = tf.expand_dims(net, axis=2)
        size = band_size * 2
        net = convolution1d(net, size, size, padding="VALID",
                            normalizer_fn=None,
                            normalizer_params=None,
                            activation_fn=None)
        net = tf.expand_dims(tf.expand_dims(flatten(net), axis=1), axis=1)
    return net


def _shadowdata_generator_model(netinput, create_only_encoder, is_training):
    with arg_scope(
            [conv2d, conv2d_transpose, convolution1d],
            # weights_initializer=initializers.variance_scaling(scale=2.0),
            weights_initializer=tf.compat.v1.initializers.zeros(),
            # weights_regularizer=l1_l2_regularizer(),
            # normalizer_fn=batch_norm,
            # normalizer_params={"is_training": is_training, "decay": 0.95},
            # normalizer_fn=instance_norm,
            # normalizer_params={"center": True, "scale": True, "epsilon": 0.001},
            activation_fn=(lambda inp: leaky_relu(inp, alpha=0.1)),
            trainable=is_training,
            data_format="NHWC"
    ):
        num_filters = 1
        band_size = netinput.get_shape()[3].value
        kernel_size = band_size

        net0 = tf.expand_dims(tf.squeeze(netinput, axis=[1, 2]), axis=2)
        net1 = convolution1d(net0, num_filters, kernel_size, scope="net1", padding="SAME")
        net1 = net1 + net0

        net2 = convolution1d(net1, num_filters, kernel_size // 2, scope="net2", padding="SAME")
        net2 = net2 + net1 + net0

        net3 = convolution1d(net2, num_filters, kernel_size // 4, scope="net3", padding="SAME")
        net3 = net3 + net2 + net1

        net4 = convolution1d(net3, num_filters, kernel_size // 8, scope="net4", padding="SAME")
        net4 = net4 + net3 + net2
        result = net4

        if not create_only_encoder:
            net5 = convolution1d(net4, num_filters, kernel_size // 4, scope="net5", padding="SAME")
            net5 = net5 + net4 + net3

            net6 = convolution1d(net5, num_filters, kernel_size // 2, scope="net6", padding="SAME")
            net6 = net6 + net5 + net4

            net7 = convolution1d(net6, num_filters, kernel_size, scope="net7", padding="SAME",
                                 normalizer_fn=None,
                                 normalizer_params=None,
                                 weights_regularizer=None,
                                 activation_fn=tanh)
            result = net7

        flattened = flatten(result)
    return tf.expand_dims(tf.expand_dims(flattened, axis=1), axis=1)


def _shadowdata_discriminator_model(generated_data, generator_input, is_training):
    with arg_scope([fully_connected, separable_conv2d, convolution1d],
                   weights_initializer=tf.compat.v1.initializers.variance_scaling(scale=2.0),
                   weights_regularizer=l2_regularizer(0.0001),
                   normalizer_fn=batch_norm,
                   normalizer_params={"is_training": is_training, "decay": 0.999},
                   # normalizer_fn=instance_norm,
                   # normalizer_params={"center": True, "scale": True, "epsilon": 0.001},
                   activation_fn=(lambda inp: leaky_relu(inp, alpha=0.1))):
        band_size = generated_data.get_shape()[3].value

        net = generated_data
        # net = tf.squeeze(net, axis=[1, 2])
        # net = tf.expand_dims(net, axis=2)
        # net1 = convolution1d(net, band_size, band_size, padding="VALID")
        # net2 = convolution1d(transpose(net1, perm=[0, 2, 1]), band_size, band_size, padding="VALID",
        #                      normalizer_fn=None,
        #                      normalizer_params=None,
        #                      weights_regularizer=None,
        #                      activation_fn=None)

        net = flatten(net)
        net1 = fully_connected(net, band_size)
        net2 = fully_connected(net1, band_size,
                               normalizer_fn=None,
                               normalizer_params=None,
                               weights_regularizer=None,
                               activation_fn=None)

    return tf.expand_dims(tf.expand_dims(flatten(net2), axis=1), axis=1)


def _shadowdata_feature_discriminator_model(generated_data, patch_count, embedded_feature_size, is_training):
    with arg_scope([fully_connected, separable_conv2d, convolution1d],
                   weights_initializer=tf.compat.v1.initializers.variance_scaling(scale=2.0),
                   weights_regularizer=l2_regularizer(0.001),
                   # normalizer_fn=batch_norm,
                   # normalizer_params={"is_training": is_training, "decay": 0.999},
                   # normalizer_fn=instance_norm,
                   # normalizer_params={"center": True, "scale": True, "epsilon": 0.001},
                   activation_fn=(lambda inp: leaky_relu(inp, alpha=0.1))):
        band_size = generated_data.get_shape()[3].value
        patch_size = generated_data.get_shape()[3].value // patch_count

        net = flatten(generated_data)

        output_tensors = []
        for patch_loc_start in range(0, band_size, patch_size):
            current_net = net[:, patch_loc_start:patch_loc_start + patch_size]
            current_net = fully_connected(current_net, patch_size)
            current_net = fully_connected(current_net, patch_size // 4)
            current_net = fully_connected(current_net, patch_size // 2)
            current_net = fully_connected(current_net, embedded_feature_size)
            output_tensors.append(tf.expand_dims(tf.math.l2_normalize(current_net), axis=1))

    return tf.concat(output_tensors, axis=1)
