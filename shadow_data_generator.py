import os
import random

import numpy
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy import linspace
from tensorflow import initializers
from tensorflow_core import transpose
from tensorflow_core.contrib import slim

from tqdm import tqdm


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


def construct_gan_inference_graph(input_data, gan_inference_wrapper):
    with tf.device('/cpu:0'):
        axis_id = 2
        band_size = input_data.get_shape()[axis_id].value
        hs_lidar_groups = tf.split(axis=axis_id, num_or_size_splits=[band_size - 1, 1],
                                   value=input_data)
        hs_converted = gan_inference_wrapper.construct_inference_graph(hs_lidar_groups[0],
                                                                       is_shadow_graph=True,
                                                                       clip_invalid_values=False)
    return tf.concat(axis=axis_id, values=[hs_converted, hs_lidar_groups[1]])


def construct_gan_inference_graph_randomized(input_data, wrapper):
    # coin = tf.less(tf.random_uniform([1], 0, 1.0)[0], 0.5)
    # images = tf.cond(coin,
    #                  lambda: GRSS2013DataLoader.construct_cyclegan_inference_graph(input_data,
    #                                                                                model_forward_generator_name),
    #                  lambda: GRSS2013DataLoader.construct_cyclegan_inference_graph(input_data,
    #                                                                                model_backward_generator_name))
    images = construct_gan_inference_graph(input_data, wrapper)
    return images


def construct_simple_shadow_inference_graph(input_data, shadow_ratio):
    # coin = tf.less(tf.random_uniform([1], 0, 1.0)[0], 0.5)
    # images = tf.cond(coin, lambda: input_data / shadow_ratio, lambda: input_data * shadow_ratio)
    images = input_data / shadow_ratio
    return images


def export(sess, input_pl, input_np, run_tensor):
    # Grab a single image and run it through inference
    output_np = sess.run(run_tensor, feed_dict={input_pl: input_np})
    return output_np


def kl_divergence(p, q):
    return numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))


def kl_div_for_ratios(mean, std):
    mean_upper_region = mean + std
    mean_down_region = mean - std
    min_base = min([numpy.min(mean), numpy.min(mean_upper_region), numpy.min(mean_down_region)])
    # Only adjust according to minimum, if it is less than zero
    if min_base > 0:
        min_base = 0
    mean = mean - min_base
    mean_down_region = mean_down_region - min_base
    mean_upper_region = mean_upper_region - min_base
    base_arr = numpy.ones_like(mean) - min_base
    mean_kl = abs(kl_divergence(mean, base_arr))
    mean_upper_kl = abs(kl_divergence(mean_upper_region, base_arr))
    mean_down_kl = abs(kl_divergence(mean_down_region, base_arr))
    return mean_kl


def calculate_stats_from_samples(sess, data_sample_list, images_x_input_tensor, generate_y_tensor,
                                 shadow_ratio, log_dir,
                                 current_iteration, plt_name):
    band_size = shadow_ratio.shape[0]
    iteration_count = len(data_sample_list)
    progress_bar = tqdm(total=iteration_count)
    total_band_ratio = numpy.zeros([iteration_count, band_size], dtype=float)
    inf_nan_value_count = 0
    for index in range(0, iteration_count):
        generated_y_data = export(sess, images_x_input_tensor, data_sample_list[index], generate_y_tensor)
        band_ratio = numpy.mean(generated_y_data / data_sample_list[index], axis=(0, 1))

        is_there_inf = numpy.any(numpy.isinf(band_ratio))
        is_there_nan = numpy.any(numpy.isnan(band_ratio))
        if is_there_inf or is_there_nan:
            inf_nan_value_count = inf_nan_value_count + 1
        else:
            total_band_ratio[index] = band_ratio
        progress_bar.update(1)

    progress_bar.close()
    mean = numpy.mean(total_band_ratio * shadow_ratio, axis=0)
    std = numpy.std(total_band_ratio * shadow_ratio, axis=0)
    print_overall_info(inf_nan_value_count, mean, numpy.mean(total_band_ratio, axis=0), std)
    plot_overall_info(mean, std, current_iteration, plt_name, log_dir)
    return kl_div_for_ratios(mean, std)


def load_samples_for_testing(loader, data_set, sample_count, neighborhood, shadow_map, fetch_shadows):
    # neighborhood aware indices finder
    band_size = loader.get_data_shape(data_set)[2] - 1
    data_sample_list = []
    shadow_check_val = 0

    if neighborhood > 0:
        shadow_map = shadow_map[neighborhood:-neighborhood, neighborhood:-neighborhood]

    if fetch_shadows:
        indices = numpy.where(shadow_map > shadow_check_val)
    else:
        indices = numpy.where(shadow_map == shadow_check_val)

    for index in range(0, sample_count):
        # Pick a random point
        random_indice_in_sample_list = random.randint(0, indices[0].size - 1)
        sample_indice_in_image = [indices[1][random_indice_in_sample_list], indices[0][random_indice_in_sample_list]]
        data_sample_list.append(loader.get_point_value(data_set, sample_indice_in_image)[:, :, 0:band_size])
    return data_sample_list


def plot_overall_info(mean, std, iteration, plt_name, log_dir):
    band_size = mean.shape[0]
    bands = linspace(1, band_size, band_size, dtype=numpy.int)
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = 14
    plt.scatter(bands, mean, label="mean ratio", s=10)
    plt.plot(bands, mean)
    lower_bound = mean - std
    upper_bound = mean + std
    plt.fill_between(bands, lower_bound, upper_bound, alpha=0.2)
    # plt.legend(loc='upper left')
    # plt.title("Band ratio")
    plt.xlabel("Spectral band index")
    plt.ylabel("Ratio between generated and original samples")
    plt.ylim([-1, 4])
    plt.yticks(list(range(-1, 5)))
    # plt.yticks(list(range(numpy.round(numpy.min(lower_bound)).astype(int),
    #                       numpy.round(numpy.max(upper_bound)).astype(int) + 1)))
    plt.grid()
    plt.savefig(os.path.join(log_dir, f"{plt_name}_{iteration}.pdf"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


def print_overall_info(inf_nan_value_count, mean, raw_mean, std):
    print("inf or nan value count: %i" % inf_nan_value_count)
    # print("Mean total ratio:")
    # print(raw_mean)
    print("Mean&std Generated vs Original Ratio: ")
    band_size = mean.shape[0]
    for band_index in range(0, band_size):
        prefix = ""
        postfix = ""
        end = "  "
        if band_index == 0:
            prefix = "[ "
        elif band_index == band_size - 1:
            postfix = " ]"

        if band_index % 5 == 1:
            end = '\n'
        print("%s%2.4f\u00B1%2.2f%s" % (prefix, mean[band_index], std[band_index], postfix), end=end)
