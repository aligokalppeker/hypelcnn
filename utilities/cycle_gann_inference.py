from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils
import random

import numpy
import tensorflow as tf
from absl import flags

from common_nn_operations import get_class
from shadow_data_generator import construct_inference_graph, model_forward_generator_name, \
    model_backward_generator_name, create_generator_restorer

required_tensorflow_version = "1.14.0"
if distutils.version.LooseVersion(tf.__version__) < distutils.version.LooseVersion(required_tensorflow_version):
    tfgan = tf.contrib.gan
else:
    pass

flags.DEFINE_integer('neighborhood', 0, 'Neighborhood of samples.')
flags.DEFINE_string('checkpoint_path', '',
                    'CycleGAN checkpoint path created by cycle_gann_train.py. '
                    '(e.g. "/mylogdir/model.ckpt-18442")')

flags.DEFINE_string('loader_name', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

flags.DEFINE_string('path', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

FLAGS = flags.FLAGS


def make_inference_graph(model_name, element_size, clip_invalid_values=True):
    input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='x')
    generated = construct_inference_graph(input_tensor, model_name, clip_invalid_values)
    return input_tensor, generated


def export(sess, input_pl, input_np, output_tensor):
    # Grab a single image and run it through inference
    output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
    return output_np


def _validate_flags():
    flags.register_validator('checkpoint_path', bool,
                             'Must provide `checkpoint_path`.')


def main(_):
    numpy.set_printoptions(precision=5, suppress=True)
    neighborhood = FLAGS.neighborhood

    _validate_flags()

    loader_name = FLAGS.loader_name
    loader = get_class(loader_name + '.' + loader_name)(FLAGS.path)
    data_set = loader.load_data(neighborhood, True)

    element_size = loader.get_data_shape(data_set)
    element_size = [element_size[0], element_size[1], element_size[2] - 1]
    images_x_hwc_pl, generated_y = make_inference_graph(model_forward_generator_name, element_size,
                                                        clip_invalid_values=False)
    images_y_hwc_pl, generated_x = make_inference_graph(model_backward_generator_name, element_size,
                                                        clip_invalid_values=False)

    # print_tensors_in_checkpoint_file(FLAGS.checkpoint_path, tensor_name='ModelX2Y', all_tensors=True)

    with tf.Session() as sess:
        create_generator_restorer().restore(sess, FLAGS.checkpoint_path)

        shadow_map, shadow_ratio = loader.load_shadow_map(neighborhood, data_set)
        # neighborhood aware indice finder
        if neighborhood > 0:
            indices = numpy.where(shadow_map[neighborhood:-neighborhood, neighborhood:-neighborhood] == 0)
        else:
            indices = numpy.where(shadow_map == 0)

        iteration_count = 3000 * 2
        band_size = element_size[2]
        total_band_ratio = numpy.zeros([1, 1, band_size], dtype=float)
        for i in range(0, iteration_count):
            # Pick a random point
            data_indice = random.randint(0, indices[0].size - 1)

            test_indice = [indices[1][data_indice], indices[0][data_indice]]
            test_x_data = loader.get_point_value(data_set, test_indice)
            test_x_data = test_x_data[:, :, 0:band_size]

            # test_x_data = numpy.full([1, 1, band_size], fill_value=1.0, dtype=float)

            generated_y_data = export(sess, images_x_hwc_pl, test_x_data, generated_y)
            generated_x_data = export(sess, images_y_hwc_pl, generated_y_data, generated_x)

            band_ratio = numpy.mean(generated_y_data / test_x_data, axis=(0, 1))
            shadow_calc_ratio = band_ratio * shadow_ratio[0:band_size]

            is_there_inf = numpy.any(numpy.isinf(band_ratio))
            is_there_nan = numpy.any(numpy.isnan(band_ratio))
            if is_there_inf or is_there_nan:
                print("inf or nan value")
            else:
                total_band_ratio = total_band_ratio + band_ratio

            if iteration_count == 1:
                print_info(test_x_data, generated_x_data, generated_y_data, band_ratio, shadow_calc_ratio)

        print("Mean total ratio")
        print(total_band_ratio / iteration_count)
        print("Mean Generated vs Original Ratio")
        print(total_band_ratio / iteration_count * shadow_ratio[0:band_size])

        # normal_data_as_matrix, shadow_data_as_matrix = loader.get_targetbased_shadowed_normal_data(data_set,
        #                                                                                            loader,
        #                                                                                            shadow_map,
        #                                                                                            loader.load_samples(
        #                                                                                                0.1))
        # # normal_data_as_matrix, shadow_data_as_matrix = loader.get_all_shadowed_normal_data(data_set,
        # #                                                                             loader,
        # #                                                                             shadow_map)
        # print("Target based shadow index")
        # print(1 / numpy.squeeze(numpy.mean(normal_data_as_matrix, axis=0) / numpy.mean(shadow_data_as_matrix, axis=0)))


def print_info(test_x_data, generated_x_data, generated_y_data, band_ratio, shadow_calc_ratio):
    print("Original Data:")
    print(test_x_data)
    print(str(numpy.mean(test_x_data)) + "," + str(numpy.std(test_x_data)))
    print("Original Data=>Generated Shadow Data:")
    print(generated_y_data)
    print(str(numpy.mean(generated_y_data)) + "," + str(numpy.std(generated_y_data)))
    print("Generated Shadow Data=>Generated Original Data:")
    print(generated_x_data)
    print(str(numpy.mean(generated_x_data)) + "," + str(numpy.std(generated_x_data)))
    print("Generated Band ratio:")
    print(band_ratio)
    print("Generated vs Original Ratio")
    print(shadow_calc_ratio)


if __name__ == '__main__':
    tf.app.run()
