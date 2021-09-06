from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils

import numpy
import tensorflow as tf
from absl import flags

from common_nn_operations import get_class
from shadow_data_generator import construct_inference_graph, model_forward_generator_name, \
    model_backward_generator_name, create_generator_restorer, load_samples_for_testing, calculate_stats_from_samples

required_tensorflow_version = "1.14.0"
if distutils.version.LooseVersion(tf.__version__) < distutils.version.LooseVersion(required_tensorflow_version):
    tfgan = tf.contrib.gan
else:
    pass

flags.DEFINE_integer('neighborhood', 0, 'Neighborhood of samples.')
flags.DEFINE_integer('number_of_samples', 6000, 'Number of samples.')
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
    images_x_input_tensor, generate_y_tensor = make_inference_graph(model_forward_generator_name, element_size,
                                                                    clip_invalid_values=False)
    images_y_input_tensor, generate_x_tensor = make_inference_graph(model_backward_generator_name, element_size,
                                                                    clip_invalid_values=False)

    # test_x_data = numpy.full([1, 1, band_size], fill_value=1.0, dtype=float)
    # print_tensors_in_checkpoint_file(FLAGS.checkpoint_path, tensor_name='ModelX2Y', all_tensors=True)
    iteration_count = FLAGS.number_of_samples
    shadow_map, shadow_ratio = loader.load_shadow_map(neighborhood, data_set)
    shadow_ratio = shadow_ratio[0:-1]

    data_sample_array_for_shadow = load_samples_for_testing(loader, data_set, iteration_count, neighborhood,
                                                            shadow_map,
                                                            fetch_shadows=False)
    data_sample_array_for_deshadow = load_samples_for_testing(loader, data_set, iteration_count, neighborhood,
                                                              shadow_map,
                                                              fetch_shadows=False)

    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    with tf.Session() as sess:
        create_generator_restorer().restore(sess, FLAGS.checkpoint_path)
        calculate_stats_from_samples(sess, data_sample_array_for_shadow, images_x_input_tensor, generate_y_tensor,
                                     shadow_ratio, "./", 0, plt_name=f"{loader_name.lower()}_band_ratio_shadowed")

        calculate_stats_from_samples(sess, data_sample_array_for_deshadow, images_y_input_tensor, generate_x_tensor,
                                     1/shadow_ratio, "./", 0, plt_name=f"{loader_name.lower()}_band_ratio_deshadowed")

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
