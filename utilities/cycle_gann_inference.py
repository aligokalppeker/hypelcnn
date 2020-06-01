from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy
import tensorflow as tf
from absl import flags

from GRSS2013DataLoader import GRSS2013DataLoader
from shadow_data_generator import construct_inference_graph, model_forward_generator_name, \
    model_backward_generator_name, create_generator_restorer

tfgan = tf.contrib.gan

flags.DEFINE_string('checkpoint_path', '',
                    'CycleGAN checkpoint path created by cycle_gann_train.py. '
                    '(e.g. "/mylogdir/model.ckpt-18442")')

FLAGS = flags.FLAGS


def make_inference_graph(model_name, clip_invalid_values=True):
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 1, 144], name='x')
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

    _validate_flags()

    images_x_hwc_pl, generated_y = make_inference_graph(model_forward_generator_name, clip_invalid_values=False)
    images_y_hwc_pl, generated_x = make_inference_graph(model_backward_generator_name, clip_invalid_values=False)

    # print_tensors_in_checkpoint_file(FLAGS.checkpoint_path, tensor_name='ModelX2Y', all_tensors=True)

    loader = GRSS2013DataLoader('C:/GoogleDriveBack/PHD/Tez/Source')
    data_set = loader.load_data(0, True)

    with tf.Session() as sess:
        create_generator_restorer().restore(sess, FLAGS.checkpoint_path)

        shadow_map, shadow_ratio = loader._load_shadow_map(0, data_set.concrete_data)
        indices = numpy.where(shadow_map == 0)

        iteration_count = 1000
        total_band_ratio = numpy.zeros([1, 1, 144], dtype=float)
        for i in range(0, iteration_count):
            # Pick a random point
            data_indice = random.randint(0, indices[0].size - 1)

            # test_x_data = numpy.random.rand(1, 1, 144)

            test_indice = [indices[1][data_indice], indices[0][data_indice]]
            test_x_data = loader.get_point_value(data_set, test_indice)
            test_x_data = test_x_data[:, :, 0:144]

            generated_y_data = export(sess, images_x_hwc_pl, test_x_data, generated_y)
            generated_x_data = export(sess, images_y_hwc_pl, generated_y_data, generated_x)

            band_ratio = generated_y_data / test_x_data
            shadow_calc_ratio = band_ratio * shadow_ratio[0:144]

            is_there_inf = numpy.any(numpy.isinf(band_ratio))
            if is_there_inf:
                print("inf value")
            else:
                total_band_ratio = total_band_ratio + band_ratio

            if iteration_count == 1:
                print_info(test_x_data, generated_x_data, generated_y_data, band_ratio, shadow_calc_ratio)

        print("Mean total ratio")
        print(total_band_ratio / iteration_count)
        print("Mean Generated vs Original Ratio")
        print(total_band_ratio / iteration_count * shadow_ratio[0:144])

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
