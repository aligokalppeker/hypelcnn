from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy
import tensorflow as tf
from absl import flags
from tifffile import imwrite
from tqdm import tqdm

from loader.GRSS2013DataLoader import GRSS2013DataLoader
from loader.GRSS2018DataLoader import GRSS2018DataLoader
from sr_data_models import construct_inference_graph, model_forward_generator_name, \
    model_backward_generator_name, create_generator_restorer, extract_common_normalizer

flags.DEFINE_string('checkpoint_path', '',
                    'CycleGAN checkpoint path created by sr_gann_train.py. '
                    '(e.g. "/mylogdir/model.ckpt-18442")')
flags.DEFINE_string('output_path', '',
                    'Output path to create tiff files. '
                    '(e.g. "/mylogdir/")')
flags.DEFINE_string('path', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

FLAGS = flags.FLAGS


def make_inference_graph(model_name, size):
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, size, size, 144], name='x')
    generated = construct_inference_graph(input_tensor, model_name)
    return input_tensor, generated


def export(sess, input_pl, input_np, output_tensor):
    # Grab a single image and run it through inference
    output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
    return output_np


def _validate_flags():
    flags.register_validator('checkpoint_path', bool,
                             'Must provide `checkpoint_path`.')
    flags.register_validator('output_path', bool, 'Must provide `output_path`.')


def main(_):
    numpy.set_printoptions(precision=5, suppress=True)

    _validate_flags()

    images_x_hwc_pl, generated_y = make_inference_graph(model_forward_generator_name, 10)
    images_y_hwc_pl, generated_x = make_inference_graph(model_backward_generator_name, 10)

    grss2013_loader = GRSS2013DataLoader(FLAGS.path)
    grss2013_data_set = grss2013_loader.load_data(0, False)

    grss2018_loader = GRSS2018DataLoader(FLAGS.path)
    grss2018_data_set = grss2018_loader.load_data(0, False)

    hsi2013_global_minimum, hsi2018_global_minimum, hsi2013_global_maximum, hsi2018_global_maximum = \
        extract_common_normalizer(grss2013_data_set.casi, grss2018_data_set.casi)

    output_scale = 5
    hsi_2013_spatial_repeat_count = 10
    hsi_2018_spatial_repeat_count = 4

    grss2013_band_count = grss2013_data_set.get_casi_band_count()
    grss2018_band_count = grss2018_data_set.get_casi_band_count()

    # grss2013_scene_shape = grss2013_data_set.get_scene_shape()
    # scene_first_dim_size = grss2013_scene_shape[0]
    # scene_second_dim_size = grss2013_scene_shape[1]

    scene_first_dim_size = 200
    scene_second_dim_size = 200
    first_dim_start_index = 15
    second_dim_start_index = 275

    dst_arr_type = numpy.uint16

    generated_grss2018_scene = numpy.zeros([scene_first_dim_size * output_scale,
                                            scene_second_dim_size * output_scale,
                                            grss2018_band_count], dtype=dst_arr_type)

    scale_down_indices = numpy.rint(numpy.linspace(0, grss2013_band_count, grss2018_band_count)).astype(int)
    progress_bar = tqdm(total=scene_first_dim_size * scene_second_dim_size)

    with tf.Session() as sess:
        create_generator_restorer().restore(sess, FLAGS.checkpoint_path)

        for first_dim_idx in range(first_dim_start_index, first_dim_start_index + scene_first_dim_size):
            for second_dim_idx in range(second_dim_start_index, second_dim_start_index + scene_second_dim_size):
                grss_2013_test_data = grss2013_loader.get_point_value(grss2013_data_set,
                                                                      [second_dim_idx, first_dim_idx])[:, :, 0:-1]
                grss_2013_test_data -= hsi2013_global_minimum
                grss_2013_test_data /= hsi2013_global_maximum.astype(numpy.float32)

                grss_2013_test_data = numpy.repeat(
                    numpy.repeat(grss_2013_test_data, hsi_2013_spatial_repeat_count, axis=0),
                    hsi_2013_spatial_repeat_count,
                    axis=1)

                grss_2013_test_data = numpy.expand_dims(grss_2013_test_data, axis=0)

                generate_data = True
                if generate_data:
                    generated_grss2018_data = convert_to_grss2018(generated_y, grss_2013_test_data,
                                                                  hsi2018_global_maximum, hsi2018_global_minimum,
                                                                  images_x_hwc_pl, output_scale,
                                                                  scale_down_indices, sess, dst_arr_type)
                else:
                    coeff = numpy.fromstring(
                        "1.5272297  1.9090617  2.3563638  1.5291632  1.7045546  1.924786 \
                         1.344934   1.4734588  2.0304475  1.6566893  1.8031071  1.9551347 \
                         1.8484792  1.9217775  1.8733425  1.7170434  1.8558326  2.0305467 \
                         1.9258988  1.9903326  2.0316174  2.0182045  2.025065   2.0396416 \
                         2.1053767  2.1198342  2.036646   2.0866528  2.1329854  2.0902588 \
                         2.1245925  2.1657488  2.1380842  2.0271137  2.1508708  2.2272987 \
                         2.2603593  2.2521007  2.2519095  2.3009799  2.3295002  2.305151  \
                         2.375265   2.342448   2.3103044  2.360908   2.3820717  2.3103168 \
                         2.2553535  2.2993126  2.3587887  2.4618702  2.4336336  2.3843746 \
                         2.5018375  2.447398   2.3952596  2.4870105  2.486155   2.4336276 \
                         2.4123988  2.3942418  2.3538475  2.5148015  2.5055854  2.4732149 \
                         2.6295698  2.5936677  2.2724116  2.060744   2.1818426  2.3007538 \
                         2.1846933  2.358493   2.2681623  1.7930862  1.9298902  2.0532281 \
                         2.1927633  2.426196   2.622884   3.0671167  3.0812511  2.6697197 \
                         1.7144157  2.488118   2.8853543  2.7110586  2.6905243  2.5987196 \
                         2.5420787  2.5291255  2.493314   2.6517634  2.474208   1.985119  \
                         2.0433202  2.1246142  2.1915617  2.2790527  2.4718482  2.5622425 \
                         2.68112    2.5816808  2.684341   2.7369573  2.6635604  2.7210138 \
                         2.7021546  2.6808908  2.6391733  2.8454063  2.1891053  1.9089663 \
                         2.2308357  1.9050708  1.8015311  2.5490248  2.367938   1.3678108 \
                         0.87415427 1.0507026  1.0382721  0.94069296 1.0381097  1.2658843 \
                         1.2275281  1.7764775  2.0466998  1.7822698  2.106053   2.4231715 \
                         2.452284   2.5351257  2.524643   2.4530883  2.4923446  2.523102  \
                         2.4890015  2.4949672  2.508757   2.461551   2.479739   2.5024395", dtype=float, count=-1,
                        sep=' ')
                    # coeff = numpy.ones(144, dtype=float)
                    generated_grss2018_data = cv2.resize(grss_2013_test_data[0] / coeff,
                                                         (output_scale, output_scale),
                                                         interpolation=cv2.INTER_LINEAR)
                    generated_grss2018_data = (((numpy.take(generated_grss2018_data, scale_down_indices,
                                                            axis=2)) * hsi2018_global_maximum.astype(
                        numpy.float32)) + hsi2018_global_minimum) \
                        .astype(dst_arr_type)

                target_first_dim_idx = first_dim_idx - first_dim_start_index
                target_second_dim_idx = second_dim_idx - second_dim_start_index
                generated_grss2018_scene[
                target_first_dim_idx * output_scale:((target_first_dim_idx + 1) * output_scale),
                target_second_dim_idx * output_scale:((target_second_dim_idx + 1) * output_scale)] = \
                    generated_grss2018_data

                progress_bar.update(1)
                # print(generated_grss2018_data[0])
                # generated_x_data = export(sess, images_y_hwc_pl, generated_grss2018_data, generated_x)

    progress_bar.close()
    imwrite(os.path.join(FLAGS.output_path, "generated_image.tif"),
            generated_grss2018_scene, planarconfig='contig')


def convert_to_grss2018(generated_y, grss_2013_test_data, dataset_global_maximum, dataset_global_minimum,
                        images_x_hwc_pl, output_scale, scale_down_indices, sess, dst_arr_type):
    generated_grss2018_data = export(sess, images_x_hwc_pl, grss_2013_test_data, generated_y)[0]
    generated_grss2018_data = numpy.take(generated_grss2018_data, scale_down_indices, axis=2)
    generated_grss2018_data = ((generated_grss2018_data * dataset_global_maximum.astype(numpy.float32)) +
                               dataset_global_minimum).astype(dst_arr_type)
    generated_grss2018_data = cv2.resize(generated_grss2018_data, (output_scale, output_scale),
                                         interpolation=cv2.INTER_LINEAR)
    return generated_grss2018_data


if __name__ == '__main__':
    tf.app.run()
