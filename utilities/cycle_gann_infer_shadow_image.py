from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils
import os

import numpy
import tensorflow as tf
from absl import flags
from tifffile import imwrite
from tqdm import tqdm

from common_nn_operations import get_class
from shadow_data_generator import construct_inference_graph, model_forward_generator_name, \
    create_generator_restorer, model_backward_generator_name
from utilities.hsi_rgb_converter import get_rgb_from_hsi

required_tensorflow_version = "1.14.0"
if distutils.version.LooseVersion(tf.__version__) < distutils.version.LooseVersion(required_tensorflow_version):
    tfgan = tf.contrib.gan
else:
    pass

flags.DEFINE_string('checkpoint_path', '',
                    'GAN checkpoint path created by gan_train_for_shadow.py. '
                    '(e.g. "/mylogdir/model.ckpt-18442")')

flags.DEFINE_string('loader_name', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

flags.DEFINE_string('path', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

flags.DEFINE_string('output_path', '',
                    'Output path to create tiff files. '
                    '(e.g. "/mylogdir/")')

flags.DEFINE_string('make_them_shadow', "",
                    'makes the scene shadowed(shadow), non shadowed(deshadow), or anything(none)')

flags.DEFINE_bool('convert_all', False,
                  'Whether to convert filtered pixels(shadowed or not) or all.')

FLAGS = flags.FLAGS


def make_inference_graph(model_name, wrapper, element_size, clip_invalid_values=True):
    input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='x')
    generated = wrapper.construct_inference_graph(input_tensor, model_name, clip_invalid_values)
    return input_tensor, generated


def export(sess, input_pl, input_np, output_tensor):
    # Grab a single image and run it through inference
    output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
    return output_np


def _validate_flags():
    flags.register_validator('checkpoint_path', bool,
                             'Must provide `checkpoint_path`.')


def main(_):
    _validate_flags()
    make_them_shadow = FLAGS.make_them_shadow

    loader_name = FLAGS.loader_name
    loader = get_class(loader_name + '.' + loader_name)(FLAGS.path)
    data_set = loader.load_data(0, True)
    offset = data_set.casi_min
    multiplier = data_set.casi_max
    if offset is None:
        offset = 0
    if multiplier is None:
        multiplier = 1
    target_data_type = loader.get_original_data_type()
    shadow_map, shadow_ratio = loader.load_shadow_map(0, data_set)

    scene_shape = loader.get_scene_shape(data_set)
    element_size = loader.get_data_shape(data_set)
    element_size = [element_size[0], element_size[1], element_size[2] - 1]

    convert_only_the_convenient_pixels = not FLAGS.convert_all
    if make_them_shadow == "shadow":
        model_name = model_forward_generator_name
        sign_to_filter_in_shadow_map = 0
    elif make_them_shadow == "deshadow":
        model_name = model_backward_generator_name
        sign_to_filter_in_shadow_map = 1
    else:
        model_name = None
        sign_to_filter_in_shadow_map = -1
        make_them_shadow = "none"

    images_hwc_pl, generated_output = make_inference_graph(model_name, element_size, clip_invalid_values=False)

    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    with tf.Session() as sess:
        if make_them_shadow != "none":
            create_generator_restorer().restore(sess, FLAGS.checkpoint_path)

        screen_size_first_dim = scene_shape[0]
        screen_size_sec_dim = scene_shape[1]

        progress_bar = tqdm(total=screen_size_first_dim * screen_size_sec_dim)
        band_size = element_size[2]
        hsi_image = numpy.zeros([screen_size_first_dim, screen_size_sec_dim, band_size], dtype=target_data_type)
        for first_idx in range(0, screen_size_first_dim):
            for second_idx in range(0, screen_size_sec_dim):
                input_data = loader.get_point_value(data_set, [second_idx, first_idx])[:, :, 0:band_size]

                if not convert_only_the_convenient_pixels or shadow_map[
                    first_idx, second_idx] == sign_to_filter_in_shadow_map:
                    generated_y_data = export(sess, images_hwc_pl, input_data, generated_output)
                else:
                    generated_y_data = input_data

                hsi_image[first_idx, second_idx, :] = \
                    ((generated_y_data * multiplier) + offset).astype(target_data_type)

                progress_bar.update(1)

        progress_bar.close()

        if convert_only_the_convenient_pixels:
            convert_region_suffix = ""
        else:
            convert_region_suffix = "all"

        imwrite(os.path.join(FLAGS.output_path, f"shadow_image_{make_them_shadow}_{convert_region_suffix}.tif"),
                hsi_image, planarconfig='contig')

        hsi_image = hsi_image.astype(float)
        hsi_image -= offset
        hsi_image /= multiplier
        hsi_as_rgb = (get_rgb_from_hsi(loader.get_band_measurements(), hsi_image) * 256).astype(numpy.uint8)
        imwrite(os.path.join(FLAGS.output_path, f"shadow_image_rgb_{make_them_shadow}_{convert_region_suffix}.tif"),
                hsi_as_rgb)


if __name__ == '__main__':
    tf.app.run()
