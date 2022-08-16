import argparse
import os

import numpy
import tensorflow as tf
from tifffile import imwrite
from tqdm import tqdm

from common.cmd_parser import add_parse_cmds_for_loaders, add_parse_cmds_for_loggers, type_ensure_strtobool
from common.common_nn_ops import set_all_gpu_config, get_loader_from_name
from gan.wrappers.cut_wrapper import CUTInferenceWrapper
from gan.wrappers.cycle_gan_wrapper import CycleGANInferenceWrapper
from gan.wrappers.gan_wrapper import GANInferenceWrapper
from common.hsi_rgb_converter import get_rgb_from_hsi


def add_parse_cmds_for_app(parser):
    parser.add_argument("--gan_type", nargs="?", type=str, default="cycle_gan",
                        help="Gan type to train, possible values; cycle_gan, gan_x2y and gan_y2x")
    parser.add_argument("--make_them_shadow", nargs="?", type=str, default="",
                        help="makes the scene shadowed(shadow), non shadowed(deshadow), or empty(none)")
    parser.add_argument("--convert_all", nargs="?", type=type_ensure_strtobool, default=False,
                        help="Whether to convert filtered pixels(shadowed or not) or all.")


def main(_):
    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loaders(parser)
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_app(parser)
    flags, unparsed = parser.parse_known_args()

    make_them_shadow = flags.make_them_shadow

    loader = get_loader_from_name(flags.loader_name, flags.path)
    data_set = loader.load_data(0, True)
    target_data_type = data_set.get_unnormalized_casi_dtype()
    shadow_map, _ = loader.load_shadow_map(0, data_set)

    scene_shape = data_set.get_scene_shape()
    element_size = data_set.get_data_shape()
    element_size = [element_size[0], element_size[1], data_set.get_casi_band_count()]

    convert_only_the_convenient_pixels = not flags.convert_all
    if make_them_shadow == "shadow":
        shadow = True
        sign_to_filter_in_shadow_map = 0
    elif make_them_shadow == "deshadow":
        shadow = False
        sign_to_filter_in_shadow_map = 1
    else:
        shadow = True
        sign_to_filter_in_shadow_map = -1
        make_them_shadow = "none"

    gan_inference_wrapper_dict = get_wrapper_dict()

    input_tensor, output_tensor = gan_inference_wrapper_dict[flags.gan_type].make_inference_graph(data_set,
                                                                                                  shadow,
                                                                                                  clip_invalid_values=False)

    set_all_gpu_config()
    with tf.compat.v1.Session() as sess:
        if make_them_shadow != "none":
            gan_inference_wrapper_dict[flags.gan_type].create_generator_restorer().restore(sess, flags.base_log_path)

        screen_size_first_dim = scene_shape[0]
        screen_size_sec_dim = scene_shape[1]

        progress_bar = tqdm(total=screen_size_first_dim * screen_size_sec_dim)
        band_size = element_size[2]
        hsi_image = numpy.zeros([screen_size_first_dim, screen_size_sec_dim, band_size], dtype=target_data_type)
        for first_idx in range(0, screen_size_first_dim):
            for second_idx in range(0, screen_size_sec_dim):
                input_data = numpy.expand_dims(
                    loader.get_point_value(data_set, [second_idx, first_idx])[:, :, 0:band_size], axis=0)

                if not convert_only_the_convenient_pixels or shadow_map[
                    first_idx, second_idx] == sign_to_filter_in_shadow_map:
                    generated_y_data = sess.run(output_tensor, feed_dict={input_tensor: input_data})
                else:
                    generated_y_data = input_data

                hsi_image[first_idx, second_idx, :] = \
                    ((generated_y_data * data_set.casi_max) + data_set.casi_min).astype(target_data_type)

                progress_bar.update(1)

        progress_bar.close()

        convert_region_suffix = "" if convert_only_the_convenient_pixels else "_all"
        chkpnt_num_str = flags.base_log_path.rsplit("-", 1)[-1]

        hsi_image_save_path = os.path.join(flags.output_path,
                                           f"shadow_image_{make_them_shadow}_{chkpnt_num_str}{convert_region_suffix}.tif")
        print(f"Saving output to {hsi_image_save_path}")
        imwrite(hsi_image_save_path, hsi_image, planarconfig="contig")

        hsi_image = hsi_image.astype(float)
        hsi_image -= data_set.casi_min
        hsi_image /= data_set.casi_max
        hsi_as_rgb = (get_rgb_from_hsi(loader.get_band_measurements(), hsi_image) * 255).astype(numpy.uint8)
        hsi_image_rgb_save_path = os.path.join(flags.output_path,
                                               f"shadow_image_rgb_{make_them_shadow}_{chkpnt_num_str}_{convert_region_suffix}.tif")
        print(f"Saving output RGB to {hsi_image_rgb_save_path}")
        imwrite(hsi_image_rgb_save_path, hsi_as_rgb)


def get_wrapper_dict():
    gan_inference_wrapper_dict = {"cycle_gan": CycleGANInferenceWrapper(),
                                  "gan_x2y": GANInferenceWrapper(fetch_shadows=False),
                                  "gan_y2x": GANInferenceWrapper(fetch_shadows=True),
                                  "cut_x2y": CUTInferenceWrapper(fetch_shadows=False),
                                  "cut_y2x": CUTInferenceWrapper(fetch_shadows=True)}
    return gan_inference_wrapper_dict


if __name__ == '__main__':
    tf.compat.v1.app.run()
