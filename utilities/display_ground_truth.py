import argparse
import os

from tifffile import imsave

from cmd_parser import add_parse_cmds_for_classification, add_parse_cmds_for_loggers
from common_nn_operations import create_colored_image, create_target_image_via_samples, get_loader_from_name


def main():
    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_classification(parser)
    flags, unparsed = parser.parse_known_args()

    loader = get_loader_from_name(flags.loader_name, flags.path)
    sample_set = loader.load_samples(0.1, 0.1)
    data_set = loader.load_data(0, False)
    scene_shape = data_set.get_scene_shape()

    imsave(os.path.join(flags.output_path, "result_colorized.tif"),
           create_colored_image(create_target_image_via_samples(sample_set, scene_shape),
                                loader.get_target_color_list()))




if __name__ == '__main__':
    main()
