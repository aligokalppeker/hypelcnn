import argparse
import os

from tifffile import imsave

from cmd_parser import parse_cmd
from common_nn_operations import get_class, create_colored_image, create_target_image_via_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', nargs='?', const=True, type=str,
                        default=os.path.dirname(__file__),
                        help='Path for saving output images')
    flags = parse_cmd(parser)

    loader_name = flags.loader_name
    loader = get_class(loader_name + '.' + loader_name)(flags.path)
    sample_set = loader.load_samples(0.1, 0.1)
    data_set = loader.load_data(0, False)
    scene_shape = data_set.get_scene_shape()

    imsave(os.path.join(flags.output_path, "result_colorized.tif"),
           create_colored_image(create_target_image_via_samples(sample_set, scene_shape),
                                loader.get_target_color_list()))


if __name__ == '__main__':
    main()
