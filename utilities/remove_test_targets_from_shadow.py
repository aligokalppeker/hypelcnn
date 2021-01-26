import argparse
import os

import matplotlib.pyplot as plt
from tifffile import imwrite

from cmd_parser import parse_cmd
from common_nn_operations import get_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', nargs='?', const=True, type=str,
                        default=os.path.dirname(__file__),
                        help='Path for saving output images')
    flags = parse_cmd(parser)

    loader_name = flags.loader_name
    loader = get_class(loader_name + '.' + loader_name)(flags.path)

    sample_set = loader.load_samples(0.1)
    data_set = loader.load_data(0, True)
    shadow_map, _ = loader.load_shadow_map(0, data_set)

    plt.imshow(shadow_map * 255)
    plt.title("figure_name"), plt.xticks([]), plt.yticks([])
    plt.show()

    non_shadow_test_sample = 0
    for point in sample_set.validation_targets:
        if shadow_map[point[1], point[0]] == 1:
            shadow_map[point[1], point[0]] = 0
        else:
            non_shadow_test_sample = non_shadow_test_sample + 1

    plt.imshow(shadow_map * 255)
    plt.title("figure_name"), plt.xticks([]), plt.yticks([])
    plt.show()

    imwrite("shadow_map.tif", shadow_map, planarconfig='contig')


if __name__ == '__main__':
    main()
