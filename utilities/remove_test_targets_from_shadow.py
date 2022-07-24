import argparse

import matplotlib.pyplot as plt
from tifffile import imwrite

from common.cmd_parser import add_parse_cmds_for_classification, add_parse_cmds_for_loggers
from common.common_nn_ops import get_loader_from_name


def main():
    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_classification(parser)
    flags, unparsed = parser.parse_known_args()

    loader = get_loader_from_name(flags.loader_name, flags.path)

    sample_set = loader.load_samples(0.1, 0.1)
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
