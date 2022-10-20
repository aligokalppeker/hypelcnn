import argparse

import numpy

from common.cmd_parser import add_parse_cmds_for_loggers, add_parse_cmds_for_loaders
from common.common_nn_ops import get_loader_from_name
from gan.wrapper_registry import get_sampling_map
from gan.wrappers.gan_common import read_hsi_data, plot_overall_info


def main():
    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_loaders(parser)
    parser.add_argument("--pairing_method", nargs="?", type=str, default="random",
                        help="Pairing method for the shadowed and non-shadowed samples. "
                             "Opts: random, target, dummy, neighbour")
    flags, unparsed = parser.parse_known_args()

    neighborhood = 0
    loader = get_loader_from_name(flags.loader_name, flags.path)
    data_set = loader.load_data(neighborhood, True)
    shadow_map, shadow_ratio = loader.load_shadow_map(neighborhood, data_set)

    normal_data_as_matrix, shadow_data_as_matrix = read_hsi_data(loader, data_set, shadow_map, flags.pairing_method,
                                                                 get_sampling_map())
    normal_data_as_matrix = numpy.squeeze(normal_data_as_matrix)
    shadow_data_as_matrix = numpy.squeeze(shadow_data_as_matrix)

    ratio = shadow_data_as_matrix / normal_data_as_matrix
    ratio = ratio[numpy.isfinite(ratio).all(axis=1)]
    mean_res = numpy.mean(ratio, axis=0)
    std_res = numpy.std(ratio, axis=0)
    plot_overall_info(loader.get_band_measurements(),
                      mean_res,
                      mean_res - std_res,
                      mean_res + std_res,
                      0, f"{flags.loader_name.lower()}_{flags.pairing_method.lower()}", "./")


if __name__ == '__main__':
    main()
