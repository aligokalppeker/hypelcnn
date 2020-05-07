import csv
import os
import sys

import numpy

from utilities.stat_extractor import extract_statistics_info, get_conf_list_from_directory


def main():
    metrics_holder_list = []
    class_dist_info_list = []
    method_name_list = []
    for index, directory in enumerate(sys.argv):
        if index > 1:
            metrics_holder_list.append(extract_statistics_info(get_conf_list_from_directory(directory)))
        elif index == 1:
            method_name_filename = os.path.join(sys.argv[1], "method_name_list.csv")
            with open(method_name_filename, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    for col in row:
                        method_name_list.append(col)

            class_dist_info_filename = os.path.join(sys.argv[1], "class_dist_info.csv")
            with open(class_dist_info_filename, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    class_dist_info_list.append(row[0])

    metrics_instance_count = len(metrics_holder_list)
    if metrics_instance_count > 0:
        class_count = len(metrics_holder_list[0].sample_count)

        header_row_str = "Methods "
        for metrics_instance_index in range(0, metrics_instance_count):
            header_row_str = header_row_str + "&" + method_name_list[metrics_instance_index] + " "
        print(header_row_str + "\\\\")

        for class_index in range(0, class_count):
            row_str = class_dist_info_list[class_index] + " "
            for metrics_instance_index in range(0, metrics_instance_count):
                aa_array = metrics_holder_list[metrics_instance_index].aa_array
                # sample_count = metrics_holder_list[metrics_instance_index].sample_count
                row_str = row_str + "& %.4f $\\pm{%.4f} " % (
                    numpy.mean(aa_array, axis=0)[class_index], numpy.std(aa_array, axis=0)[class_index])
            print(row_str + "\\\\")

        oa_str = "OA "
        for metrics_instance_index in range(0, metrics_instance_count):
            oa_str = oa_str + "& %.4f $\\pm{%.4f} " \
                     % (numpy.mean(metrics_holder_list[metrics_instance_index].oa_array),
                        numpy.std(metrics_holder_list[metrics_instance_index].oa_array))
        print(oa_str + "\\\\")

        aa_str = "AA "
        for metrics_instance_index in range(0, metrics_instance_count):
            aa_str = aa_str + "& %.4f $\\pm{%.4f} " \
                     % (numpy.mean(numpy.mean(metrics_holder_list[metrics_instance_index].aa_array, axis=1)),
                        numpy.std(numpy.mean(metrics_holder_list[metrics_instance_index].aa_array, axis=1)))
        print(aa_str + "\\\\")

        kappa_str = "Kappa "
        for metrics_instance_index in range(0, metrics_instance_count):
            kappa_str = kappa_str + "& %.4f $\\pm{%.4f} " \
                        % (numpy.mean(metrics_holder_list[metrics_instance_index].kappa_array),
                           numpy.std(metrics_holder_list[metrics_instance_index].kappa_array))
        print(kappa_str + "\\\\")


if __name__ == '__main__':
    main()
