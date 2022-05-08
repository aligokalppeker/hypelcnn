import csv
import os
import sys
from collections import namedtuple

import numpy

from stat_extractor import extract_statistics_info, get_conf_list_from_directory

TableInfo = namedtuple('TableInfo', ['title', 'label'])

FLOAT_FORMAT = "%.2f"
PERCENTILE_COEFF = 100
MATRIX_OA_INDEX = 0
MATRIX_AA_INDEX = 1
MATRIX_KAPPA_INDEX = 2

PERFORMANCE_STR = "Performance"
CLASSES_STR = "Classes (Train/Test)"


def main():
    one_column = False
    metrics_holder_list = []
    class_dist_info_list = []
    method_name_list = []
    table_info = None
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

            table_info_filename = os.path.join(sys.argv[1], "table_info.csv")
            with open(table_info_filename, 'r') as file:
                reader = csv.reader(file)
                col_list = []
                for row in reader:
                    for col in row:
                        col_list.append(col)
                table_info = TableInfo(title=col_list[0], label=col_list[1])

    print_column_wise_methods(method_name_list, class_dist_info_list, metrics_holder_list, table_info, one_column)
    # print_row_wise_methods(method_name_list, class_dist_info_list, metrics_holder_list, table_info, one_column)


def get_lengths_from_metrics(metrics_holder_list):
    metrics_instance_count = len(metrics_holder_list)
    class_count = len(metrics_holder_list[0].sample_count)
    return class_count, metrics_instance_count


def convert_metrics_holder_list_to_matrix(metrics_holder_list):
    class_count, metrics_instance_count = get_lengths_from_metrics(metrics_holder_list)

    matrix = numpy.zeros([metrics_instance_count, class_count], dtype=numpy.float)
    for metrics_instance_index in range(0, metrics_instance_count):
        for class_index in range(0, class_count):
            matrix[metrics_instance_index][class_index], std = calculate_class_based_mean_aa_metric_for_cell(
                metrics_holder_list, metrics_instance_index, class_index)
    return matrix


def convert_overall_metrics_to_matrix(metrics_holder_list):
    class_count, metrics_instance_count = get_lengths_from_metrics(metrics_holder_list)
    matrix = numpy.zeros([metrics_instance_count, 3], dtype=numpy.float)
    for metrics_instance_index in range(0, metrics_instance_count):
        matrix[metrics_instance_index][MATRIX_OA_INDEX], std = calculate_oa_metric_for_instance(metrics_holder_list,
                                                                                                metrics_instance_index)
        matrix[metrics_instance_index][MATRIX_AA_INDEX], std = calculate_aa_metric_for_instance(metrics_holder_list,
                                                                                                metrics_instance_index)
        matrix[metrics_instance_index][MATRIX_KAPPA_INDEX], std = calculate_kappa_metric_for_instance(
            metrics_holder_list,
            metrics_instance_index)
    return matrix


def extract_bold_values(metrics_as_matrix):
    class_count = metrics_as_matrix.shape[1]
    matrix = numpy.zeros(metrics_as_matrix.shape, dtype=int)
    for class_index in range(0, class_count):
        compared_vals = metrics_as_matrix[:, class_index]
        # use this instead of numpy.argmax to find max idx with multiple occurences
        idx_list = numpy.argwhere(compared_vals == numpy.amax(compared_vals))
        for idx in idx_list:
            matrix[idx, class_index] = 1
    return matrix


def print_row_wise_methods(method_name_list, class_dist_info_list, metrics_holder_list, table_info, one_column):
    class_count, metrics_instance_count = get_lengths_from_metrics(metrics_holder_list)
    boldness_matrix = extract_bold_values(convert_metrics_holder_list_to_matrix(metrics_holder_list))
    overall_metric_boldness_matrix = extract_bold_values(convert_overall_metrics_to_matrix(metrics_holder_list))

    col_count = 1 + class_count + 3  # title + class count + (OA;AA;KAPPA)
    begin_tabular(table_info, col_count, one_column)
    draw_double_line()
    print_combine_column(col_count, PERFORMANCE_STR, CLASSES_STR)

    header_row_str = "\\cline{2-%i} " % col_count
    for class_dist_info in class_dist_info_list:
        header_row_str = header_row_str + " & " + class_dist_info
    header_row_str = header_row_str + " & OA & AA & KAPPA"
    print(header_row_str + "\\\\")
    draw_line()

    for metrics_instance_index in range(0, metrics_instance_count):
        row_str = method_name_list[metrics_instance_index]
        for class_index in range(0, class_count):
            is_bold = boldness_matrix[metrics_instance_index, class_index] == 1
            row_str = row_str + prepare_class_based_aa_metric_for_cell(metrics_holder_list, metrics_instance_index,
                                                                       class_index, is_bold)
        oa_is_bold = overall_metric_boldness_matrix[metrics_instance_index][MATRIX_OA_INDEX]
        aa_is_bold = overall_metric_boldness_matrix[metrics_instance_index][MATRIX_AA_INDEX]
        kappa_is_bold = overall_metric_boldness_matrix[metrics_instance_index][MATRIX_KAPPA_INDEX]
        row_str = row_str + \
                  prepare_oa_for_cell(metrics_holder_list, metrics_instance_index, oa_is_bold) + \
                  prepare_aa_for_cell(metrics_holder_list, metrics_instance_index, aa_is_bold) + \
                  prepare_kappa_for_cell(metrics_holder_list, metrics_instance_index, kappa_is_bold)
        print(row_str + "\\\\")

    draw_double_line()
    end_tabular(one_column)


def print_column_wise_methods(method_name_list, class_dist_info_list, metrics_holder_list, table_info, one_column):
    class_count, metrics_instance_count = get_lengths_from_metrics(metrics_holder_list)
    boldness_matrix = extract_bold_values(convert_metrics_holder_list_to_matrix(metrics_holder_list))
    overall_metric_boldness_matrix = extract_bold_values(convert_overall_metrics_to_matrix(metrics_holder_list))

    col_count = 1 + len(method_name_list)  # title + method list
    begin_tabular(table_info, col_count, one_column)
    draw_double_line()
    print_combine_column(col_count, PERFORMANCE_STR, CLASSES_STR)

    header_row_str = "\\cline{2-%i} " % col_count
    for metrics_instance_index in range(0, metrics_instance_count):
        header_row_str = header_row_str + "&" + method_name_list[metrics_instance_index] + " "
    print(header_row_str + "\\\\")
    draw_line()

    for class_index in range(0, class_count):
        row_str = class_dist_info_list[class_index] + " "
        for metrics_instance_index in range(0, metrics_instance_count):
            is_bold = boldness_matrix[metrics_instance_index, class_index] == 1
            row_str = row_str + prepare_class_based_aa_metric_for_cell(metrics_holder_list,
                                                                       metrics_instance_index,
                                                                       class_index,
                                                                       is_bold)
        print(row_str + "\\\\")

    draw_line()
    oa_str = "OA "
    for metrics_instance_index in range(0, metrics_instance_count):
        is_bold = overall_metric_boldness_matrix[metrics_instance_index][MATRIX_OA_INDEX]
        oa_str = oa_str + prepare_oa_for_cell(metrics_holder_list, metrics_instance_index, is_bold)
    print(oa_str + "\\\\")

    aa_str = "AA "
    for metrics_instance_index in range(0, metrics_instance_count):
        is_bold = overall_metric_boldness_matrix[metrics_instance_index][MATRIX_AA_INDEX]
        aa_str = aa_str + prepare_aa_for_cell(metrics_holder_list, metrics_instance_index, is_bold)
    print(aa_str + "\\\\")

    kappa_str = "Kappa "
    for metrics_instance_index in range(0, metrics_instance_count):
        is_bold = overall_metric_boldness_matrix[metrics_instance_index][MATRIX_KAPPA_INDEX]
        kappa_str = kappa_str + prepare_kappa_for_cell(metrics_holder_list, metrics_instance_index, is_bold)
    print(kappa_str + "\\\\")

    draw_double_line()
    end_tabular(one_column)


def calculate_kappa_metric_for_instance(metrics_holder_list, metrics_instance_index):
    mean = numpy.mean(metrics_holder_list[metrics_instance_index].kappa_array)
    std = numpy.std(metrics_holder_list[metrics_instance_index].kappa_array)
    return mean, std


def calculate_aa_metric_for_instance(metrics_holder_list, metrics_instance_index):
    mean = numpy.mean(numpy.mean(metrics_holder_list[metrics_instance_index].aa_array, axis=1))
    std = numpy.std(numpy.mean(metrics_holder_list[metrics_instance_index].aa_array, axis=1))
    return mean, std


def calculate_oa_metric_for_instance(metrics_holder_list, metrics_instance_index):
    mean = numpy.mean(metrics_holder_list[metrics_instance_index].oa_array)
    std = numpy.std(metrics_holder_list[metrics_instance_index].oa_array)
    return mean, std


def prepare_kappa_for_cell(metrics_holder_list, metrics_instance_index, is_bold):
    mean, std = calculate_kappa_metric_for_instance(metrics_holder_list, metrics_instance_index)
    return prepare_mean_std_cell(mean * PERCENTILE_COEFF, std * PERCENTILE_COEFF, is_bold)


def prepare_aa_for_cell(metrics_holder_list, metrics_instance_index, is_bold):
    mean, std = calculate_aa_metric_for_instance(metrics_holder_list, metrics_instance_index)
    return prepare_mean_std_cell(mean * PERCENTILE_COEFF, std * PERCENTILE_COEFF, is_bold)


def prepare_oa_for_cell(metrics_holder_list, metrics_instance_index, is_bold):
    mean, std = calculate_oa_metric_for_instance(metrics_holder_list, metrics_instance_index)
    return prepare_mean_std_cell(mean * PERCENTILE_COEFF, std * PERCENTILE_COEFF, is_bold)


def prepare_class_based_aa_metric_for_cell(metrics_holder_list, metrics_instance_index, class_index, is_bold):
    mean_aa, std_aa = \
        calculate_class_based_mean_aa_metric_for_cell(metrics_holder_list, metrics_instance_index, class_index)
    mean_aa = mean_aa * 100
    std_aa = std_aa * 100
    return prepare_mean_std_cell(mean_aa, std_aa, is_bold)


def prepare_mean_std_cell(mean_aa, std_aa, is_bold):
    if is_bold:
        metric_str = ("& \\textbf{" + FLOAT_FORMAT + "}$\\pm{\\textbf{" + FLOAT_FORMAT + "}}$ ") % (mean_aa, std_aa)
    else:
        metric_str = ("& " + FLOAT_FORMAT + "$\\pm{" + FLOAT_FORMAT + "}$ ") % (mean_aa, std_aa)
    return metric_str


def calculate_class_based_mean_aa_metric_for_cell(metrics_holder_list, metrics_instance_index, class_index):
    aa_array = metrics_holder_list[metrics_instance_index].aa_array
    mean_aa = numpy.mean(aa_array, axis=0)[class_index]
    std_aa = numpy.std(aa_array, axis=0)[class_index]
    return mean_aa, std_aa


def draw_line():
    print("\\hline")


def draw_double_line():
    print("\\hline\\hline")


def print_combine_column(col_count, combined_col_title, combined_row_title):
    multi_row_len = 1.0 / col_count
    multi_col_len = 1.0 - multi_row_len
    multi_row = "\\multirow{2}{%.2f\\linewidth}{%s} & " % (
        multi_row_len, combined_row_title)
    multi_col = "\\multicolumn{%i}{>{\\centering\\arraybackslash}p{%.2f\\linewidth}}{%s}\\\\" % (
        col_count - 1, multi_col_len, combined_col_title)
    print(multi_row + multi_col)


def begin_tabular(table_info, col_count, one_column):
    if one_column:
        print("\\begin {table}[ht!]")
    else:
        print("\\begin {table*}")

    print("\\centering")
    print("\\caption {%s} \\label{tab:%s}" % (table_info.title, table_info.label))
    print("\\def\\arraystretch{1}")
    print("\\resizebox{\\linewidth}{!}{%")
    col_len = 1 / col_count
    row_config = ">{}p{%.2f\\linewidth}" % col_len
    for col_index in range(1, col_count):
        row_config = row_config + "| >{\\centering\\arraybackslash}p{%.2f\\linewidth}" % col_len

    print("\\begin{tabu}{%s}" % row_config)


def end_tabular(one_column):
    print("\\end{tabu}")
    print("}")
    if one_column:
        print("\\end {table}")
    else:
        print("\\end {table*}")


if __name__ == '__main__':
    main()
