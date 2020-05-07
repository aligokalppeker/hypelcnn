import glob
import os
import sys
from collections import namedtuple

import numpy

MetricsHolder = namedtuple('MetricsHolder', ['aa_array', 'kappa_array', 'oa_array', 'sample_count'])


def histogram(confusion_matrix, index):
    length_of_elements = confusion_matrix.shape[index]
    result = numpy.zeros(length_of_elements, dtype=int)
    for traverse_index in range(0, length_of_elements):
        total_row = 0
        if index == 0:
            total_row = numpy.sum(confusion_matrix[traverse_index, :])
        elif index == 1:
            total_row = numpy.sum(confusion_matrix[:, traverse_index])
        result[traverse_index] = total_row
    return result


def calc_kappa(conf_mat):
    """
    Calculates the kappa
    kappa calculates the kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    num_ratings = len(conf_mat)

    hist_rater_a = histogram(conf_mat, 0)
    hist_rater_b = histogram(conf_mat, 1)

    num_scored_items = float(numpy.sum(hist_rater_a))

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (float(hist_rater_a[i]) * float(hist_rater_b[j]) / num_scored_items)
            if i == j:
                d = 0.0
            else:
                d = 1.0
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def calc_mean_quadratic_weighted_kappa(kappas, weights=None):
    """
    Calculates the mean of the quadratic
    weighted kappas after applying Fisher's r-to-z transform, which is
    approximately a variance-stabilizing transformation.  This
    transformation is undefined if one of the kappas is 1.0, so all kappa
    values are capped in the range (-0.999, 0.999).  The reverse
    transformation is then applied before returning the result.
    mean_quadratic_weighted_kappa(kappas), where kappas is a vector of
    kappa values
    mean_quadratic_weighted_kappa(kappas, weights), where weights is a vector
    of weights that is the same size as kappas.  Weights are applied in the
    z-space
    """
    kappas = numpy.array(kappas, dtype=float)
    if weights is None:
        weights = numpy.ones(numpy.shape(kappas))
    else:
        weights = weights / numpy.mean(weights)

    # ensure that kappas are in the range [-.999, .999]
    kappas = numpy.array([min(x, .999) for x in kappas])
    kappas = numpy.array([max(x, -.999) for x in kappas])

    z = 0.5 * numpy.log((1 + kappas) / (1 - kappas)) * weights
    z = numpy.mean(z)
    return (numpy.exp(2 * z) - 1) / (numpy.exp(2 * z) + 1)


def extract_accuracy_metrics(confusion_matrix):
    total_samples = numpy.sum(confusion_matrix)
    hit_count = 0
    for x_index in range(0, confusion_matrix.shape[0]):
        for y_index in range(0, confusion_matrix.shape[1]):
            if x_index == y_index:
                hit_count = hit_count + confusion_matrix[x_index][y_index]
    overall_accuracy = hit_count / total_samples

    class_accuracy = numpy.zeros(confusion_matrix.shape[0], dtype=float)
    class_based_samples = numpy.zeros(confusion_matrix.shape[0], dtype=int)
    for x_index in range(0, confusion_matrix.shape[0]):
        total_row = numpy.sum(confusion_matrix[x_index, :])
        class_based_samples[x_index] = total_row
        class_accuracy[x_index] = confusion_matrix[x_index, x_index] / total_row

    kappa = calc_kappa(confusion_matrix)

    return overall_accuracy, class_accuracy, kappa, class_based_samples


def extract_statistics_info(confusion_matrix_list):
    oa_array = None
    aa_array = None
    kappa_array = None
    sample_count = None
    file_count = len(confusion_matrix_list)
    for index, confusion_matrix in enumerate(confusion_matrix_list):
        oa, aa, kappa, class_based_samples = extract_accuracy_metrics(confusion_matrix)
        if oa_array is None and aa_array is None and sample_count is None:
            oa_array = numpy.zeros(file_count, dtype=float)
            aa_array = numpy.zeros([file_count, aa.shape[0]], dtype=float)
            kappa_array = numpy.zeros(file_count, dtype=float)
            sample_count = class_based_samples
        oa_array[index - 1] = oa
        aa_array[index - 1, :] = aa
        kappa_array[index - 1] = kappa
    return MetricsHolder(aa_array=aa_array, kappa_array=kappa_array, oa_array=oa_array, sample_count=sample_count)


def get_conf_list_from_directory(directory):
    confusion_matrix_list = []
    for index, filename in enumerate(glob.glob(os.path.join(directory, "*.csv"))):
        absolute_path = os.path.join(os.path.dirname(__file__), filename)
        confusion_matrix = numpy.loadtxt(absolute_path, dtype=numpy.int, delimiter=",")
        confusion_matrix_list.append(confusion_matrix)
    return confusion_matrix_list


def calculate_mean_std_metrics(oa_array, aa_array, kappa_array):
    mean_oa_array = numpy.mean(oa_array)
    std_oa_array = numpy.std(oa_array)
    mean_aa_array = numpy.mean(numpy.mean(aa_array, axis=1))
    std_aa_array = numpy.std(numpy.mean(aa_array, axis=1))
    mean_kappa_array = numpy.mean(kappa_array)
    std_kappa_array = numpy.std(kappa_array)
    return mean_oa_array, std_oa_array, mean_aa_array, std_aa_array, mean_kappa_array, std_kappa_array


def print_statistics_info(metrics_holder):
    for oa, aa, kappa in zip(metrics_holder.oa_array, metrics_holder.aa_array, metrics_holder.kappa_array):
        print("OA: %.4f AA: %.4f Kappa: %.4f" % (oa, numpy.mean(aa), kappa))
    print("#Metrics statistics:")
    mean_oa_array, std_oa_array, mean_aa_array, std_aa_array, mean_kappa_array, std_kappa_array = \
        calculate_mean_std_metrics(metrics_holder.oa_array, metrics_holder.aa_array, metrics_holder.kappa_array)
    print("OA:    %.4f +- %.4f" % (mean_oa_array, std_oa_array))
    print("AA:    %.4f +- %.4f" % (mean_aa_array, std_aa_array))
    print("Kappa: %.4f +- %.4f" % (mean_kappa_array, std_kappa_array))
    print("#Class based accuracy")
    for aa_mean, aa_std, a_sample_count in zip(numpy.mean(metrics_holder.aa_array, axis=0),
                                               numpy.std(metrics_holder.aa_array, axis=0),
                                               metrics_holder.sample_count):
        print("%.4f +- %.4f %d" % (aa_mean, aa_std, a_sample_count))


def main():
    directory = sys.argv[1]
    print_statistics_info(extract_statistics_info(get_conf_list_from_directory(directory)))


if __name__ == '__main__':
    main()
