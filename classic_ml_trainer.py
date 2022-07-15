import argparse
import os
import time
from math import sqrt

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from tifffile import imsave
from tqdm import tqdm

from importer import InMemoryImporter, GeneratorImporter
from common_nn_operations import create_colored_image, get_loader_from_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', const=True, type=str,
                        default='C:/Users/AliGökalp/Documents/phd/data/2013_DFTC/2013_DFTC',
                        help='Input data path')
    parser.add_argument('--loader_name', nargs='?', const=True, type=str,
                        default='GRSS2013DataLoader',
                        help='Data set loader name, Values : GRSS2013DataLoader')
    parser.add_argument('--neighborhood', nargs='?', type=int,
                        default=5,
                        help='Neighborhood for data extraction')
    parser.add_argument('--hyperparamopt', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, performs hyper parameter optimization.')
    parser.add_argument('--fullscene', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, performs full scene classification.')
    parser.add_argument('--batch_size', nargs='?', type=int,
                        default=20,
                        help='Batch size')
    parser.add_argument('--split_count', nargs='?', type=int,
                        default=1,
                        help='Split count')
    parser.add_argument('--base_log_path', nargs='?', const=True, type=str,
                        default=os.path.dirname(__file__),
                        help='Base path for saving logs')

    flags, unparsed = parser.parse_known_args()

    loader_name = flags.loader_name
    data_path = flags.path
    neighborhood = flags.neighborhood

    for run_index in range(flags.split_count):
        print('Starting episode#%d' % run_index)

        data_importer = InMemoryImporter.InMemoryImporter()
        training_data_with_labels, test_data_with_labels, validation_data_with_labels, shadow_dict, class_range, scene_shape, color_list = \
            data_importer.read_data_set(loader_name=loader_name, path=data_path,
                                        test_data_ratio=0,
                                        train_data_ratio=0.1,
                                        neighborhood=neighborhood, normalize=False)

        flattened_training_data = flatten_data(training_data_with_labels.data)
        flattened_validation_data = flatten_data(validation_data_with_labels.data)

        start_time = time.time()
        estimator = RandomForestClassifier(n_estimators=50, n_jobs=8, max_features=int(2 * sqrt(144)), verbose=False)
        # estimator = ExtraTreesClassifier(n_estimators=10000, n_jobs=8, verbose=1)
        # estimator = SVC(kernel='poly', degree=1, cache_size=200, verbose=True)  # GRSS2013
        # estimator = SVC(kernel='rbf', gamma=1e-09, C=10000, cache_size=200) # GRSS2013
        # estimator = SVC(kernel='rbf', gamma=1e-06, C=1000000, cache_size=1000, verbose=True)  # GULFPORT

        estimator.fit(flattened_training_data, training_data_with_labels.labels)
        print('Completed training(%.3f sec)' % (time.time() - start_time))
        predicted_validation_data = estimator.predict(flattened_validation_data)

        overall_accuracy = accuracy_score(validation_data_with_labels.labels, predicted_validation_data)
        average_accuracy = balanced_accuracy_score(validation_data_with_labels.labels, predicted_validation_data)
        kappa = cohen_kappa_score(validation_data_with_labels.labels, predicted_validation_data)
        conf_matrix = confusion_matrix(validation_data_with_labels.labels, predicted_validation_data)
        print_output(estimator.get_params(), average_accuracy, conf_matrix, kappa, overall_accuracy, run_index,
                     loader_name,
                     flags.base_log_path)

        if flags.hyperparamopt:
            perform_hyperparamopt(flattened_training_data, training_data_with_labels)

        if flags.fullscene:
            perform_full_scene_classification(data_path, loader_name, neighborhood, estimator, flags.batch_size)


def perform_full_scene_classification(data_path, loader_name, neighborhood, estimator, batch_size):
    loader = get_loader_from_name(loader_name, data_path)
    data_set = loader.load_data(neighborhood, False)
    scene_shape = data_set.get_scene_shape()
    all_scene_target_array = GeneratorImporter.GeneratorImporter.create_all_scene_target_array(scene_shape)
    predict_pixel_count = scene_shape[0] * scene_shape[1]
    progress_bar = tqdm(total=predict_pixel_count)

    prediction = numpy.empty([predict_pixel_count], dtype=numpy.uint8)
    batch_cache = numpy.empty([batch_size,
                               data_set.get_data_shape()[0],
                               data_set.get_data_shape()[1],
                               data_set.get_data_shape()[2]], dtype=numpy.float32)
    current_pixel_index = 0
    batch_pixel_index = 0
    for point in all_scene_target_array:
        batch_cache[batch_pixel_index] = loader.get_point_value(data_set, point)
        current_pixel_index = current_pixel_index + 1
        batch_pixel_index = batch_pixel_index + 1
        if current_pixel_index == predict_pixel_count or batch_pixel_index == batch_size:
            point_val = flatten_data(batch_cache[0:batch_pixel_index])
            prediction[(current_pixel_index - batch_pixel_index):current_pixel_index] = \
                estimator.predict(point_val)
            progress_bar.update(batch_pixel_index)
            batch_pixel_index = 0

    progress_bar.close()
    scene_as_image = numpy.reshape(prediction, scene_shape)
    output_path = "."
    imsave(os.path.join(output_path, "result_raw.tif"),
           scene_as_image)
    imsave(os.path.join(output_path, "result_colorized.tif"),
           create_colored_image(scene_as_image, loader.get_target_color_list()))


def flatten_data(data):
    return numpy.reshape(data, [data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]])


def flatten_single_data(data):
    return numpy.reshape(data, [1, data.shape[0] * data.shape[1] * data.shape[2]])


def perform_hyperparamopt(flattened_training_data, training_data_with_labels):
    # c_range = numpy.logspace(-6, 16, 23)
    # gamma_range = numpy.logspace(-11, 5, 17)
    c_range = numpy.logspace(-2, 10, 13)
    gamma_range = numpy.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=c_range)
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=16, verbose=1)
    grid.fit(flattened_training_data, training_data_with_labels.labels)
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))


def print_output(algorithm_params, average_accuracy, conf_matrix, kappa, overall_accuracy, index, name, base_log_path):
    print("OA:%5.5f" % overall_accuracy)
    print("AA:%5.5f" % average_accuracy)
    print("KAPPA:%5.5f" % kappa)
    print("Confusion Matrix:")
    print(conf_matrix)
    file_id = f"{name}_run{index}"

    log_path = os.path.join(base_log_path, f"confusion_matrix_{file_id}.csv")
    numpy.savetxt(log_path, conf_matrix, fmt="%d", delimiter=",")

    metrics_file = open(os.path.join(base_log_path, f"metrics_{file_id}.txt"), "w")
    print("OA,AA,KAPPA", file=metrics_file)
    print("%.6f,%.6f,%.6f" % (overall_accuracy, average_accuracy, kappa), file=metrics_file)
    metrics_file.close()

    params_file = open(os.path.join(base_log_path, f"params_{file_id}.json"), "w")
    print(algorithm_params, file=params_file)
    params_file.close()


if __name__ == '__main__':
    main()
