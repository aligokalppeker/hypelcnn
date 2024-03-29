import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from tf_slim import get_variables_to_restore

from importer.DataImporter import DataImporter
from importer.GeneratorImporter import GeneratorDataTensor, GeneratorImporter
from common.cmd_parser import add_parse_cmds_for_loggers, add_parse_cmds_for_loaders
from common.common_nn_ops import simple_nn_iterator, ModelInputParams, NNParams, get_model_from_name


class ControlledDataImporter(DataImporter):

    def __init__(self):
        self.target_class = []
        self.target_data = []
        self.generator_importer = GeneratorImporter()

    def read_data_set(self, loader_name, path, train_data_ratio, test_data_ratio, neighborhood, normalize):
        training_data_with_labels, test_data_with_labels, validation_data_with_labels, shadow_dict, class_range, \
        scene_shape, color_list = \
            self.generator_importer.read_data_set(loader_name=loader_name, path=path, train_data_ratio=train_data_ratio,
                                                  test_data_ratio=test_data_ratio,
                                                  neighborhood=neighborhood, normalize=normalize)
        shape = training_data_with_labels.dataset.get_data_shape()
        for index in range(0, 5000):
            self.target_class.append(0)
            element_data = numpy.zeros(shape=shape, dtype=numpy.float)
            element_data[:, :, -1] = numpy.ones(element_data[:, :, -1].shape)
            self.target_data.append(element_data)

        return training_data_with_labels, test_data_with_labels, validation_data_with_labels, shadow_dict, class_range, \
               scene_shape, color_list

    def convert_data_to_tensor(self, test_data_with_labels, training_data_with_labels, validation_data_with_labels,
                               class_range):
        tensor_type_info = (tf.float32, tf.uint8)
        training_data_set = tf.data.Dataset.from_generator(
            lambda: self._iterator_function(), tensor_type_info,
            (tf.TensorShape(training_data_with_labels.dataset.get_data_shape()),
             tf.TensorShape([])))
        class_count = class_range.stop
        training_data_set = training_data_set.map(
            lambda image, label: GeneratorImporter.extract_fn(image, label, class_count, 'training'),
            num_parallel_calls=8)

        testing_data_set = tf.data.Dataset.from_generator(
            lambda: self._iterator_function(), tensor_type_info,
            (tf.TensorShape(test_data_with_labels.dataset.get_data_shape()),
             tf.TensorShape([])))
        testing_data_set = testing_data_set.map(
            lambda image, label: GeneratorImporter.extract_fn(image, label, class_count, 'testing'),
            num_parallel_calls=8)

        validation_data_set = tf.data.Dataset.from_generator(
            lambda: self._iterator_function(), tensor_type_info,
            (tf.TensorShape(validation_data_with_labels.dataset.get_data_shape()),
             tf.TensorShape([])))
        validation_data_set = validation_data_set.map(
            lambda image, label: GeneratorImporter.extract_fn(image, label, class_count, 'validation'),
            num_parallel_calls=8)

        return GeneratorDataTensor(dataset=testing_data_set), \
               GeneratorDataTensor(dataset=training_data_set), \
               GeneratorDataTensor(dataset=validation_data_set)

    def init_tensors(self, session, tensor, nn_params):
        session.run(nn_params.input_iterator.initializer)

    def requires_separate_validation_branch(self):
        return self.generator_importer.requires_separate_validation_branch()

    def _iterator_function(self):
        for data, tar_class in zip(self.target_data, self.target_class):
            yield data, tar_class


def calculate_tensor_size(tensor):
    size = 1
    for shape_dim in tensor.shape.dims:
        if shape_dim.value is not None:
            size = size * shape_dim.value
    return size


def main(_):
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_loaders(parser)
    flags, unparsed = parser.parse_known_args()

    nn_model = get_model_from_name(flags.model_name)()
    if flags.algorithm_param_path is not None:
        algorithm_params = json.load(open(flags.algorithm_param_path, 'r'))
    else:
        raise IOError("Algorithm parameter file is not given")
    algorithm_params["batch_size"] = flags.batch_size

    data_importer = ControlledDataImporter()

    training_data_with_labels, test_data_with_labels, validation_data_with_labels, shadow_dict, class_range, \
    scene_shape, color_list = \
        data_importer.read_data_set(flags.loader_name, flags.path, flags.test_ratio, flags.neighborhood, True)

    validation_data_with_labels = data_importer.create_all_scene_data(scene_shape,
                                                                      validation_data_with_labels)
    testing_tensor, training_tensor, validation_tensor = data_importer.convert_data_to_tensor(
        test_data_with_labels,
        training_data_with_labels,
        validation_data_with_labels,
        class_range)

    deep_nn_template = tf.compat.v1.make_template('nn_core', nn_model.create_tensor_graph, class_count=class_range.stop)

    start_time = time.time()

    validation_data_set = validation_tensor.dataset

    validation_input_iter = simple_nn_iterator(validation_data_set, flags.batch_size)
    validation_images, validation_labels = validation_input_iter.get_next()
    model_input_params = ModelInputParams(x=validation_images, y=None, device_id='/gpu:0', is_training=False)
    validation_tensor_outputs = deep_nn_template(model_input_params, algorithm_params=algorithm_params)
    validation_nn_params = NNParams(input_iterator=validation_input_iter, data_with_labels=None,
                                    metrics=None, predict_tensor=validation_tensor_outputs.y_conv)
    validation_nn_params.data_with_labels = validation_data_with_labels

    saver = tf.compat.v1.train.Saver(var_list=get_variables_to_restore(include=["nn_core"],
                                                                       exclude=["image_gen_net_"]))
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    with tf.compat.v1.Session(config=config) as session:
        # Restore variables from disk.
        saver.restore(session, flags.base_log_path)

        # Init for imaging the results
        validation_tensor.importer.init_tensors(session, validation_tensor, validation_nn_params)

        histogram_tensors = []
        histogram_tensor_names = []
        result_map = {}
        bin_map = {}
        base_bin = 480 / calculate_tensor_size(validation_tensor_outputs.histogram_tensors[0].tensor)
        for histogram_tensor_instance in validation_tensor_outputs.histogram_tensors:
            histogram_tensors.append(histogram_tensor_instance.tensor)
            histogram_tensor_names.append(histogram_tensor_instance.name)
            result_map[histogram_tensor_instance.name] = []
            bin_map[histogram_tensor_instance.name] = int(
                base_bin * calculate_tensor_size(histogram_tensor_instance.tensor))

        sample_idx = 0
        while True:
            try:
                # prediction, current_prediction = session.run([
                #     tf.argmax(validation_nn_params.predict_tensor, 1), histogram_tensors])

                current_prediction = session.run(histogram_tensors)
                if sample_idx > 2000:
                    for tensor_result, tensor_name in zip(current_prediction, histogram_tensor_names):
                        result_map[tensor_name].append(tensor_result)
                if sample_idx == 5000:
                    break

                sample_idx = sample_idx + 1

            except tf.errors.OutOfRangeError:
                break

        for tensor_name in histogram_tensor_names:
            range_min = sys.float_info.max
            range_max = sys.float_info.min
            for result in result_map[tensor_name]:
                range_min = min(range_min, result.min())
                range_max = max(range_max, result.max())

            all_hists = numpy.zeros([len(result_map[tensor_name]), bin_map[tensor_name]], dtype=numpy.int)
            bin_edges = None
            for idx, result in enumerate(result_map[tensor_name]):
                hist, bin_edges = numpy.histogram(result, range=(range_min, range_max), bins=bin_map[tensor_name])
                all_hists[idx] = hist

            mean = numpy.mean(all_hists, axis=0)
            plt.plot(bin_edges[:-1], mean, label=tensor_name)
            plt.xlim(min(bin_edges), max(bin_edges))
            plt.xlabel("Value")
            plt.ylabel("Counts")
            plt.fill_between(bin_edges[:-1], mean)
            plt.title(tensor_name)
            plt.savefig(os.path.join(flags.output_path, tensor_name + "_fig.jpg"))
            plt.clf()

    print('Done evaluation(%.3f sec)' % (time.time() - start_time))


if __name__ == '__main__':
    tf.compat.v1.app.run(main=main)
