import time
from collections import namedtuple

import numpy
import tensorflow as tf

from common.common_nn_ops import get_loader_from_name
from importer.DataImporter import DataImporter

GeneratorDataTensor = namedtuple('GeneratorDataTensor', ['dataset'])

GeneratorDataInfo = namedtuple('GeneratorDataInfo', ['data', 'targets', 'loader', 'dataset'])
GeneratorSpecialData = namedtuple('GeneratorSpecialData', ['shape', 'size'])


class GeneratorImporter(DataImporter):

    @staticmethod
    def _iterator_function(targets, loader, data_set):
        for point in targets:
            yield loader.get_point_value(data_set, point), point[2]

    def read_data_set(self, loader_name, path, train_data_ratio, test_data_ratio, neighborhood, normalize):
        start_time = time.time()

        loader = get_loader_from_name(loader_name, path)

        data_set = loader.load_data(neighborhood, normalize)
        sample_set = loader.load_samples(train_data_ratio, test_data_ratio)

        training_data_shape = numpy.concatenate(
            ([sample_set.training_targets.shape[0]], data_set.get_data_shape()))
        testing_data_shape = numpy.concatenate(
            ([sample_set.test_targets.shape[0]], data_set.get_data_shape()))
        validation_data_shape = numpy.concatenate(
            ([sample_set.validation_targets.shape[0]], data_set.get_data_shape()))

        print(f"Loaded dataset({time.time() - start_time:.3f} sec)")
        return \
            GeneratorDataInfo(
                data=GeneratorSpecialData(shape=training_data_shape, size=numpy.prod(training_data_shape)),
                targets=sample_set.training_targets,
                loader=loader,
                dataset=data_set), \
            GeneratorDataInfo(
                data=GeneratorSpecialData(shape=testing_data_shape, size=numpy.prod(testing_data_shape)),
                targets=sample_set.test_targets,
                loader=loader,
                dataset=data_set), \
            GeneratorDataInfo(
                data=GeneratorSpecialData(shape=validation_data_shape, size=numpy.prod(validation_data_shape)),
                targets=sample_set.validation_targets,
                loader=loader,
                dataset=data_set), \
            data_set.shadow_creator_dict, \
            loader.get_class_count(), data_set.get_scene_shape(), loader.get_samples_color_list()

    @staticmethod
    def extract_fn(image, label, class_count, prefix):
        with tf.device("/cpu:0"):
            label_one_hot = tf.one_hot(label, class_count, dtype=tf.uint8, name=f"{prefix}_one_hot")
        return image, label_one_hot

    def convert_data_to_tensor(self, test_data_with_labels, training_data_with_labels, validation_data_with_labels,
                               class_range):
        tensor_type_info = (tf.float32, tf.uint8)
        training_data_set = tf.data.Dataset.from_generator(
            lambda: self._iterator_function(training_data_with_labels.targets, training_data_with_labels.loader,
                                            training_data_with_labels.dataset), tensor_type_info,
            (tf.TensorShape(training_data_with_labels.dataset.get_data_shape()),
             tf.TensorShape([])))
        class_count = class_range.stop
        training_data_set = training_data_set.map(
            lambda image, label: self.extract_fn(image, label, class_count, "training"), num_parallel_calls=8)

        testing_data_set = tf.data.Dataset.from_generator(
            lambda: self._iterator_function(test_data_with_labels.targets, test_data_with_labels.loader,
                                            test_data_with_labels.dataset), tensor_type_info,
            (tf.TensorShape(test_data_with_labels.dataset.get_data_shape()),
             tf.TensorShape([])))
        testing_data_set = testing_data_set.map(
            lambda image, label: self.extract_fn(image, label, class_count, "testing"), num_parallel_calls=8)

        validation_data_set = tf.data.Dataset.from_generator(
            lambda: self._iterator_function(validation_data_with_labels.targets, validation_data_with_labels.loader,
                                            validation_data_with_labels.dataset), tensor_type_info,
            (tf.TensorShape(validation_data_with_labels.dataset.get_data_shape()),
             tf.TensorShape([])))
        validation_data_set = validation_data_set.map(
            lambda image, label: self.extract_fn(image, label, class_count, "validation"), num_parallel_calls=8)

        return GeneratorDataTensor(dataset=testing_data_set), \
               GeneratorDataTensor(dataset=training_data_set), \
               GeneratorDataTensor(dataset=validation_data_set)

    def init_tensors(self, session, tensor, nn_params):
        session.run(nn_params.input_iterator.initializer)

    def requires_separate_validation_branch(self):
        return True
