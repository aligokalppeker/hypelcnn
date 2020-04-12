import time
from collections import namedtuple

import numpy
import tensorflow as tf

from DataImporter import DataImporter
from common_nn_operations import get_class

GeneratorDataTensor = namedtuple('GeneratorDataTensor', ['dataset', 'importer'])

GeneratorDataInfo = namedtuple('GeneratorDataInfo', ['data', 'targets', 'loader', 'dataset'])
GeneratorSpecialData = namedtuple('GeneratorSpecialData', ['shape', 'size'])


class GeneratorImporter(DataImporter):

    @staticmethod
    def _iterator_function(targets, loader, data_set):
        for point in targets:
            yield (loader.get_point_value(data_set, point), point[2])

    def read_data_set(self, loader_name, path, test_data_ratio, neighborhood, normalize):
        start_time = time.time()

        loader = get_class(loader_name + '.' + loader_name)(path)

        data_set = loader.load_data(neighborhood, normalize)
        sample_set = loader.load_samples(test_data_ratio)

        training_data_shape = numpy.concatenate(
            ([sample_set.training_targets.shape[0]], loader.get_data_shape(data_set)))
        testing_data_shape = numpy.concatenate(
            ([sample_set.test_targets.shape[0]], loader.get_data_shape(data_set)))
        validation_data_shape = numpy.concatenate(
            ([sample_set.validation_targets.shape[0]], loader.get_data_shape(data_set)))

        print('Loaded dataset(%.3f sec)' % (time.time() - start_time))
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
            loader.get_class_count(), loader.get_scene_shape(data_set), loader.get_target_color_list()

    @staticmethod
    def extract_fn(image, label, class_count, prefix):
        with tf.device('/cpu:0'):
            label_one_hot = tf.one_hot(label, class_count, dtype=tf.uint8, name=prefix + '_one_hot')
        return image, label_one_hot

    def convert_data_to_tensor(self, test_data_with_labels, training_data_with_labels, validation_data_with_labels,
                               class_range):
        tensor_type_info = (tf.float32, tf.uint8)
        training_data_set = tf.data.Dataset.from_generator(
            lambda: self._iterator_function(training_data_with_labels.targets, training_data_with_labels.loader,
                                            training_data_with_labels.dataset), tensor_type_info,
            (tf.TensorShape(training_data_with_labels.loader.get_data_shape(training_data_with_labels.dataset)),
             tf.TensorShape([])))
        class_count = class_range.stop
        training_data_set = training_data_set.map(
            lambda image, label: self.extract_fn(image, label, class_count, 'training'), num_parallel_calls=8)

        testing_data_set = tf.data.Dataset.from_generator(
            lambda: self._iterator_function(test_data_with_labels.targets, test_data_with_labels.loader,
                                            test_data_with_labels.dataset), tensor_type_info,
            (tf.TensorShape(test_data_with_labels.loader.get_data_shape(test_data_with_labels.dataset)),
             tf.TensorShape([])))
        testing_data_set = testing_data_set.map(
            lambda image, label: self.extract_fn(image, label, class_count, 'testing'), num_parallel_calls=8)

        validation_data_set = tf.data.Dataset.from_generator(
            lambda: self._iterator_function(validation_data_with_labels.targets, validation_data_with_labels.loader,
                                            validation_data_with_labels.dataset), tensor_type_info,
            (tf.TensorShape(validation_data_with_labels.loader.get_data_shape(validation_data_with_labels.dataset)),
             tf.TensorShape([])))
        validation_data_set = validation_data_set.map(
            lambda image, label: self.extract_fn(image, label, class_count, 'validation'), num_parallel_calls=8)

        return GeneratorDataTensor(dataset=testing_data_set, importer=self), \
               GeneratorDataTensor(dataset=training_data_set, importer=self), \
               GeneratorDataTensor(dataset=validation_data_set, importer=self)

    def perform_tensor_initialize(self, session, tensor, nn_params):
        session.run(nn_params.input_iterator.initializer)

    def requires_separate_validation_branch(self):
        return True

    def create_all_scene_data(self, scene_shape, data_with_labels_to_copy):
        targets = self.create_all_scene_target_array(scene_shape)
        return GeneratorDataInfo(data=None,
                                 targets=targets,
                                 loader=data_with_labels_to_copy.loader,
                                 dataset=data_with_labels_to_copy.dataset)

    @staticmethod
    def create_all_scene_target_array(scene_shape):
        targets = numpy.zeros([scene_shape[0] * scene_shape[1], 3], dtype=int)
        total_index = 0
        for col_index in range(0, scene_shape[0]):
            for row_index in range(0, scene_shape[1]):
                targets[total_index] = [row_index, col_index, 0]
                total_index = total_index + 1
        return targets
