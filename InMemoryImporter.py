import time
from collections import namedtuple

import numpy
import tensorflow as tf

from DataImporter import DataImporter
from common_nn_operations import get_class

Target = namedtuple('Target', ['data', 'labels'])
InMemoryDataTensor = namedtuple('InMemoryDataTensor', ['dataset', 'importer', 'x', 'y_'])


class InMemoryImporter(DataImporter):

    @staticmethod
    def _input_nn(data_type, data_shape, label_type, label_shape, class_range, prefix):
        x = tf.placeholder(dtype=data_type,
                           shape=data_shape,
                           name=prefix + '_x')
        y_ = tf.placeholder(dtype=label_type,
                            shape=label_shape,
                            name=prefix + '_y_')
        return x, y_, tf.one_hot(y_, class_range.stop, dtype=tf.uint8, name=prefix + '_one_hot')

    @staticmethod
    def _get_data_with_labels(targets, loader, data_set):
        data_as_matrix = numpy.zeros(numpy.concatenate([[targets.shape[0]], loader.get_data_shape(data_set)]),
                                     dtype=numpy.float32)
        label_as_matrix = numpy.zeros(targets.shape[0], dtype=numpy.uint8)

        index = 0
        for point in targets:
            data_as_matrix[index] = loader.get_point_value(data_set, point)
            label_as_matrix[index] = point[2]
            index = index + 1

        return Target(data=data_as_matrix, labels=label_as_matrix)

    def read_data_set(self, loader_name, path, train_data_ratio, test_data_ratio, neighborhood, normalize):
        start_time = time.time()

        loader = get_class(loader_name + '.' + loader_name)(path)

        data_set = loader.load_data(neighborhood, normalize)
        sample_set = loader.load_samples(train_data_ratio, test_data_ratio)

        training_data_with_labels = self._get_data_with_labels(sample_set.training_targets, loader, data_set)
        validation_data_with_labels = self._get_data_with_labels(sample_set.validation_targets, loader, data_set)
        test_data_with_labels = self._get_data_with_labels(sample_set.test_targets, loader, data_set)

        print('Loaded dataset(%.3f sec)' % (time.time() - start_time))
        return training_data_with_labels, test_data_with_labels, validation_data_with_labels, data_set.shadow_creator_dict, \
               loader.get_class_count(), loader.get_scene_shape(data_set), loader.get_target_color_list()

    def convert_data_to_tensor(self, test_data_with_labels, training_data_with_labels, validation_data_with_labels,
                               class_range):
        # modify first element as none
        training_x, training_y_, training_y_one_hot = \
            self._input_nn(training_data_with_labels.data.dtype,
                           (tuple([None]) + training_data_with_labels.data.shape[1:]),
                           training_data_with_labels.labels.dtype,
                           (tuple([None]) + training_data_with_labels.labels.shape[1:]),
                           class_range,
                           'training')
        training_data_set = tf.data.Dataset.from_tensor_slices((training_x, training_y_one_hot))
        ###################
        # modify first element as none
        testing_x, testing_y_, testing_y_one_hot = \
            self._input_nn(test_data_with_labels.data.dtype, (tuple([None]) + test_data_with_labels.data.shape[1:]),
                           test_data_with_labels.labels.dtype, (tuple([None]) + test_data_with_labels.labels.shape[1:]),
                           class_range,
                           'testing')
        testing_data_set = tf.data.Dataset.from_tensor_slices((testing_x, testing_y_one_hot))
        ##################################
        return InMemoryDataTensor(dataset=testing_data_set, importer=self, x=testing_x, y_=testing_y_), \
               InMemoryDataTensor(dataset=training_data_set, importer=self, x=training_x, y_=training_y_), \
               InMemoryDataTensor(dataset=testing_data_set, importer=self, x=testing_x, y_=testing_y_)

    def perform_tensor_initialize(self, session, tensor, nn_params):
        session.run(nn_params.input_iterator.initializer,
                    feed_dict={tensor.x: nn_params.data_with_labels.data,
                               tensor.y_: nn_params.data_with_labels.labels})

    def requires_separate_validation_branch(self):
        return True

    def create_all_scene_data(self, scene_shape, data_with_labels_to_copy):
        raise NotImplementedError
