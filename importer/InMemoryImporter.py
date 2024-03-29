import time
from collections import namedtuple

import numpy
import tensorflow as tf

from importer.DataImporter import DataImporter
from common.common_nn_ops import get_loader_from_name

Target = namedtuple('Target', ['data', 'labels'])
InMemoryDataTensor = namedtuple('InMemoryDataTensor', ['dataset', 'x', 'y_'])


class InMemoryImporter(DataImporter):

    @staticmethod
    def _input_nn(data_type, data_shape, label_type, label_shape, class_range, prefix):
        x = tf.compat.v1.placeholder(dtype=data_type,
                                     shape=data_shape,
                                     name=f"{prefix}_x")
        y_ = tf.compat.v1.placeholder(dtype=label_type,
                                      shape=label_shape,
                                      name=f"{prefix}_y_")
        return x, y_, tf.one_hot(y_, class_range.stop, dtype=tf.uint8, name=f"{prefix}_one_hot")

    @staticmethod
    def _get_data_with_labels(targets, loader, data_set):
        data_as_matrix = numpy.zeros(numpy.concatenate([[targets.shape[0]], data_set.get_data_shape()]),
                                     dtype=numpy.float32)
        label_as_matrix = numpy.zeros(targets.shape[0], dtype=numpy.uint8)

        index = 0
        for point in targets:
            data_as_matrix[index] = data_set.get_data_point(point[0], point[1])
            label_as_matrix[index] = point[2]
            index = index + 1

        return Target(data=data_as_matrix, labels=label_as_matrix)

    def read_data_set(self, loader_name, path, train_data_ratio, test_data_ratio, neighborhood, normalize):
        start_time = time.time()

        loader = get_loader_from_name(loader_name, path)

        data_set = loader.load_data(neighborhood, normalize)
        sample_set = loader.load_samples(train_data_ratio, test_data_ratio)

        training_data_with_labels = self._get_data_with_labels(sample_set.training_targets, loader, data_set)
        validation_data_with_labels = self._get_data_with_labels(sample_set.validation_targets, loader, data_set)
        test_data_with_labels = self._get_data_with_labels(sample_set.test_targets, loader, data_set)

        print(f"Loaded dataset({time.time() - start_time:.3f} sec)")
        return training_data_with_labels, test_data_with_labels, validation_data_with_labels, data_set.shadow_creator_dict, \
               loader.get_class_count(), data_set.get_scene_shape(), loader.get_samples_color_list()

    def convert_data_to_tensor(self, test_data_with_labels, training_data_with_labels, validation_data_with_labels,
                               class_range):
        # modify first element as none
        training_x, training_y_, training_y_one_hot = \
            self._input_nn(training_data_with_labels.data.dtype,
                           (tuple([None]) + training_data_with_labels.data.shape[1:]),
                           training_data_with_labels.labels.dtype,
                           (tuple([None]) + training_data_with_labels.labels.shape[1:]),
                           class_range,
                           "training")
        training_data_set = tf.data.Dataset.from_tensor_slices((training_x, training_y_one_hot))
        ###################
        # modify first element as none
        testing_x, testing_y_, testing_y_one_hot = \
            self._input_nn(test_data_with_labels.data.dtype, (tuple([None]) + test_data_with_labels.data.shape[1:]),
                           test_data_with_labels.labels.dtype, (tuple([None]) + test_data_with_labels.labels.shape[1:]),
                           class_range,
                           "testing")
        testing_data_set = tf.data.Dataset.from_tensor_slices((testing_x, testing_y_one_hot))
        ##################################
        return InMemoryDataTensor(dataset=testing_data_set, x=testing_x, y_=testing_y_), \
               InMemoryDataTensor(dataset=training_data_set, x=training_x, y_=training_y_), \
               InMemoryDataTensor(dataset=testing_data_set, x=testing_x, y_=testing_y_)

    def init_tensors(self, session, tensor, nn_params):
        session.run(nn_params.input_iterator.initializer,
                    feed_dict={tensor.x: nn_params.data_with_labels.data,
                               tensor.y_: nn_params.data_with_labels.labels})

    def requires_separate_validation_branch(self):
        return True
