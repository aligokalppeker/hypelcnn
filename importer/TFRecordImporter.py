from collections import namedtuple

import numpy as np
import tensorflow as tf

from importer.DataImporter import DataImporter
from common.common_nn_ops import get_loader_from_name

TFRecordDataInfo = namedtuple('TFRecordDataInfo', ['data', 'path'])
TFRecordDataTensor = namedtuple('InMemoryDataTensor', ['dataset', 'path_placeholder'])
TFRecordSpecialData = namedtuple('TFRecordSpecialData', ['shape'])


class TFRecordImporter(DataImporter):

    def read_data_set(self, loader_name, path, train_data_ratio, test_data_ratio, neighborhood, normalize):
        loader = get_loader_from_name(loader_name, path)

        model_base_dir = loader.get_model_base_dir()
        training_data_shape = None
        testing_data_shape = None
        validation_data_shape = None
        for record in tf.compat.v1.python_io.tf_record_iterator(model_base_dir + "metadata.tfrecord"):
            example = tf.train.Example()
            example.ParseFromString(record)  # calling protocol buffer API

            training_data_shape = np.array(example.features.feature["training_data_shape"].int64_list.value)
            testing_data_shape = np.array(example.features.feature["testing_data_shape"].int64_list.value)
            validation_data_shape = np.array(example.features.feature["validation_data_shape"].int64_list.value)

        return TFRecordDataInfo(data=TFRecordSpecialData(training_data_shape),
                                path=model_base_dir + "training.tfrecord"), \
               TFRecordDataInfo(data=TFRecordSpecialData(testing_data_shape),
                                path=model_base_dir + "test.tfrecord"), \
               TFRecordDataInfo(data=TFRecordSpecialData(validation_data_shape),
                                path=model_base_dir + "validation.tfrecord"), None, \
               loader.get_class_count(), None, loader.get_samples_color_list()

    @staticmethod
    def extract_fn(data_record, shape, class_count, prefix):
        features = {
            # Extract features using the keys set during creation
            "label": tf.io.FixedLenFeature([], tf.int64),
            "image": tf.io.FixedLenFeature(shape=[np.prod(shape)], dtype=tf.float32),
        }
        sample = tf.io.parse_single_example(serialized=data_record, features=features)
        image = tf.reshape(sample["image"], shape)
        label_one_hot = tf.one_hot(sample["label"], class_count, dtype=tf.uint8, name=f"{prefix}_one_hot")

        return image, label_one_hot

    def convert_data_to_tensor(self, test_data_with_labels, training_data_with_labels, validation_data_with_labels,
                               class_range):
        training_path_placeholder = tf.compat.v1.placeholder(tf.string, 1, "training_path_placeholder")
        class_count = class_range.stop
        training_data_set = tf.data.TFRecordDataset(training_path_placeholder).map(
            lambda inp: self.extract_fn(inp, training_data_with_labels.data.shape[1:4], class_count, "training"))

        testing_path_placeholder = tf.compat.v1.placeholder(tf.string, 1, "testing_path_placeholder")
        testing_data_set = tf.data.TFRecordDataset(testing_path_placeholder).map(
            lambda inp: self.extract_fn(inp, test_data_with_labels.data.shape[1:4], class_count, "testing"))

        return TFRecordDataTensor(dataset=testing_data_set, path_placeholder=testing_path_placeholder), \
               TFRecordDataTensor(dataset=training_data_set, path_placeholder=training_path_placeholder), \
               TFRecordDataTensor(dataset=testing_data_set, path_placeholder=testing_path_placeholder)

    def init_tensors(self, session, tensor, nn_params):
        session.run(nn_params.input_iterator.initializer,
                    feed_dict={tensor.path_placeholder: [nn_params.data_with_labels.path]})

    def requires_separate_validation_branch(self):
        return False
