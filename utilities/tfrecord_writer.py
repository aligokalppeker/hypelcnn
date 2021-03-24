import argparse
import os

import tensorflow as tf

from common_nn_operations import get_class


def main(_):
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
    parser.add_argument('--test_ratio', nargs='?', type=float,
                        default=0.05,
                        help='Ratio of training data to use in testing')
    parser.add_argument('--compressed', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, performs compression')
    parser.add_argument('--target_path', nargs='?', const=True, type=str,
                        default=os.path.dirname(__file__),
                        help='Target path to write the files to')
    flags, unparsed = parser.parse_known_args()

    inmemoryimporter = get_class('InMemoryImporter.InMemoryImporter')()
    training_data_with_labels, test_data_with_labels, validation_data_with_labels, shadow_ratio, class_count, scene_shape, color_list = \
        inmemoryimporter.read_data_set(flags.loader_name, flags.path, flags.train_ratio, flags.test_ratio,
                                       flags.neighborhood, True)

    write_metadata_record(os.path.join(flags.target_path, 'metadata.tfrecord'),
                          training_data_with_labels.data, test_data_with_labels.data,
                          validation_data_with_labels.data)

    write_to_tfrecord(os.path.join(flags.target_path, 'training.tfrecord'),
                      training_data_with_labels.data, training_data_with_labels.labels,
                      flags.compressed)
    write_to_tfrecord(os.path.join(flags.target_path, 'test.tfrecord'),
                      test_data_with_labels.data, test_data_with_labels.labels,
                      flags.compressed)
    write_to_tfrecord(os.path.join(flags.target_path, 'validation.tfrecord'),
                      validation_data_with_labels.data, validation_data_with_labels.labels,
                      flags.compressed)
    pass


def write_to_tfrecord(train_filename, data, labels, compressed):
    if compressed:
        writer = tf.python_io.TFRecordWriter(train_filename,
                                             options=tf.python_io.TFRecordOptions(
                                                 tf.python_io.TFRecordCompressionType.GZIP))
    else:
        writer = tf.python_io.TFRecordWriter(train_filename)

    data_len = len(data)
    for i in range(data_len):

        if not i % 1000:
            print('Data: {}/{}'.format(i, data_len))

        # Create a feature
        feature = {'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
                   'image': tf.train.Feature(float_list=tf.train.FloatList(value=data[i].reshape(-1)))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


def write_metadata_record(metadata_filename, training_data, testing_data, validation_data):
    writer = tf.python_io.TFRecordWriter(metadata_filename)
    # Create a feature
    feature = {'training_data_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=training_data.shape)),
               'testing_data_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=testing_data.shape)),
               'validation_data_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=validation_data.shape))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())


if __name__ == '__main__':
    tf.app.run(main=main)
