import argparse
import os

import tensorflow as tf

from cmd_parser import add_parse_cmds_for_loader, add_parse_cmds_for_loggers, type_ensure_strtobool
from common_nn_operations import get_importer_from_name


def main(_):
    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loader(parser)
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_apps(parser)
    flags, unparsed = parser.parse_known_args()

    inmemory_importer = get_importer_from_name("InMemoryImporter")
    training_data_with_labels, test_data_with_labels, validation_data_with_labels, shadow_ratio, class_count, scene_shape, color_list = \
        inmemory_importer.read_data_set(flags.loader_name, flags.path, flags.train_ratio, 0.05, flags.neighborhood,
                                        True)

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


def add_parse_cmds_for_apps(parser):
    parser.add_argument('--compressed', nargs='?', const=True, type=type_ensure_strtobool, default=False,
                        help='If true, performs compression')


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
