import argparse
import json
import os
import time

import numpy
import tensorflow as tf
from tensorflow.contrib import slim
from tifffile import imsave

from cmd_parser import parse_cmd
from common_nn_operations import get_class, simple_nn_iterator, ModelInputParams, NNParams, \
    perform_prediction, create_colored_image


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', nargs='?', const=True, type=str,
                        default=os.path.dirname(__file__),
                        help='Path for saving output images')
    flags = parse_cmd(parser)

    model = get_class(flags.model_name + '.' + flags.model_name)()
    algorithm_params = model.get_default_params(flags.batch_size)
    if flags.algorithm_param_path is not None:
        algorithm_params = json.load(open(flags.algorithm_param_path, 'r'))

    importer_name = flags.importer_name
    data_importer = get_class(importer_name + '.' + importer_name)()

    training_data_with_labels, test_data_with_labels, validation_data_with_labels, shadow_dict, class_range, \
    scene_shape, color_list = \
        data_importer.read_data_set(flags.loader_name, flags.path, flags.train_ratio, flags.test_ratio,
                                    flags.neighborhood, True)

    validation_data_with_labels = data_importer.create_all_scene_data(scene_shape,
                                                                      validation_data_with_labels)
    testing_tensor, training_tensor, validation_tensor = data_importer.convert_data_to_tensor(
        test_data_with_labels,
        training_data_with_labels,
        validation_data_with_labels,
        class_range)

    deep_nn_template = tf.make_template('nn_core', model.create_tensor_graph, class_count=class_range.stop)

    start_time = time.time()

    validation_data_set = validation_tensor.dataset

    validation_input_iter = simple_nn_iterator(validation_data_set, flags.batch_size)
    validation_images, validation_labels = validation_input_iter.get_next()
    model_input_params = ModelInputParams(x=validation_images, y=None, device_id='/gpu:0', is_training=False)
    validation_tensor_outputs = deep_nn_template(model_input_params, algorithm_params=algorithm_params)
    validation_nn_params = NNParams(input_iterator=validation_input_iter, data_with_labels=None,
                                    metrics=None, predict_tensor=validation_tensor_outputs.y_conv)
    validation_nn_params.data_with_labels = validation_data_with_labels

    prediction = numpy.empty([scene_shape[0] * scene_shape[1]], dtype=numpy.uint8)

    saver = tf.train.Saver(var_list=slim.get_variables_to_restore(include=["nn_core"],
                                                                  exclude=["image_gen_net_"]))
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    with tf.Session(config=config) as session:
        # Restore variables from disk.
        saver.restore(session, flags.base_log_path)

        # Init for imaging the results
        validation_tensor.importer.perform_tensor_initialize(session, validation_tensor, validation_nn_params)
        perform_prediction(session, validation_nn_params, prediction)
        scene_as_image = numpy.reshape(prediction, scene_shape)

        imsave(os.path.join(flags.output_path, "result_raw.tif"),
               scene_as_image)

        imsave(os.path.join(flags.output_path, "result_colorized.tif"),
               create_colored_image(scene_as_image, color_list))

        # Init for accuracy
        # validation_tensor.importer.perform_tensor_initialize(session, validation_tensor, validation_nn_params)
        # validation_accuracy = calculate_accuracy(session, validation_nn_params)
        # print('Validation accuracy=%g' % validation_accuracy)

    print('Done evaluation(%.3f sec)' % (time.time() - start_time))


if __name__ == '__main__':
    tf.app.run(main=main)
