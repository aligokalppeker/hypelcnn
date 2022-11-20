import argparse
import json
import os
import time

import numpy
import tensorflow as tf
from tf_slim import get_variables_to_restore
from tifffile import imwrite

from common.cmd_parser import add_parse_cmds_for_loaders, add_parse_cmds_for_loggers, add_parse_cmds_for_trainers, \
    add_parse_cmds_for_models, add_parse_cmds_for_importers
from common.common_nn_ops import simple_nn_iterator, ModelInputParams, NNParams, \
    perform_prediction, create_colored_image, get_model_from_name, get_loader_from_name, create_target_image_via_samples
from importer.GeneratorImporter import GeneratorImporter, GeneratorDataInfo


def add_parse_cmds_for_app(parser):
    parser.add_argument("--domain", nargs="?", type=str, default="all",
                        help="Conversion domain for inferencing. It can be all(all scene inference), "
                             "sample(sample based inference) or gt(ground truth)")


def create_all_scene_data(scene_shape, data_with_labels_to_copy):
    targets = numpy.zeros([scene_shape[0] * scene_shape[1], 3], dtype=int)
    total_index = 0
    for col_index in range(0, scene_shape[0]):
        for row_index in range(0, scene_shape[1]):
            targets[total_index] = [row_index, col_index, 0]
            total_index = total_index + 1

    return GeneratorDataInfo(data=None,
                             targets=targets,
                             loader=data_with_labels_to_copy.loader,
                             dataset=data_with_labels_to_copy.dataset)


def create_sample_data(test_data_with_labels,
                       training_data_with_labels,
                       validation_data_with_labels):
    targets = numpy.vstack([test_data_with_labels.targets.astype(numpy.int32),
                            training_data_with_labels.targets.astype(numpy.int32),
                            validation_data_with_labels.targets.astype(numpy.int32)])
    return GeneratorDataInfo(data=None,
                             targets=targets,
                             loader=test_data_with_labels.loader,
                             dataset=test_data_with_labels.dataset)


def main(_):
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loaders(parser)
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_trainers(parser)
    add_parse_cmds_for_models(parser)
    add_parse_cmds_for_importers(parser)
    add_parse_cmds_for_app(parser)
    flags, unparsed = parser.parse_known_args()

    start_time = time.time()

    if flags.domain == "all" or flags.domain == "sample":
        scene_as_image, color_list = prediction_process(flags)
    elif flags.domain == "gt":
        scene_as_image, color_list = gt_process(flags)
    else:
        raise ValueError(f"Domain flags does not support value:{flags.domain}")

    imwrite(os.path.join(flags.output_path, "result_raw.tif"), scene_as_image)
    imwrite(os.path.join(flags.output_path, "result_colorized.tif"), create_colored_image(scene_as_image, color_list))
    print(f"Done evaluation({time.time() - start_time:.3f} sec)")


def gt_process(flags):
    loader = get_loader_from_name(flags.loader_name, flags.path)
    sample_set = loader.load_samples(0.1, 0)
    data_set = loader.load_data(0, False)
    scene_shape = data_set.get_scene_shape()
    scene_as_image = create_target_image_via_samples(sample_set, scene_shape)
    color_list = loader.get_samples_color_list()
    return scene_as_image, color_list


def prediction_process(flags):
    data_importer = GeneratorImporter()

    training_data_with_labels, test_data_with_labels, validation_data_with_labels, shadow_dict, class_range, \
    scene_shape, color_list = \
        data_importer.read_data_set(flags.loader_name, flags.path, 0.1, 0, flags.neighborhood, True)

    if flags.domain == "all":
        validation_data_with_labels = create_all_scene_data(scene_shape, validation_data_with_labels)
    elif flags.domain == "sample":
        validation_data_with_labels = create_sample_data(training_data_with_labels, test_data_with_labels,
                                                         validation_data_with_labels)

    scene_as_image = numpy.full(shape=scene_shape, dtype=numpy.uint8, fill_value=255)

    if flags.algorithm_param_path is not None:
        algorithm_params = json.load(open(flags.algorithm_param_path, "r"))
    else:
        raise IOError("Algorithm parameter file is not given")

    algorithm_params["batch_size"] = flags.batch_size
    nn_model = get_model_from_name(flags.model_name)

    testing_tensor, training_tensor, validation_tensor = data_importer.convert_data_to_tensor(
        test_data_with_labels,
        training_data_with_labels,
        validation_data_with_labels,
        class_range)
    deep_nn_template = tf.compat.v1.make_template("nn_core", nn_model.create_tensor_graph, class_count=class_range.stop)
    validation_input_iter = simple_nn_iterator(validation_tensor.dataset, flags.batch_size)
    validation_images, validation_labels = validation_input_iter.get_next()
    model_input_params = ModelInputParams(x=validation_images, y=None, device_id="/gpu:0", is_training=False)
    validation_tensor_outputs = deep_nn_template(model_input_params, algorithm_params=algorithm_params)
    validation_nn_params = NNParams(input_iterator=validation_input_iter, data_with_labels=validation_data_with_labels,
                                    metrics=None, predict_tensor=validation_tensor_outputs.y_conv)
    saver = tf.compat.v1.train.Saver(var_list=get_variables_to_restore(include=["nn_core"],
                                                                       exclude=["image_gen_net_"]))
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    with tf.compat.v1.Session(config=config) as session:
        # Restore variables from disk.
        saver.restore(session, flags.base_log_path)

        # Init for imaging the results
        data_importer.init_tensors(session, validation_tensor, validation_nn_params)
        perform_prediction(session, validation_nn_params, scene_as_image)

    return scene_as_image, color_list


if __name__ == "__main__":
    tf.compat.v1.app.run(main=main)
