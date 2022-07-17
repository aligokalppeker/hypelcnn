import argparse
import gc
import json
import os
import pickle
import time

import tensorflow as tf
from hyperopt import fmin, tpe, Trials, space_eval
from numpy import std, mean
from tensorflow_core.contrib.framework.python.ops.variables import get_model_variables

from cmd_parser import add_parse_cmds_for_classification
from common_nn_operations import create_graph, TrainingResult, AugmentationInfo, get_model_from_name, \
    get_importer_from_name
from monitored_session_runner import run_monitored_session

episode_run_index = 0


def perform_an_episode(flags, algorithm_params, model, base_log_path):
    data_importer = get_importer_from_name(flags.importer_name)

    training_data_with_labels, test_data_with_labels, validation_data_with_labels, shadow_dict, class_range, \
    scene_shape, color_list = \
        data_importer.read_data_set(flags.loader_name, flags.path, flags.train_ratio, flags.test_ratio,
                                    flags.neighborhood, True)

    shadow_struct = None
    if flags.augment_data_with_shadow is not None and shadow_dict is not None:
        shadow_struct = shadow_dict[flags.augment_data_with_shadow]

    augmentation_info = AugmentationInfo(shadow_struct=shadow_struct,
                                         perform_shadow_augmentation=flags.augment_data_with_shadow is not None,
                                         perform_rotation_augmentation=flags.augment_data_with_rotation,
                                         perform_reflection_augmentation=flags.augment_data_with_reflection,
                                         offline_or_online=flags.offline_augmentation,
                                         augmentation_random_threshold=flags.augmentation_random_threshold)

    batch_size = algorithm_params["batch_size"]
    epoch = flags.epoch
    required_steps = flags.step if epoch is None else int(
        ((training_data_with_labels.data.shape[0] * epoch) / batch_size))

    global episode_run_index
    print("Starting episode#%d with steps#%d : %s" % (episode_run_index, required_steps, algorithm_params))

    validation_accuracy_list = []
    testing_accuracy_list = []
    loss_list = []

    device_id = '/gpu:0'
    if flags.device == "gpu":
        device_id = '/gpu:0'
    elif flags.device == "cpu":
        device_id = '/cpu:0'
    elif flags.device == "tpu":
        device_id = None

    for run_index in range(0, flags.split_count):
        with tf.Graph().as_default():
            # Set random seed as the same value to get consistent results
            tf.set_random_seed(1234)

            print(f"Starting Run #{run_index:d}")

            testing_tensor, training_tensor, validation_tensor = data_importer.convert_data_to_tensor(
                test_data_with_labels,
                training_data_with_labels,
                validation_data_with_labels,
                class_range)

            # Collect unnecessary data
            gc.collect()

            cross_entropy, learning_rate, testing_nn_params, training_nn_params, validation_nn_params, train_step = create_graph(
                training_tensor.dataset, testing_tensor.dataset, validation_tensor.dataset, class_range,
                batch_size, device_id, epoch, augmentation_info=augmentation_info,
                algorithm_params=algorithm_params, model=model,
                create_separate_validation_branch=data_importer.requires_separate_validation_branch)

            # Collect unnecessary data
            gc.collect()

            #######################################################################
            training_nn_params.data_with_labels = training_data_with_labels
            testing_nn_params.data_with_labels = test_data_with_labels
            validation_nn_params.data_with_labels = validation_data_with_labels
            ############################################################################

            if not flags.perform_validation:
                validation_nn_params = None

            tf.summary.scalar("training_cross_entropy", cross_entropy)
            tf.summary.scalar("training_learning_rate", learning_rate)

            tf.summary.text("test_confusion", tf.as_string(testing_nn_params.metrics.confusion))
            tf.summary.scalar("test_overall_accuracy", testing_nn_params.metrics.accuracy)

            tf.summary.text("validation_confusion", tf.as_string(validation_nn_params.metrics.confusion))
            tf.summary.scalar("validation_overall_accuracy", validation_nn_params.metrics.accuracy)

            tf.summary.scalar("validation_average_accuracy", validation_nn_params.metrics.mean_per_class_accuracy)
            tf.summary.scalar("validation_kappa", validation_nn_params.metrics.kappa)

            if flags.log_model_params:
                for variable in get_model_variables():
                    tf.summary.histogram(variable.op.name, variable)

            episode_start_time = time.time()

            log_dir = os.path.join(base_log_path, "episode_" + str(episode_run_index) + "/run_" + str(run_index))
            training_result = run_monitored_session(cross_entropy, log_dir, class_range,
                                                    flags.save_checkpoint_steps, flags.validation_steps,
                                                    train_step, required_steps,
                                                    augmentation_info, flags.device,
                                                    training_nn_params, training_tensor,
                                                    testing_nn_params, testing_tensor,
                                                    validation_nn_params, validation_tensor,
                                                    json.dumps(vars(flags), indent=3),
                                                    json.dumps(algorithm_params, indent=3))

            print(f"Done training for {time.time() - episode_start_time:.3f} sec")

            testing_accuracy_list.append(training_result.test_accuracy)
            loss_list.append(training_result.loss)
            if flags.perform_validation:
                print(
                    f"Run #{run_index:d}, Validation accuracy={training_result.validation_accuracy:g}, "
                    f"Testing accuracy={training_result.test_accuracy:g}, loss={training_result.loss:.2f}")
                validation_accuracy_list.append(training_result.validation_accuracy)
            else:
                print(
                    f"Run #{run_index:d}, Testing accuracy={training_result.test_accuracy:g}, "
                    f"loss={training_result.loss:.2f}")

    mean_validation_accuracy = None
    if flags.perform_validation:
        mean_validation_accuracy = mean(validation_accuracy_list)
        std_validation_accuracy = std(validation_accuracy_list)
        print(
            f"Validation result: ({mean_validation_accuracy[0]:g}) +- ({std_validation_accuracy[0]:g})")
    mean_testing_accuracy = mean(testing_accuracy_list)
    std_testing_accuracy = std(testing_accuracy_list)
    mean_loss = mean(loss_list)
    std_loss = std(loss_list)
    print(
        f"Mean testing accuracy result: ({mean_testing_accuracy[0]:g}) +- ({std_testing_accuracy[0]:g}), "
        f"Loss result: ({mean_loss[0]:g}) +- ({std_loss[0]:g})")

    episode_run_index = episode_run_index + 1

    return TrainingResult(validation_accuracy=mean_validation_accuracy, test_accuracy=mean_testing_accuracy,
                          loss=mean_loss)


def convert_trial_to_dictvalues(trial):
    dict_value_results = {}
    for k, v in trial.items():
        if len(v) == 0:
            dict_value_results[k] = None
        else:
            dict_value_results[k] = v[0]
    return dict_value_results


def main(_):
    parser = argparse.ArgumentParser()
    add_parse_cmds_for_classification(parser)
    flags, unparsed = parser.parse_known_args()
    print("Args:", json.dumps(vars(flags), indent=3))

    nn_model = get_model_from_name(flags.model_name)

    if flags.max_evals == 1:
        print("Running in single execution training mode")

        algorithm_params = nn_model.get_default_params(flags.batch_size)
        if flags.algorithm_param_path is not None:
            algorithm_params = json.load(open(flags.algorithm_param_path, 'r'))
            # algorithm_params = namedtuple('GenericDict', algorithm_params_dict.keys())(**algorithm_params_dict)
        perform_an_episode(flags, algorithm_params, nn_model, flags.base_log_path)
        # code for dumping the parameters as json
        # json.dump(algorithm_params, open('algorithm_param_output_cnnv4.json', 'w'), indent=3)
    else:
        print("Running in hyper parameter optimization mode")
        model_space_fun = nn_model.get_hyper_param_space

        global episode_run_index
        trial_fileaddress = os.path.join(flags.base_log_path, "trial.p")
        while True:
            try:
                with open(trial_fileaddress, "rb") as read_file:
                    trials = pickle.load(read_file)
                episode_run_index = len(trials.trials)
                best = convert_trial_to_dictvalues(trials.best_trial["misc"]["vals"])
            except IOError:
                print("No trials file found. Starting trials from scratch")
                episode_run_index = 0
                trials = Trials()

            if episode_run_index == flags.max_evals:
                break

            best = fmin(
                fn=lambda params: 1 - (
                    perform_an_episode(flags, params, nn_model, flags.base_log_path).validation_accuracy),
                space=model_space_fun(),
                algo=tpe.suggest,
                trials=trials,
                max_evals=episode_run_index + 1)
            pickle.dump(trials, open(trial_fileaddress, "wb"))

        json.dump(trials.results, open("trial_results.json", "w"), indent=3)
        print(space_eval(model_space_fun(), best))


if __name__ == '__main__':
    tf.app.run(main=main)
