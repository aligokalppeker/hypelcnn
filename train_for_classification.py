import argparse
import json
import os
import pickle
import time

import tensorflow as tf
from hyperopt import fmin, tpe, Trials, space_eval
from numpy import std, mean

from cmd_parser import add_parse_cmds_for_classification
from common_nn_operations import create_graph, TrainingResult, AugmentationInfo, get_model_from_name, \
    get_importer_from_name
from monitored_session_runner import run_monitored_session, add_classification_summaries, set_run_seed

episode_run_index = 0


def perform_an_episode(flags, algorithm_params, model, base_log_path):
    data_importer = get_importer_from_name(flags.importer_name)

    train_data_w_labels, test_data_w_labels, val_data_w_labels, shadow_dict, class_range, scene_shape, color_list = \
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
    required_steps = flags.step if epoch is None else (train_data_w_labels.data.shape[0] * epoch) // batch_size

    global episode_run_index
    print(f"Starting episode#{episode_run_index:d} with steps#{required_steps:d} : {algorithm_params}")

    validation_accuracy_list = []
    testing_accuracy_list = []
    loss_list = []

    device_id = "/gpu:0"
    if flags.device == "gpu":
        device_id = "/gpu:0"
    elif flags.device == "cpu":
        device_id = "/cpu:0"
    elif flags.device == "tpu":
        device_id = None

    for run_index in range(0, flags.split_count):
        with tf.Graph().as_default():
            set_run_seed()

            print(f"Starting Run #{run_index:d}")

            testing_tensor, training_tensor, validation_tensor = data_importer.convert_data_to_tensor(
                test_data_w_labels, train_data_w_labels, val_data_w_labels, class_range)

            cross_entropy, learning_rate, testing_nn_params, training_nn_params, validation_nn_params, train_step = create_graph(
                training_tensor.dataset, testing_tensor.dataset, validation_tensor.dataset, class_range,
                batch_size, device_id, epoch, augmentation_info=augmentation_info,
                algorithm_params=algorithm_params, model=model,
                create_separate_validation_branch=data_importer.requires_separate_validation_branch)

            #######################################################################
            training_nn_params.data_with_labels = train_data_w_labels
            testing_nn_params.data_with_labels = test_data_w_labels
            validation_nn_params.data_with_labels = val_data_w_labels
            ############################################################################

            if not flags.perform_validation:
                validation_nn_params = None

            add_classification_summaries(cross_entropy, learning_rate, flags.log_model_params, testing_nn_params,
                                         validation_nn_params)

            episode_start_time = time.time()

            log_dir = os.path.join(base_log_path, f"episode_{episode_run_index:d}/run_{run_index:d}")
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


def main(_):
    def convert_trial_to_dict(trial):
        dict_value_results = {}
        for k, v in trial.items():
            dict_value_results[k] = None if len(v) == 0 else v[0]
        return dict_value_results

    parser = argparse.ArgumentParser()
    add_parse_cmds_for_classification(parser)
    flags, unparsed = parser.parse_known_args()
    print("Args:", json.dumps(vars(flags), indent=3))

    nn_model = get_model_from_name(flags.model_name)

    if flags.max_evals == 1:
        print("Running in single execution training mode")
        algorithm_params = nn_model.get_default_params(flags.batch_size)
        if flags.algorithm_param_path is not None:
            algorithm_params = json.load(open(flags.algorithm_param_path, "r"))
        perform_an_episode(flags, algorithm_params, nn_model, flags.base_log_path)
    else:
        print("Running in hyper parameter optimization mode")
        model_space_fun = nn_model.get_hyper_param_space

        global episode_run_index
        trial_file_path = os.path.join(flags.base_log_path, "trial.p")
        best = None
        while True:
            try:
                with open(trial_file_path, "rb") as read_file:
                    trials = pickle.load(read_file)
                episode_run_index = len(trials.trials)
                best = convert_trial_to_dict(trials.best_trial["misc"]["vals"])
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
            pickle.dump(trials, open(trial_file_path, "wb"))

        json.dump(trials.results, open("trial_results.json", "w"), indent=3)
        if best is not None:
            print(space_eval(model_space_fun(), best))


if __name__ == '__main__':
    tf.app.run(main=main)
