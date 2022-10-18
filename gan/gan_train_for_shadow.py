import argparse
import json
import random
import string
from collections.abc import Sequence
from statistics import mean
from types import SimpleNamespace

import optuna
import tensorflow as tf
from optuna.samplers import TPESampler
from tensorflow.python import data
from tensorflow.python.data.experimental import shuffle_and_repeat
from tensorflow.python.platform import gfile
from tensorflow.python.training.basic_session_run_hooks import StopAtStepHook, LoggingTensorHook
from tensorflow.python.training.device_setter import replica_device_setter
from tensorflow.python.training.monitored_session import USE_DEFAULT
from tensorflow.python.training.training_util import get_or_create_global_step
from tensorflow_gan.python.train import get_sequential_train_hooks

from common.cmd_parser import add_parse_cmds_for_loaders, add_parse_cmds_for_loggers, add_parse_cmds_for_trainers, \
    type_ensure_strtobool, add_parse_cmds_for_json_loader
from common.common_nn_ops import set_all_gpu_config, get_loader_from_name, TextSummaryAtStartHook
from common.common_ops import replace_abbrs
from gan.wrapper_registry import get_wrapper_dict, get_infer_wrapper_dict
from gan.wrappers.gan_common import InitializerHook
from gan_sampling_methods import TargetBasedSampler, RandomBasedSampler, DummySampler, NeighborhoodBasedSampler


def add_parse_cmds_for_app(parser):
    parser.add_argument("--gan_type", nargs="?", type=str, default="cycle_gan",
                        help="Gan type to train, possible values; cycle_gan, gan_x2y and gan_y2x")

    parser.add_argument("--use_identity_loss", nargs="?", type=type_ensure_strtobool, default=True,
                        help="Whether to use identity loss during training.")
    parser.add_argument("--identity_loss_weight", nargs="?", type=float, default=0.5,
                        help="The weight of cycle consistency loss.")

    parser.add_argument("--regularization_support_rate", nargs="?", type=float, default=0.0,
                        help="The regularization support rate, ranges from 0 to 1, 1 means full support.")

    parser.add_argument("--cycle_consistency_loss_weight", nargs="?", type=float, default=10.0,
                        help="The weight of cycle consistency loss.")
    parser.add_argument("--nce_loss_weight", nargs="?", type=float, default=10.0,
                        help="The weight of NCE loss.")
    parser.add_argument("--tau", nargs="?", type=float, default=0.07,
                        help="Tau value for the NCE loss.")
    parser.add_argument("--patches", nargs="?", type=int, default=6,
                        help="Patch count for feature discriminator. (for the CUT and DCL GANs)")
    parser.add_argument("--embedded_feat_size", nargs="?", type=int, default=2,
                        help="Embedded feature size for feature discriminator. (for the CUT and DCL GANs)")

    parser.add_argument("--validation_steps", nargs="?", type=int, default=1000,
                        help="Validation frequency")
    parser.add_argument("--validation_sample_count", nargs="?", type=int, default=300,
                        help="Validation sample count")

    parser.add_argument("--generator_lr", nargs="?", type=float, default=0.0002,
                        help="The compression model learning rate.")
    parser.add_argument("--discriminator_lr", nargs="?", type=float, default=0.0001,
                        help="The discriminator learning rate.")
    parser.add_argument("--gen_discriminator_lr", nargs="?", type=float, default=0.0001,
                        help="The generator discriminator learning rate.")
    parser.add_argument("--discriminator_reg_scale", nargs="?", type=float, default=0.00001,
                        help="The discriminator regularization scale.")
    parser.add_argument("--gen_disc_reg_scale", nargs="?", type=float, default=0.0001,
                        help="The generator discriminator regularization scale.")
    parser.add_argument("--pairing_method", nargs="?", type=str, default="random",
                        help="Pairing method for the shadowed and non-shadowed samples. "
                             "Opts: random, target, dummy, neighbour")

    parser.add_argument("--master", nargs="?", type=str, default="",
                        help="Name of the TensorFlow master to use.")
    parser.add_argument("--ps_tasks", nargs="?", type=int, default=0,
                        help="The number of parameter servers. If the value is 0, "
                             "then the parameters are handled locally by the worker.")
    parser.add_argument("--task", nargs="?", type=int, default=0,
                        help="The Task ID. This value is used when training with multiple workers to "
                             "identify each worker.")

    parser.add_argument("--flag_config_file_opt", nargs="?", type=str,
                        default=None,
                        help="Flag config file for hyper parameter optimization")
    parser.add_argument("--opt_trial_count", nargs="?", type=int,
                        default=10,
                        help="Trial count for the optimization part.")
    parser.add_argument("--opt_run_count", nargs="?", type=int,
                        default=1,
                        help="Retry count for each trial during the optimization.")


def gan_train(train_ops,
              logdir,
              get_hooks_fn=get_sequential_train_hooks(),
              master="",
              is_chief=True,
              scaffold=None,
              hooks=None,
              chief_only_hooks=None,
              save_checkpoint_secs=USE_DEFAULT,
              save_summaries_steps=USE_DEFAULT,
              save_checkpoint_steps=USE_DEFAULT,
              max_wait_secs=7200,
              config=None):
    """A wrapper around `training.train` that uses GAN hooks.

    Args:
      save_checkpoint_steps: Checkpoint steps to
      train_ops: A GANTrainOps named tuple.
      logdir: The directory where the graph and checkpoints are saved.
      get_hooks_fn: A function that takes a GANTrainOps tuple and returns a list
        of hooks.
      master: The URL of the master.
      is_chief: Specifies whether the training is being run by the primary
        replica during replica training.
      scaffold: An tf.train.Scaffold instance.
      hooks: List of `tf.train.SessionRunHook` callbacks which are run inside the
        training loop.
      chief_only_hooks: List of `tf.train.SessionRunHook` instances which are run
        inside the training loop for the chief trainer only.
      save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
        using a default checkpoint saver. If `save_checkpoint_secs` is set to
        `None`, then the default checkpoint saver isn't used.
      save_summaries_steps: The frequency, in number of global steps, that the
        summaries are written to disk using a default summary saver. If
        `save_summaries_steps` is set to `None`, then the default summary saver
        isn't used.
      max_wait_secs: Maximum time that workers should wait for the session to
        become available. This should be kept relatively short to help detect
        incorrect code, but sometimes may need to be increased if the chief takes
        a while to start up.
      config: An instance of `tf.ConfigProto`.

    Returns:
      Output of the call to `training.train`.
    """
    new_hooks = get_hooks_fn(train_ops)
    hooks = new_hooks if hooks is None else list(hooks) + list(new_hooks)

    with tf.compat.v1.train.MonitoredTrainingSession(
            master=master,
            is_chief=is_chief,
            checkpoint_dir=logdir,
            scaffold=scaffold,
            hooks=hooks,
            chief_only_hooks=chief_only_hooks,
            save_checkpoint_secs=save_checkpoint_secs,
            save_summaries_steps=save_summaries_steps,
            save_checkpoint_steps=save_checkpoint_steps,
            config=config,
            max_wait_secs=max_wait_secs) as session:
        gstep = None
        while not session.should_stop():
            gstep = session.run(train_ops.global_step_inc_op)
    tf.compat.v1.reset_default_graph()
    return gstep


def load_op(batch_size, iteration_count, loader, data_set, shadow_map, shadow_ratio, reg_support_rate, pairing_method):
    sampling_method_map = {"target": TargetBasedSampler(margin=5),
                           "random": RandomBasedSampler(multiply_shadowed_data=False),
                           "neighbour": NeighborhoodBasedSampler(neighborhood_size=20, margin=2),
                           "dummy": DummySampler(element_count=2000, fill_value=0.5, coefficient=2)}
    if pairing_method in sampling_method_map:
        normal_data_as_matrix, shadow_data_as_matrix = sampling_method_map[pairing_method].get_sample_pairs(
            data_set,
            loader,
            shadow_map)
    else:
        raise ValueError(f"Wrong sampling parameter value ({pairing_method}).")

    normal_data_as_matrix = normal_data_as_matrix[:, :, :, 0:data_set.get_casi_band_count()]
    shadow_data_as_matrix = shadow_data_as_matrix[:, :, :, 0:data_set.get_casi_band_count()]

    normal_data_holder = tf.compat.v1.placeholder(dtype=normal_data_as_matrix.dtype, shape=normal_data_as_matrix.shape,
                                                  name="x")
    shadow_data_holder = tf.compat.v1.placeholder(dtype=shadow_data_as_matrix.dtype, shape=shadow_data_as_matrix.shape,
                                                  name="y")

    epoch = (iteration_count * batch_size) // normal_data_as_matrix.shape[0]
    data_set = data.Dataset.from_tensor_slices((normal_data_holder, shadow_data_holder)).apply(
        shuffle_and_repeat(buffer_size=10000, count=epoch))
    data_set = data_set.map(
        lambda param_x, param_y_: perform_shadow_augmentation_random(param_x, param_y_, shadow_ratio, reg_support_rate),
        num_parallel_calls=4)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set_itr = tf.compat.v1.data.make_initializable_iterator(data_set)

    return InitializerHook(data_set_itr,
                           normal_data_holder, shadow_data_holder,
                           normal_data_as_matrix, shadow_data_as_matrix)


def perform_shadow_augmentation_random(normal_images, shadow_images, shadow_ratio, reg_support_rate):
    with tf.compat.v1.name_scope("shadow_ratio_augmenter"):
        with tf.device("/cpu:0"):
            normal_images_rand = tf.cond(pred=tf.less(tf.random.uniform([1], 0.01, 0.99)[0], reg_support_rate),
                                         false_fn=lambda: normal_images,
                                         true_fn=lambda: (shadow_images * shadow_ratio))

            shadow_images_rand = tf.cond(pred=tf.less(tf.random.uniform([1], 0.01, 0.99)[0], reg_support_rate),
                                         false_fn=lambda: shadow_images,
                                         true_fn=lambda: (normal_images_rand / shadow_ratio))

    return normal_images_rand, shadow_images_rand


def get_log_suffix(flags):
    abbreviations = {"dataloader": "ldr"
                     }

    patch_size = (flags.neighborhood * 2) + 1
    suffix = f"{flags.loader_name.lower():s}_{flags.gan_type.lower():s}_" \
             f"{patch_size:d}x{patch_size:d}_" \
             f"regsup{flags.regularization_support_rate:.2f}_" \
             f"batch{flags.batch_size:d}".replace(".", "")
    if flags.use_identity_loss is True:
        suffix = suffix + f"_idnty{flags.use_identity_loss:.2f}".replace(".", "")

    return replace_abbrs(suffix, abbreviations)


def main(_):
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loaders(parser)
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_trainers(parser)
    add_parse_cmds_for_json_loader(parser)
    add_parse_cmds_for_app(parser)
    flags, unparsed = parser.parse_known_args()

    if flags.flag_config_file:
        flags = update_flags_from_json(flags, flags.flag_config_file)

    if flags.flag_config_file_opt:
        flags_from_json_opt = json.load(open(flags.flag_config_file_opt, "r"))

        def objective(trial):
            flags_as_dict = dict(vars(flags))
            for key, value in flags_from_json_opt.items():
                if type(value) is dict:
                    if "min" in value and "max" in value:
                        min_range_val = value["min"]
                        max_range_val = value["max"]
                        if type(min_range_val) is float and type(max_range_val) is float:
                            flags_as_dict[key] = trial.suggest_float(key, min_range_val, max_range_val,
                                                                     step=value["step"] if "step" in value else None,
                                                                     log=value["log"] if "log" in value else False)
                        elif type(min_range_val) is int and type(max_range_val) is int:
                            flags_as_dict[key] = trial.suggest_int(key, min_range_val, max_range_val,
                                                                   step=value["step"] if "step" in value else 1)
                        else:
                            print(f"Parameter value is put in hyper optimization "
                                  f"config but its min max type is inconsistent: {key}. "
                                  f"Using the default value")
                elif type(value) is list:
                    flags_as_dict[key] = trial.suggest_categorical(key, value)
                else:
                    flags_as_dict[key] = value

            losses = []
            for run_idx in range(0, flags.opt_run_count):
                trial_postfix = f"_{''.join(random.choices(string.ascii_lowercase + string.digits, k=5))}"
                flags_as_dict["base_log_path"] = flags_as_dict["base_log_path"] + trial_postfix
                print(f"Starting run#{run_idx}")
                losses.append(mean(run_session(SimpleNamespace(**flags_as_dict))))

            print("Trial runs are completed. Losses:")
            print(*losses, sep=",")
            return max(losses)

        print("Running on hyper parameter mode")
        study_name = "gan_shadow_opt"
        study = optuna.create_study(study_name=study_name, direction="minimize",
                                    sampler=TPESampler(), storage=f"sqlite:///{study_name}.db",
                                    load_if_exists=True)
        study.optimize(objective, n_trials=flags.opt_trial_count)
    else:
        print("Running on training mode")
        print("Output divergence values:", run_session(flags))


def run_session(flags):
    log_dir = f"{flags.base_log_path}_{get_log_suffix(flags)}"
    if not gfile.Exists(log_dir):
        gfile.MakeDirs(log_dir)

    gan_train_wrapper_dict = get_wrapper_dict(flags)
    gan_inference_wrapper_dict = get_infer_wrapper_dict()

    validation_iteration_count = flags.validation_steps
    validation_sample_count = flags.validation_sample_count
    neighborhood = 0
    checkpoint_count = flags.step // validation_iteration_count

    loader = get_loader_from_name(flags.loader_name, flags.path)
    data_set = loader.load_data(neighborhood, True)
    shadow_map, shadow_ratio = loader.load_shadow_map(neighborhood, data_set)

    with tf.device(replica_device_setter(flags.ps_tasks)):
        with tf.compat.v1.name_scope("inputs"):
            initializer_hook = load_op(flags.batch_size, flags.step, loader, data_set,
                                       shadow_map, shadow_ratio,
                                       flags.regularization_support_rate,
                                       flags.pairing_method)
            images_x, images_y = initializer_hook.input_itr.get_next()

        # Define model.
        wrapper = gan_train_wrapper_dict[flags.gan_type]

        the_gan_model = wrapper.define_model(images_x, images_y)
        the_gan_loss = wrapper.define_loss(the_gan_model)

        peer_validation_hook = gan_inference_wrapper_dict[flags.gan_type].create_inference_hook(
            data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
            validation_iteration_count, validation_sample_count)

        # Define GAN train ops.
        train_ops = wrapper.define_train_ops(the_gan_model, the_gan_loss, max_number_of_steps=flags.step,
                                             generator_lr=flags.generator_lr, discriminator_lr=flags.discriminator_lr,
                                             gen_discriminator_lr=flags.gen_discriminator_lr)

        # Training
        status_message = tf.strings.join(["Starting train step: ", tf.as_string(get_or_create_global_step())],
                                         name="status_message")

        set_all_gpu_config()

        train_hooks_fn = wrapper.get_train_hooks_fn()
        gan_train(
            train_ops,
            log_dir,
            scaffold=tf.compat.v1.train.Scaffold(saver=tf.compat.v1.train.Saver(max_to_keep=checkpoint_count)),
            save_checkpoint_steps=validation_iteration_count,
            get_hooks_fn=train_hooks_fn,
            hooks=[
                initializer_hook,
                peer_validation_hook,
                StopAtStepHook(num_steps=flags.step),
                LoggingTensorHook([status_message], every_n_iter=1000),
                TextSummaryAtStartHook(log_dir, "flags", json.dumps(vars(flags), indent=3))
            ],
            master=flags.master,
            is_chief=flags.task == 0)
    # Use the return value for hyperparameter optimization
    best_upper_div = peer_validation_hook.get_best_upper_div()
    best_mean_div = peer_validation_hook.get_best_mean_div()
    return [max(best_upper_div) if isinstance(best_upper_div, Sequence) else best_upper_div,
            max(best_mean_div) if isinstance(best_mean_div, Sequence) else best_mean_div]


def update_flags_from_json(flags, flag_config_file):
    print("Updating flags from json file,", flag_config_file)
    flags_from_json = json.load(open(flag_config_file, "r"))
    flags_as_dict = vars(flags)
    flags_as_dict.update(flags_from_json)
    flags = SimpleNamespace(**flags_as_dict)
    return flags


if __name__ == "__main__":
    tf.compat.v1.app.run()
