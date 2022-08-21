import argparse
import json

import tensorflow as tf
from tensorflow.python.training.monitored_session import USE_DEFAULT
from tensorflow.python import data
from tensorflow.python.data.experimental import shuffle_and_repeat
from tensorflow.python.platform import gfile
from tensorflow.python.training.basic_session_run_hooks import StopAtStepHook, LoggingTensorHook
from tensorflow.python.training.device_setter import replica_device_setter
from tensorflow.python.training.training_util import get_or_create_global_step
from tensorflow_gan.python.train import get_sequential_train_hooks

from common.cmd_parser import add_parse_cmds_for_loaders, add_parse_cmds_for_loggers, add_parse_cmds_for_trainers, \
    type_ensure_strtobool
from common.common_nn_ops import set_all_gpu_config, get_loader_from_name, TextSummaryAtStartHook
from common.common_ops import replace_abbrs
from gan.wrappers.cut_wrapper import CUTWrapper
from gan.wrappers.cycle_gan_wrapper import CycleGANWrapper
from gan.wrappers.gan_common import InitializerHook, model_base_name
from gan_sampling_methods import TargetBasedSampler, RandomBasedSampler, DummySampler, NeighborhoodBasedSampler
from gan.wrappers.gan_wrapper import GANWrapper


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
      max_wait_secs: Maximum time workers should wait for the session to
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
    data_set = data_set.batch(batch_size)
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
    add_parse_cmds_for_app(parser)
    flags, unparsed = parser.parse_known_args()

    log_dir = f"{flags.base_log_path}_{get_log_suffix(flags)}"
    if not gfile.Exists(log_dir):
        gfile.MakeDirs(log_dir)

    with tf.device(replica_device_setter(flags.ps_tasks)):
        validation_iteration_count = flags.validation_steps
        validation_sample_count = flags.validation_sample_count
        neighborhood = 0
        loader = get_loader_from_name(flags.loader_name, flags.path)
        data_set = loader.load_data(neighborhood, True)

        shadow_map, shadow_ratio = loader.load_shadow_map(neighborhood, data_set)

        with tf.compat.v1.name_scope("inputs"):
            initializer_hook = load_op(flags.batch_size, flags.step, loader, data_set,
                                       shadow_map, shadow_ratio,
                                       flags.regularization_support_rate,
                                       flags.pairing_method)
            training_input_iter = initializer_hook.input_itr
            images_x, images_y = training_input_iter.get_next()

        # Define model.
        gan_type = flags.gan_type
        gan_train_wrapper_dict = {
            "cycle_gan": CycleGANWrapper(cycle_consistency_loss_weight=flags.cycle_consistency_loss_weight,
                                         identity_loss_weight=flags.identity_loss_weight,
                                         use_identity_loss=flags.use_identity_loss),
            "gan_x2y": GANWrapper(identity_loss_weight=flags.identity_loss_weight,
                                  use_identity_loss=flags.use_identity_loss,
                                  swap_inputs=False),
            "gan_y2x": GANWrapper(identity_loss_weight=flags.identity_loss_weight,
                                  use_identity_loss=flags.use_identity_loss,
                                  swap_inputs=True),
            "cut_x2y": CUTWrapper(nce_loss_weight=flags.nce_loss_weight,
                                  identity_loss_weight=flags.identity_loss_weight,
                                  use_identity_loss=flags.use_identity_loss,
                                  swap_inputs=False),
            "cut_y2x": CUTWrapper(nce_loss_weight=flags.nce_loss_weight,
                                  identity_loss_weight=flags.identity_loss_weight,
                                  use_identity_loss=flags.use_identity_loss,
                                  swap_inputs=True)}
        wrapper = gan_train_wrapper_dict[gan_type]

        with tf.compat.v1.variable_scope(model_base_name, reuse=tf.compat.v1.AUTO_REUSE):
            the_gan_model = wrapper.define_model(images_x, images_y)
            peer_validation_hook = wrapper.create_validation_hook(data_set, loader, log_dir, neighborhood,
                                                                  shadow_map, shadow_ratio, validation_iteration_count,
                                                                  validation_sample_count)

            the_gan_loss = wrapper.define_loss(the_gan_model)

        # Define GAN train ops.
        train_ops = wrapper.define_train_ops(the_gan_model, the_gan_loss, max_number_of_steps=flags.step,
                                             generator_lr=flags.generator_lr, discriminator_lr=flags.discriminator_lr,
                                             gen_discriminator_lr=flags.gen_discriminator_lr)

        # Training
        status_message = tf.strings.join(
            ["Starting train step: ", tf.as_string(get_or_create_global_step())],
            name="status_message")

        set_all_gpu_config()

        checkpoint_count = flags.step // validation_iteration_count

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


if __name__ == "__main__":
    tf.compat.v1.app.run()
