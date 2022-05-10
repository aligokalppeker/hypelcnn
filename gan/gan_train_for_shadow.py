from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils
import os

import tensorflow as tf
from absl import flags
from tensorflow.contrib.data import shuffle_and_repeat
from tensorflow.python.training.monitored_session import Scaffold, USE_DEFAULT
from tensorflow_gan.python.train import get_sequential_train_hooks

from common_nn_operations import get_class
from cycle_gan_wrapper import CycleGANWrapper
from gan_common import InitializerHook, model_base_name
from gan_sampling_methods import TargetBasedSampler, RandomBasedSampler, DummySampler, NeighborhoodBasedSampler
from gan_wrapper import GANWrapper

required_tensorflow_version = "1.14.0"
if distutils.version.LooseVersion(tf.__version__) < distutils.version.LooseVersion(required_tensorflow_version):
    tfgan = tf.contrib.gan
else:
    import tensorflow_gan as tfgan

flags.DEFINE_integer('batch_size', 128 * 20, 'The number of images in each batch.')

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('train_log_dir', os.path.join(os.path.dirname(__file__), 'log'),
                    'Directory where to write event logs.')

flags.DEFINE_float('generator_lr', 0.0002,
                   'The compression model learning rate.')

flags.DEFINE_float('discriminator_lr', 0.0001,
                   'The discriminator learning rate.')

flags.DEFINE_integer('max_number_of_steps', 50000,
                     'The maximum number of gradient steps.')

flags.DEFINE_string('loader_name', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

flags.DEFINE_string('path', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

flags.DEFINE_string('pairing_method', "random",
                    'Pairing method for the shadowed and non-shadowed samples. Opts: random, target, dummy, neighbour')

flags.DEFINE_string('gan_type', "cycle_gan",
                    'Gan type to train, one of the values can be selected for it; cycle_gan, gan_x2y and gan_y2x')

flags.DEFINE_integer(
    'validation_itr_count', 1000,
    'Iteration count to calculate validation statistics')

flags.DEFINE_integer(
    'validation_sample_count', 300,
    'Validation sample count')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

flags.DEFINE_float('cycle_consistency_loss_weight', 10.0,
                   'The weight of cycle consistency loss')

flags.DEFINE_float('identity_loss_weight', 0.5,
                   'The weight of cycle consistency loss')

flags.DEFINE_bool('use_identity_loss', True,
                  'Whether to use identity loss during training.')

flags.DEFINE_float('regularization_support_rate', 0.0,
                   'The regularization support rate, ranges from 0 to 1, 1 means full support')

FLAGS = flags.FLAGS


def gan_train(train_ops,
              logdir,
              get_hooks_fn=get_sequential_train_hooks(),
              master='',
              is_chief=True,
              scaffold=None,
              hooks=None,
              chief_only_hooks=None,
              save_checkpoint_secs=USE_DEFAULT,
              save_summaries_steps=USE_DEFAULT,
              save_checkpoint_steps=USE_DEFAULT,
              max_wait_secs=7200,
              config=None):
    """A wrapper around `contrib.training.train` that uses GAN hooks.

    Args:
      save_checkpoint_steps: Checkpoint steps to
      train_ops: A GANTrainOps named tuple.
      logdir: The directory where the graph and checkpoints are saved.
      get_hooks_fn: A function that takes a GANTrainOps tuple and returns a list
        of hooks.
      master: The URL of the master.
      is_chief: Specifies whether or not the training is being run by the primary
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
    _validate_gan_train_inputs(logdir, is_chief, save_summaries_steps,
                               save_checkpoint_secs)
    new_hooks = get_hooks_fn(train_ops)
    if hooks is not None:
        hooks = list(hooks) + list(new_hooks)
    else:
        hooks = new_hooks

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


def _validate_gan_train_inputs(logdir, is_chief, save_summaries_steps,
                               save_checkpoint_secs):
    if logdir is None and is_chief:
        if save_summaries_steps:
            raise ValueError(
                "logdir cannot be None when save_summaries_steps is not None")
        if save_checkpoint_secs:
            raise ValueError(
                "logdir cannot be None when save_checkpoint_secs is not None")


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
        raise ValueError("Wrong sampling parameter value (%s)." % pairing_method)

    normal_data_as_matrix = normal_data_as_matrix[:, :, :, 0:data_set.get_casi_band_count()]
    shadow_data_as_matrix = shadow_data_as_matrix[:, :, :, 0:data_set.get_casi_band_count()]

    normal_data_holder = tf.placeholder(dtype=normal_data_as_matrix.dtype, shape=normal_data_as_matrix.shape, name="x")
    shadow_data_holder = tf.placeholder(dtype=shadow_data_as_matrix.dtype, shape=shadow_data_as_matrix.shape, name="y")

    epoch = int((iteration_count * batch_size) / normal_data_as_matrix.shape[0])
    data_set = tf.data.Dataset.from_tensor_slices((normal_data_holder, shadow_data_holder)).apply(
        shuffle_and_repeat(buffer_size=10000, count=epoch))
    data_set = data_set.map(
        lambda param_x, param_y_: perform_shadow_augmentation_random(param_x, param_y_, shadow_ratio, reg_support_rate),
        num_parallel_calls=4)
    data_set = data_set.batch(batch_size)
    data_set_itr = data_set.make_initializable_iterator()

    return InitializerHook(data_set_itr,
                           normal_data_holder, shadow_data_holder,
                           normal_data_as_matrix, shadow_data_as_matrix)


def perform_shadow_augmentation_random(normal_images, shadow_images, shadow_ratio, reg_support_rate):
    with tf.name_scope("shadow_ratio_augmenter"):
        with tf.device("/cpu:0"):
            rand_number_for_augmentation = tf.random_uniform([1], 0.01, 0.99)[0]

            normal_images_rand = tf.cond(tf.less(rand_number_for_augmentation, reg_support_rate),
                                         false_fn=lambda: normal_images,
                                         true_fn=lambda: (shadow_images * shadow_ratio))

            shadow_images_rand = tf.cond(tf.less(rand_number_for_augmentation, reg_support_rate),
                                         false_fn=lambda: shadow_images,
                                         true_fn=lambda: (normal_images / shadow_ratio))

    return normal_images_rand, shadow_images_rand


def _get_lr(base_lr):
    """Returns a learning rate `Tensor`.

    Args:
      base_lr: A scalar float `Tensor` or a Python number.  The base learning
          rate.

    Returns:
      A scalar float `Tensor` of learning rate which equals `base_lr` when the
      global training step is less than FLAGS.max_number_of_steps / 2, afterwards
      it linearly decays to zero.
    """
    global_step = tf.train.get_or_create_global_step()
    lr_constant_steps = FLAGS.max_number_of_steps // 2

    def _lr_decay():
        return tf.train.polynomial_decay(
            learning_rate=base_lr,
            global_step=(global_step - lr_constant_steps),
            decay_steps=(FLAGS.max_number_of_steps - lr_constant_steps),
            end_learning_rate=0.0)

    return tf.cond(global_step < lr_constant_steps, lambda: base_lr, _lr_decay)


def _get_optimizer(gen_lr, dis_lr):
    """Returns generator optimizer and discriminator optimizer.

    Args:
      gen_lr: A scalar float `Tensor` or a Python number.  The Generator learning
          rate.
      dis_lr: A scalar float `Tensor` or a Python number.  The Discriminator
          learning rate.

    Returns:
      A tuple of generator optimizer and discriminator optimizer.
    """
    # beta1 follows
    # https://github.com/junyanz/CycleGAN/blob/master/options.lua
    gen_opt = tf.train.AdamOptimizer(gen_lr, beta1=0.5, use_locking=True)
    dis_opt = tf.train.AdamOptimizer(dis_lr, beta1=0.5, use_locking=True)
    return gen_opt, dis_opt


def _define_train_ops(gan_model, gan_loss):
    """Defines train ops that trains `cyclegan_model` with `cyclegan_loss`.

    Args:
      gan_model: A `CycleGANModel` namedtuple.
      gan_loss: A `CycleGANLoss` namedtuple containing all losses for
          `cyclegan_model`.

    Returns:
      A `GANTrainOps` namedtuple.
    """
    gen_lr = _get_lr(FLAGS.generator_lr)
    dis_lr = _get_lr(FLAGS.discriminator_lr)
    gen_opt, dis_opt = _get_optimizer(gen_lr, dis_lr)

    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=gen_opt,
        discriminator_optimizer=dis_opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True,
        check_for_unused_update_ops=False,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    tf.summary.scalar("generator_lr", gen_lr)
    tf.summary.scalar("discriminator_lr", dis_lr)
    return train_ops


def main(_):
    log_dir = FLAGS.train_log_dir
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        validation_iteration_count = FLAGS.validation_itr_count
        validation_sample_count = FLAGS.validation_sample_count
        loader_name = FLAGS.loader_name
        neighborhood = 0
        loader = get_class(loader_name + "." + loader_name)(FLAGS.path)
        data_set = loader.load_data(neighborhood, True)

        shadow_map, shadow_ratio = loader.load_shadow_map(neighborhood, data_set)

        with tf.name_scope('inputs'):
            initializer_hook = load_op(FLAGS.batch_size, FLAGS.max_number_of_steps, loader, data_set,
                                       shadow_map, shadow_ratio,
                                       FLAGS.regularization_support_rate,
                                       FLAGS.pairing_method)
            training_input_iter = initializer_hook.input_itr
            images_x, images_y = training_input_iter.get_next()
            # Set batch size for summaries.
            # images_x.set_shape([FLAGS.batch_size, None, None, None])
            # images_y.set_shape([FLAGS.batch_size, None, None, None])

        # Define model.
        gan_type = FLAGS.gan_type
        gan_train_wrapper_dict = {
            "cycle_gan": CycleGANWrapper(cycle_consistency_loss_weight=FLAGS.cycle_consistency_loss_weight,
                                         identity_loss_weight=FLAGS.identity_loss_weight,
                                         use_identity_loss=FLAGS.use_identity_loss),
            "gan_x2y": GANWrapper(identity_loss_weight=FLAGS.identity_loss_weight,
                                  use_identity_loss=FLAGS.use_identity_loss,
                                  swap_inputs=False),
            "gan_y2x": GANWrapper(identity_loss_weight=FLAGS.identity_loss_weight,
                                  use_identity_loss=FLAGS.use_identity_loss,
                                  swap_inputs=True)}
        wrapper = gan_train_wrapper_dict[gan_type]

        with tf.variable_scope(model_base_name, reuse=tf.AUTO_REUSE):
            the_gan_model = wrapper.define_model(images_x, images_y)
            peer_validation_hook = wrapper.create_validation_hook(data_set, loader, log_dir, neighborhood,
                                                                  shadow_map, shadow_ratio, validation_iteration_count,
                                                                  validation_sample_count)

            the_gan_loss = wrapper.define_loss(the_gan_model)

        # Define CycleGAN train ops.
        train_ops = _define_train_ops(the_gan_model, the_gan_loss)

        # Training
        train_steps = tfgan.GANTrainSteps(1, 1)
        status_message = tf.string_join(
            [
                "Starting train step: ",
                tf.as_string(tf.train.get_or_create_global_step())
            ],
            name="status_message")
        if not FLAGS.max_number_of_steps:
            return

        gpu = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(gpu[0], True)

        training_scaffold = Scaffold(saver=tf.train.Saver(max_to_keep=20))

        gan_train(
            train_ops,
            log_dir,
            scaffold=training_scaffold,
            save_checkpoint_steps=validation_iteration_count,
            get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
            hooks=[
                initializer_hook,
                peer_validation_hook,
                tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
                tf.train.LoggingTensorHook([status_message], every_n_iter=1000)
            ],
            master=FLAGS.master,
            is_chief=FLAGS.task == 0)


if __name__ == "__main__":
    tf.app.run()
