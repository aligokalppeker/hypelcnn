from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils
import json
import os
from json import JSONDecodeError

import numpy
import tensorflow as tf
from absl import flags
from tensorflow.contrib.data import shuffle_and_repeat
from tensorflow.python.training.monitored_session import Scaffold, USE_DEFAULT
from tensorflow_gan import gan_loss
from tensorflow_gan.python import namedtuples
from tensorflow_gan.python.losses import tuple_losses
from tensorflow_gan.python.train import _validate_aux_loss_weight, get_sequential_train_hooks

from DataLoader import SampleSet
from common_nn_operations import get_class, get_all_shadowed_normal_data, get_targetbased_shadowed_normal_data
from shadow_data_generator import _shadowdata_generator_model, _shadowdata_discriminator_model, \
    calculate_stats_from_samples, \
    load_samples_for_testing

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

flags.DEFINE_bool('use_target_map', False,
                  'Whether to use target map to create train data pairs.')

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
                'logdir cannot be None when save_summaries_steps is not None')
        if save_checkpoint_secs:
            raise ValueError(
                'logdir cannot be None when save_checkpoint_secs is not None')


class CycleGANModelWithIdentity(tfgan.CycleGANModel):
    """An CycleGANModel contains all the pieces needed for CycleGAN training and additional outputs for calculating identity loss.

    The model `model_x2y` generator F maps data set X to Y, while the model
    `model_y2x` generator G maps data set Y to X.

    See https://arxiv.org/abs/1703.10593 for more details.

    Args:
      model_x2y: A `GANModel` namedtuple whose generator maps data set X to Y.
      model_y2x: A `GANModel` namedtuple whose generator maps data set Y to X.
      reconstructed_x: A `Tensor` of reconstructed data X which is G(F(X)).
      reconstructed_y: A `Tensor` of reconstructed data Y which is F(G(Y)).
      identity_x: A `Tensor` of data X processed by generator function F which is F(X).
      identity_y: A `Tensor` of data Y processed by generator function G which is G(Y).
    """
    __slots__ = ()
    _fields = tfgan.CycleGANModel._fields + ('identity_x', 'identity_y',)


def cyclegan_model_with_identity(
        # Lambdas defining models.
        generator_fn,
        discriminator_fn,
        # data X and Y.
        data_x,
        data_y,
        # Optional scopes.
        generator_scope='Generator',
        discriminator_scope='Discriminator',
        model_x2y_scope='ModelX2Y',
        model_y2x_scope='ModelY2X',
        # Options.
        check_shapes=True):
    """Returns a CycleGAN model outputs and variables.

    See https://arxiv.org/abs/1703.10593 for more details.

    Args:
      generator_fn: A python lambda that takes `data_x` or `data_y` as inputs and
        returns the outputs of the GAN generator.
      discriminator_fn: A python lambda that takes `real_data`/`generated data`
        and `generator_inputs`. Outputs a Tensor in the range [-inf, inf].
      data_x: A `Tensor` of dataset X. Must be the same shape as `data_y`.
      data_y: A `Tensor` of dataset Y. Must be the same shape as `data_x`.
      generator_scope: Optional generator variable scope. Useful if you want to
        reuse a subgraph that has already been created. Defaults to 'Generator'.
      discriminator_scope: Optional discriminator variable scope. Useful if you
        want to reuse a subgraph that has already been created. Defaults to
        'Discriminator'.
      model_x2y_scope: Optional variable scope for model x2y variables. Defaults
        to 'ModelX2Y'.
      model_y2x_scope: Optional variable scope for model y2x variables. Defaults
        to 'ModelY2X'.
      check_shapes: If `True`, check that generator produces Tensors that are the
        same shape as `data_x` (`data_y`). Otherwise, skip this check.

    Returns:
      A `CycleGANModel` namedtuple.

    Raises:
      ValueError: If `check_shapes` is True and `data_x` or the generator output
        does not have the same shape as `data_y`.
      ValueError: If TF is executing eagerly.
    """
    original_model = tfgan.cyclegan_model(generator_fn=generator_fn, discriminator_fn=discriminator_fn, data_x=data_x,
                                          data_y=data_y, generator_scope=generator_scope,
                                          discriminator_scope=discriminator_scope, model_x2y_scope=model_x2y_scope,
                                          model_y2x_scope=model_y2x_scope, check_shapes=check_shapes)

    with tf.compat.v1.variable_scope(original_model.model_x2y.generator_scope, reuse=True):
        identity_x = original_model.model_x2y.generator_fn(data_x)
    with tf.compat.v1.variable_scope(original_model.model_y2x.generator_scope, reuse=True):
        identity_y = original_model.model_y2x.generator_fn(data_y)

    model_w_identity = CycleGANModelWithIdentity(model_x2y=original_model.model_x2y,
                                                 model_y2x=original_model.model_y2x,
                                                 reconstructed_x=original_model.reconstructed_x,
                                                 reconstructed_y=original_model.reconstructed_y)
    model_w_identity.identity_x = identity_x
    model_w_identity.identity_y = identity_y

    return model_w_identity


def identity_loss(model, kwargs):
    # Defines identity loss
    identity_loss_x = tf.compat.v1.losses.absolute_difference(model.model_x2y.generator_inputs,
                                                              model.identity_x)
    identity_loss_y = tf.compat.v1.losses.absolute_difference(model.model_y2x.generator_inputs,
                                                              model.identity_y)
    if kwargs.get('add_summaries', True):
        tf.compat.v1.summary.scalar('identity_loss_x', identity_loss_x)
        tf.compat.v1.summary.scalar('identity_loss_y', identity_loss_y)

    return identity_loss_x, identity_loss_y


def cyclegan_loss_with_identity(
        model,
        # Loss functions.
        generator_loss_fn=tuple_losses.least_squares_generator_loss,
        discriminator_loss_fn=tuple_losses.least_squares_discriminator_loss,
        # Auxiliary losses.
        cycle_consistency_loss_fn=tuple_losses.cycle_consistency_loss,
        cycle_consistency_loss_weight=10.0,
        identity_loss_weight=0.5,
        # Options
        **kwargs):
    """Returns the losses for a `CycleGANModel`.

    See https://arxiv.org/abs/1703.10593 for more details.

    Args:
      model: A `CycleGANModel` namedtuple.
      generator_loss_fn: The loss function on the generator. Takes a `GANModel`
        named tuple.
      discriminator_loss_fn: The loss function on the discriminator. Takes a
        `GANModel` namedtuple.
      cycle_consistency_loss_fn: The cycle consistency loss function. Takes a
        `CycleGANModel` namedtuple.
      cycle_consistency_loss_weight: A non-negative Python number or a scalar
        `Tensor` indicating how much to weigh the cycle consistency loss.
      identity_loss_weight: A non-negative Python number or a scalar
        `Tensor` indicating how much to weigh the identity loss.
      **kwargs: Keyword args to pass directly to `gan_loss` to construct the loss
        for each partial model of `model`.

    Returns:
      A `CycleGANLoss` namedtuple.

    Raises:
      ValueError: If `model` is not a `CycleGANModel` namedtuple.
    """
    # Sanity checks.
    if not isinstance(model, CycleGANModelWithIdentity):
        raise ValueError(
            '`model` must be a `CycleGANModelWithIdentity`. Instead, was %s.' % type(model))

    identity_loss_x, identity_loss_y = identity_loss(model, kwargs)

    # Defines cycle consistency loss.
    cycle_consistency_loss = cycle_consistency_loss_fn(
        model, add_summaries=kwargs.get('add_summaries', True))
    cycle_consistency_loss_weight = _validate_aux_loss_weight(
        cycle_consistency_loss_weight, 'cycle_consistency_loss_weight')

    aux_loss = (cycle_consistency_loss_weight * cycle_consistency_loss) + (
            identity_loss_weight * (identity_loss_x + identity_loss_y))

    # Defines losses for each partial model.
    def _partial_loss(partial_model):
        partial_loss = gan_loss(
            partial_model,
            generator_loss_fn=generator_loss_fn,
            discriminator_loss_fn=discriminator_loss_fn,
            **kwargs)
        return partial_loss._replace(generator_loss=partial_loss.generator_loss +
                                                    aux_loss)

    with tf.compat.v1.name_scope('cyclegan_loss_x2y'):
        loss_x2y = _partial_loss(model.model_x2y)
    with tf.compat.v1.name_scope('cyclegan_loss_y2x'):
        loss_y2x = _partial_loss(model.model_y2x)

    return namedtuples.CycleGANLoss(loss_x2y, loss_y2x)


def create_dummy_shadowed_normal_data(data_set, loader):
    data_shape_info = loader.get_data_shape(data_set)
    element_count = 2000
    shadow_data_as_matrix = numpy.full(numpy.concatenate([[element_count], data_shape_info]),
                                       fill_value=0.5, dtype=numpy.float32)

    return shadow_data_as_matrix * 2, shadow_data_as_matrix


class InitializerHook(tf.train.SessionRunHook):

    def __init__(self, input_itr, normal_placeholder, shadow_placeholder, normal_data, shadow_data):
        self.input_itr = input_itr
        self.shadow_data = shadow_data
        self.normal_data = normal_data
        self.shadow_placeholder = shadow_placeholder
        self.normal_placeholder = normal_placeholder

    def after_create_session(self, session, coord):
        session.run(self.input_itr.initializer,
                    feed_dict={self.shadow_placeholder: self.shadow_data,
                               self.normal_placeholder: self.normal_data})


class BestRatioHolder:

    def __init__(self, max_size) -> None:
        super().__init__()
        self.data_holder = []
        self.max_size = max_size

    def add_point(self, iteration, kl_val):
        iteration = int(iteration)  # For seralization purposes int64 => int
        kl_val = float(kl_val)  # For seralization purposes float64 => float
        insert_idx = 0
        for (curr_iter, curr_kl) in self.data_holder:
            if kl_val > curr_kl:
                insert_idx = insert_idx + 1

        self.data_holder.insert(insert_idx, (iteration, kl_val))
        if len(self.data_holder) > self.max_size:
            self.data_holder.pop()

    def get_point_with_itr(self, iteration):
        result = (None, None)
        for (curr_iter, curr_kl) in self.data_holder:
            if curr_iter == iteration:
                result = (curr_iter, curr_kl)
                break

        return result

    def load(self, file_address):
        try:
            with open(file_address, "rb") as read_file:
                self.data_holder = json.load(read_file)
            print(f"Best ratio file {file_address} is loaded.", self.data_holder)
        except IOError:
            print(f"File {file_address} file found. No best ratio is loaded.")
        except JSONDecodeError:
            print(f"File {file_address} file can not be decoded. No best ratio is loaded.")

    def save(self, file_address):
        serialized_out = json.dumps(self.data_holder)
        with open(file_address, "w") as write_file:
            write_file.write(serialized_out)

    @staticmethod
    def create_common_iterations(ratio_holder_1, ratio_holder_2):
        result = BestRatioHolder(ratio_holder_1.max_size)
        for (curr_iter, curr_kl) in ratio_holder_1.data_holder:
            (found_itr, found_kl) = ratio_holder_2.get_point_with_itr(curr_iter)
            if found_itr is not None:
                result.add_point(found_itr, found_kl)

        return result

    def __str__(self) -> str:
        return str(self.data_holder)


class BaseValidationHook(tf.train.SessionRunHook):
    @staticmethod
    def export(sess, input_pl, input_np, output_tensor):
        # Grab a single image and run it through inference
        output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
        return output_np

    def __init__(self, iteration_freq, log_dir, shadow_ratio):
        self._iteration_frequency = iteration_freq
        self._global_step_tensor = None
        self._shadow_ratio = shadow_ratio[0:-1]
        self._log_dir = log_dir
        self.best_ratio_holder = BestRatioHolder(10)
        self.validation_itr_mark = False

    def after_create_session(self, session, coord):
        self._global_step_tensor = tf.train.get_global_step()

    def _is_validation_itr(self, current_iteration):
        return current_iteration % self._iteration_frequency == 1 and current_iteration != 1


class PeerValidationHook(tf.train.SessionRunHook):
    def __init__(self, *validation_base_hooks):
        self._validation_base_hooks = validation_base_hooks

    def after_create_session(self, session, coord):
        for validation_base_hook in self._validation_base_hooks:
            validation_base_hook.after_create_session(session, coord)

    def after_run(self, run_context, run_values):
        ratio_holder_list = []
        for validation_base_hook in self._validation_base_hooks:
            validation_base_hook.after_run(run_context, run_values)
            ratio_holder_list.append(validation_base_hook.best_ratio_holder)
        if self._validation_base_hooks[0].validation_itr_mark:
            print("Best common options:",
                  BestRatioHolder.create_common_iterations(ratio_holder_list[0], ratio_holder_list[1]))


class ValidationHook(BaseValidationHook):

    def __init__(self, iteration_freq, sample_count, log_dir, loader, data_set, neighborhood, shadow_map, shadow_ratio,
                 input_tensor, model, name_suffix, fetch_shadows):
        super().__init__(iteration_freq, log_dir, shadow_ratio)
        self._forward_model = model
        self._input_tensor = input_tensor
        self._name_suffix = name_suffix
        self._plt_name = f"band_ratio_{name_suffix}"
        self._best_ratio_addr = os.path.join(self._log_dir, f"best_ratio_{name_suffix}.json")
        self.best_ratio_holder.load(self._best_ratio_addr)
        self._data_sample_list = load_samples_for_testing(loader, data_set, sample_count, neighborhood,
                                                          shadow_map, fetch_shadows=fetch_shadows)
        for idx, _data_sample in enumerate(self._data_sample_list):
            self._data_sample_list[idx] = numpy.expand_dims(_data_sample, axis=0)

    def after_run(self, run_context, run_values):
        session = run_context.session
        current_iteration = session.run(self._global_step_tensor)

        self.validation_itr_mark = self._is_validation_itr(current_iteration)
        if self.validation_itr_mark:
            print(f"Validation metrics for {self._name_suffix} #{current_iteration}")
            kl_shadowed = calculate_stats_from_samples(session, self._data_sample_list, self._input_tensor,
                                                       self._forward_model,
                                                       self._shadow_ratio, self._log_dir, current_iteration,
                                                       plt_name=self._plt_name)
            self.best_ratio_holder.add_point(current_iteration, kl_shadowed)
            self.best_ratio_holder.save(self._best_ratio_addr)
            print(f"KL divergence for {self._name_suffix}:{kl_shadowed}")
            print(f"Best {self._name_suffix} options:{self.best_ratio_holder}")


def load_op(batch_size, iteration_count, loader, data_set, shadow_map, shadow_ratio, reg_support_rate):
    if FLAGS.use_target_map:
        normal_data_as_matrix, shadow_data_as_matrix = get_data_from_scene(data_set, loader, shadow_map)
    else:
        normal_data_as_matrix, shadow_data_as_matrix = get_all_shadowed_normal_data(
            data_set,
            loader,
            shadow_map, multiply_shadowed_data=False)

    # normal_data_as_matrix, shadow_data_as_matrix = create_dummy_shadowed_normal_data(data_set, loader)

    hsi_channel_len = normal_data_as_matrix.shape[3] - 1
    normal_data_as_matrix = normal_data_as_matrix[:, :, :, 0:hsi_channel_len]
    shadow_data_as_matrix = shadow_data_as_matrix[:, :, :, 0:hsi_channel_len]

    normal_holder = tf.placeholder(dtype=normal_data_as_matrix.dtype, shape=normal_data_as_matrix.shape, name='x')
    shadow_holder = tf.placeholder(dtype=shadow_data_as_matrix.dtype, shape=shadow_data_as_matrix.shape, name='y')

    epoch = int((iteration_count * batch_size) / normal_data_as_matrix.shape[0])
    data_set = tf.data.Dataset.from_tensor_slices((normal_holder, shadow_holder)).apply(
        shuffle_and_repeat(buffer_size=10000, count=epoch))
    data_set = data_set.map(
        lambda param_x, param_y_: perform_shadow_augmentation_random(param_x, param_y_,
                                                                     shadow_ratio[0:hsi_channel_len],
                                                                     reg_support_rate),
        num_parallel_calls=4)
    data_set = data_set.batch(batch_size)
    data_set_itr = data_set.make_initializable_iterator()

    return InitializerHook(data_set_itr, normal_holder, shadow_holder, normal_data_as_matrix, shadow_data_as_matrix)


def perform_shadow_augmentation_random(normal_images, shadow_images, shadow_ratio, reg_support_rate):
    with tf.name_scope('shadow_ratio_augmenter'):
        with tf.device('/cpu:0'):
            rand_number = tf.random_uniform([1], 0.01, 0.99)[0]
            shadow_images = tf.cond(tf.less(rand_number, reg_support_rate),
                                    false_fn=lambda: shadow_images,
                                    true_fn=lambda: (normal_images / shadow_ratio))
    return normal_images, shadow_images


def get_data_from_scene(data_set, loader, shadow_map):
    samples = SampleSet(training_targets=loader.read_targets("shadow_cycle_gan/class_result.tif"),
                        test_targets=None,
                        validation_targets=None)
    first_margin_start = 5
    first_margin_end = loader.get_scene_shape(data_set)[0] - 5
    second_margin_start = 5
    second_margin_end = loader.get_scene_shape(data_set)[1] - 5
    for target_index in range(0, samples.training_targets.shape[0]):
        current_target = samples.training_targets[target_index]
        if not (first_margin_start < current_target[1] < first_margin_end and
                second_margin_start < current_target[0] < second_margin_end):
            current_target[2] = -1
    normal_data_as_matrix, shadow_data_as_matrix = get_targetbased_shadowed_normal_data(data_set, loader, shadow_map,
                                                                                        samples)
    return normal_data_as_matrix, shadow_data_as_matrix


def _define_model(images_x, images_y, use_identity_loss):
    """Defines a CycleGAN model that maps between images_x and images_y.

    Args:
      images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
      images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.
      use_identity_loss: Whether to use identity loss or not

    Returns:
      A `CycleGANModel` namedtuple.
    """
    if use_identity_loss:
        cyclegan_model = cyclegan_model_with_identity(
            generator_fn=_shadowdata_generator_model,
            discriminator_fn=_shadowdata_discriminator_model,
            data_x=images_x,
            data_y=images_y)
    else:
        cyclegan_model = tfgan.cyclegan_model(
            generator_fn=_shadowdata_generator_model,
            discriminator_fn=_shadowdata_discriminator_model,
            data_x=images_x,
            data_y=images_y)

    # Add summaries for generated images.
    # tfgan.eval.add_cyclegan_image_summaries(cyclegan_model)

    return cyclegan_model


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


def _define_train_ops(cyclegan_model, cyclegan_loss):
    """Defines train ops that trains `cyclegan_model` with `cyclegan_loss`.

    Args:
      cyclegan_model: A `CycleGANModel` namedtuple.
      cyclegan_loss: A `CycleGANLoss` namedtuple containing all losses for
          `cyclegan_model`.

    Returns:
      A `GANTrainOps` namedtuple.
    """
    gen_lr = _get_lr(FLAGS.generator_lr)
    dis_lr = _get_lr(FLAGS.discriminator_lr)
    gen_opt, dis_opt = _get_optimizer(gen_lr, dis_lr)

    train_ops = tfgan.gan_train_ops(
        cyclegan_model,
        cyclegan_loss,
        generator_optimizer=gen_opt,
        discriminator_optimizer=dis_opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True,
        check_for_unused_update_ops=False,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    tf.summary.scalar('generator_lr', gen_lr)
    tf.summary.scalar('discriminator_lr', dis_lr)
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
        loader = get_class(loader_name + '.' + loader_name)(FLAGS.path)
        data_set = loader.load_data(neighborhood, True)

        element_size = loader.get_data_shape(data_set)
        element_size = [1, element_size[0], element_size[1], element_size[2] - 1]

        shadow_map, shadow_ratio = loader.load_shadow_map(neighborhood, data_set)

        with tf.name_scope('inputs'):
            initializer_hook = load_op(FLAGS.batch_size, FLAGS.max_number_of_steps, loader, data_set,
                                       shadow_map, shadow_ratio, FLAGS.regularization_support_rate)
            training_input_iter = initializer_hook.input_itr
            images_x, images_y = training_input_iter.get_next()
            # Set batch size for summaries.
            # images_x.set_shape([FLAGS.batch_size, None, None, None])
            # images_y.set_shape([FLAGS.batch_size, None, None, None])

        # Define CycleGAN model.
        with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
            cyclegan_model = _define_model(images_x, images_y, FLAGS.use_identity_loss)
            x_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='x')
            y_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='y')
            cyclegan_model_for_validation = _define_model(x_input_tensor, y_input_tensor, FLAGS.use_identity_loss)
            shadowed_validation_hook = ValidationHook(iteration_freq=validation_iteration_count,
                                                      sample_count=validation_sample_count,
                                                      log_dir=log_dir,
                                                      loader=loader, data_set=data_set, neighborhood=neighborhood,
                                                      shadow_map=shadow_map, shadow_ratio=shadow_ratio,
                                                      input_tensor=x_input_tensor,
                                                      model=cyclegan_model_for_validation.model_x2y.generated_data,
                                                      fetch_shadows=False, name_suffix="shadowed")
            de_shadowed_validation_hook = ValidationHook(iteration_freq=validation_iteration_count,
                                                         sample_count=validation_sample_count,
                                                         log_dir=log_dir,
                                                         loader=loader, data_set=data_set, neighborhood=neighborhood,
                                                         shadow_map=shadow_map, shadow_ratio=1. / shadow_ratio,
                                                         input_tensor=y_input_tensor,
                                                         model=cyclegan_model_for_validation.model_y2x.generated_data,
                                                         fetch_shadows=True, name_suffix="deshadowed")
            peer_validation_hook = PeerValidationHook(shadowed_validation_hook, de_shadowed_validation_hook)

        if FLAGS.use_identity_loss:
            cyclegan_loss = cyclegan_loss_with_identity(
                cyclegan_model,
                # generator_loss_fn=wasserstein_generator_loss,
                # discriminator_loss_fn=wasserstein_discriminator_loss,
                cycle_consistency_loss_weight=FLAGS.cycle_consistency_loss_weight,
                identity_loss_weight=FLAGS.identity_loss_weight,
                tensor_pool_fn=tfgan.features.tensor_pool)
        else:
            # Define CycleGAN loss.
            cyclegan_loss = tfgan.cyclegan_loss(
                cyclegan_model,
                # generator_loss_fn=wasserstein_generator_loss,
                # discriminator_loss_fn=wasserstein_discriminator_loss,
                cycle_consistency_loss_weight=FLAGS.cycle_consistency_loss_weight,
                tensor_pool_fn=tfgan.features.tensor_pool)

        # Define CycleGAN train ops.
        train_ops = _define_train_ops(cyclegan_model, cyclegan_loss)

        # Training
        train_steps = tfgan.GANTrainSteps(1, 1)
        status_message = tf.string_join(
            [
                'Starting train step: ',
                tf.as_string(tf.train.get_or_create_global_step())
            ],
            name='status_message')
        if not FLAGS.max_number_of_steps:
            return

        gpu = tf.config.experimental.list_physical_devices('GPU')
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


if __name__ == '__main__':
    # tf.flags.mark_flag_as_required('image_set_x_file_pattern')
    # tf.flags.mark_flag_as_required('image_set_y_file_pattern')
    tf.app.run()
