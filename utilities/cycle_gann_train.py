from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils
import os

import numpy
import tensorflow as tf
from absl import flags
from tensorflow.contrib.data import shuffle_and_repeat
from tensorflow.python.training.monitored_session import Scaffold

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

FLAGS = flags.FLAGS


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


class ValidationHook(tf.train.SessionRunHook):
    @staticmethod
    def export(sess, input_pl, input_np, output_tensor):
        # Grab a single image and run it through inference
        output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
        return output_np

    def __init__(self, iteration_freq, sample_count, log_dir,
                 loader, data_set, neighborhood, shadow_map, shadow_ratio, input_tensor,
                 forward_model):
        self._forward_model = forward_model
        self._input_tensor = input_tensor
        self._iteration_frequency = iteration_freq
        self._global_step_tensor = None
        self._shadow_ratio = shadow_ratio[0:-1]
        self._log_dir = log_dir
        self._data_sample_list = load_samples_for_testing(loader, data_set, sample_count, neighborhood, shadow_map,
                                                          fetch_shadows=False)
        for idx, _data_sample in enumerate(self._data_sample_list):
            self._data_sample_list[idx] = numpy.expand_dims(_data_sample, axis=0)

    def after_create_session(self, session, coord):
        self._global_step_tensor = tf.train.get_global_step()

    def after_run(self, run_context, run_values):
        session = run_context.session
        current_iteration = session.run(self._global_step_tensor)

        if current_iteration % self._iteration_frequency == 1 and current_iteration != 1:
            print('Validation metrics #%d' % current_iteration)
            calculate_stats_from_samples(session, self._data_sample_list, self._input_tensor, self._forward_model,
                                         self._shadow_ratio, self._log_dir, current_iteration,
                                         plt_name="band_ratio_shadowed")


def load_op(batch_size, iteration_count, loader, data_set, shadow_map, shadow_ratio):
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
                                                                     shadow_ratio[0:hsi_channel_len]),
        num_parallel_calls=4)
    data_set = data_set.batch(batch_size)
    data_set_itr = data_set.make_initializable_iterator()

    return InitializerHook(data_set_itr, normal_holder, shadow_holder, normal_data_as_matrix, shadow_data_as_matrix)


def perform_shadow_augmentation_random(normal_images, shadow_images, shadow_ratio):
    with tf.name_scope('shadow_ratio_augmenter'):
        with tf.device('/cpu:0'):
            rand_number = tf.random_uniform([1], 0, 0.5)[0]
            shadow_images = tf.cond(tf.less(rand_number, 1),
                                    true_fn=lambda: shadow_images,
                                    false_fn=lambda: (normal_images / shadow_ratio))
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


def _define_model(images_x, images_y):
    """Defines a CycleGAN model that maps between images_x and images_y.

    Args:
      images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
      images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.

    Returns:
      A `CycleGANModel` namedtuple.
    """
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
                                       shadow_map, shadow_ratio)
            training_input_iter = initializer_hook.input_itr
            images_x, images_y = training_input_iter.get_next()
            # Set batch size for summaries.
            # images_x.set_shape([FLAGS.batch_size, None, None, None])
            # images_y.set_shape([FLAGS.batch_size, None, None, None])

        # Define CycleGAN model.
        with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
            cyclegan_model = _define_model(images_x, images_y)
            input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='x')
            dummy_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='x')
            cyclegan_model_validation = _define_model(input_tensor, dummy_tensor)
            validation_hook = ValidationHook(validation_iteration_count,
                                             validation_sample_count,
                                             log_dir,
                                             loader, data_set, neighborhood,
                                             shadow_map, shadow_ratio,
                                             input_tensor,
                                             cyclegan_model_validation.model_x2y.generated_data)

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
        tfgan.gan_train(
            train_ops,
            log_dir,
            scaffold=training_scaffold,
            save_checkpoint_secs=120,
            get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
            hooks=[
                initializer_hook,
                validation_hook,
                tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
                tf.train.LoggingTensorHook([status_message], every_n_iter=1000)
            ],
            master=FLAGS.master,
            is_chief=FLAGS.task == 0)


if __name__ == '__main__':
    # tf.flags.mark_flag_as_required('image_set_x_file_pattern')
    # tf.flags.mark_flag_as_required('image_set_y_file_pattern')
    tf.app.run()
