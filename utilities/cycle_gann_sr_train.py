from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils
import math
import os
from math import floor, ceil

import cv2
import numpy
import tensorflow as tf
from absl import flags
from tensorflow.contrib.data import shuffle_and_repeat
from tifffile import imwrite

from GRSS2013DataLoader import GRSS2013DataLoader
from GRSS2018DataLoader import GRSS2018DataLoader
from sr_data_generator import _srdata_generator_model, _srdata_discriminator_model, extract_common_normalizer

required_tensorflow_version = "1.14.0"
if distutils.version.LooseVersion(tf.__version__) < distutils.version.LooseVersion(required_tensorflow_version):
    tfgan = tf.contrib.gan
else:
    import tensorflow_gan as tfgan

flags.DEFINE_integer('batch_size', 3, 'The number of images in each batch.')

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('train_log_dir', os.path.join(os.path.dirname(__file__), 'log'),
                    'Directory where to write event logs.')

flags.DEFINE_string('path', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

flags.DEFINE_float('generator_lr', 0.0002,
                   'The compression model learning rate.')

flags.DEFINE_float('discriminator_lr', 0.0002,
                   'The discriminator learning rate.')

flags.DEFINE_integer('max_number_of_steps', 5000,
                     'The maximum number of gradient steps.')

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


class InitializerHook(tf.train.SessionRunHook):

    def __init__(self, input_itr):
        self.input_itr = input_itr

    def after_create_session(self, session, coord):
        session.run(self.input_itr.initializer)


def _matched_data_generator(hsi_grss2013, hsi_grss2018, hsi_2013_spatial_repeat, hsi_2018_spectral_repeat, scale_diff,
                            x_start, y_start, x_end, y_end, scale_offset):
    hsi_2018_spatial_repeat = int(hsi_2013_spatial_repeat / scale_diff)
    start_x = x_start + scale_offset
    start_y = y_start + scale_offset
    end_x = x_end - scale_offset
    end_y = y_end - scale_offset

    for hsi2013_y_idx in range(start_y, end_y):
        for hsi2013_x_idx in range(start_x, end_x):
            hsi_2018_data = get_grss2018_data(hsi_grss2018, hsi2013_x_idx, hsi2013_y_idx, scale_diff,
                                              hsi_2018_spatial_repeat, hsi_2018_spectral_repeat)
            hsi_2013_data = numpy.repeat(
                numpy.repeat(numpy.expand_dims(hsi_grss2013[hsi2013_x_idx, hsi2013_y_idx, :], axis=(0, 1)),
                             hsi_2013_spatial_repeat,
                             axis=1), hsi_2013_spatial_repeat, axis=0)

            # mean_list.append(numpy.mean(numpy.mean(hsi_2013_data / hsi_2018_data, axis=0), axis=0))
            yield hsi_2013_data, hsi_2018_data


def get_grss2018_data(hsi_grss2018, hsi2013_x_idx, hsi2013_y_idx, scale, spatial_repeat, spectral_repeat):
    hsi2018_start_x_idx = ((hsi2013_x_idx * scale) - (scale / 2))
    hsi2018_start_y_idx = ((hsi2013_y_idx * scale) - (scale / 2))
    hsi2018_end_x_idx = (hsi2018_start_x_idx + scale)
    hsi2018_end_y_idx = (hsi2018_start_y_idx + scale)
    start_x_diff = int((hsi2018_start_x_idx - floor(hsi2018_start_x_idx)) * spatial_repeat)
    end_x_diff = int((ceil(hsi2018_end_x_idx) - hsi2018_end_x_idx) * spatial_repeat)
    start_y_diff = int((hsi2018_start_y_idx - floor(hsi2018_start_y_idx)) * spatial_repeat)
    end_y_diff = int((ceil(hsi2018_end_y_idx) - hsi2018_end_y_idx) * spatial_repeat)

    grss_2018_range = hsi_grss2018[math.floor(hsi2018_start_x_idx):ceil(hsi2018_end_x_idx),
                      math.floor(hsi2018_start_y_idx):ceil(hsi2018_end_y_idx), :]
    grss_2018_scaled = numpy.repeat(numpy.repeat(grss_2018_range, spatial_repeat, axis=1), spatial_repeat, axis=0)
    grss_2018_scaled = grss_2018_scaled[start_x_diff:-end_x_diff, start_y_diff:-end_y_diff, :]
    grss_2018_scaled = numpy.repeat(grss_2018_scaled, spectral_repeat, axis=2)
    return grss_2018_scaled


def load_op(batch_size, iteration_count):
    hsi_2013_spatial_repeat = 10
    hsi_2018_spectral_repeat = 3

    hsi_2013_scale_diff = 2.5
    neighborhood = 0
    band_size = 144

    grss2013_data_set = GRSS2013DataLoader(FLAGS.path).load_data(neighborhood, False)
    grss2018_data_set = GRSS2018DataLoader(FLAGS.path).load_data(neighborhood, False)
    hsi2013_global_minimum, hsi2018_global_minimum, hsi2013_global_maximum, hsi2018_global_maximum = \
        extract_common_normalizer(grss2013_data_set.concrete_data[:, :, 0:-1], grss2018_data_set.casi)

    hsi_grss2013 = grss2013_data_set.concrete_data[7:347, 256 + 8:1894 + 8, 0:-1]
    hsi_grss2018 = grss2018_data_set.casi[0 + 265:-350 + 265, 0:-75, :].astype(numpy.float32)

    debug_data = False
    if debug_data:
        test_match(band_size, hsi_2013_scale_diff,
                   hsi_2013_spatial_repeat, hsi_2018_spectral_repeat,
                   hsi_grss2013, hsi_grss2018)

    hsi_grss2013 -= hsi2013_global_minimum
    hsi_grss2018 -= hsi2018_global_minimum
    hsi_grss2013 /= hsi2013_global_maximum.astype(numpy.float32)
    hsi_grss2018 /= hsi2018_global_maximum.astype(numpy.float32)

    tensor_output_shape = [hsi_2013_spatial_repeat, hsi_2013_spatial_repeat, band_size]

    tensor_type_info = (tf.float32, tf.float32)
    epoch = int((iteration_count * batch_size) / hsi_grss2013.shape[0])
    data_set = tf.data.Dataset.from_generator(
        lambda: _matched_data_generator(hsi_grss2013, hsi_grss2018, hsi_2013_spatial_repeat,
                                        hsi_2018_spectral_repeat, hsi_2013_scale_diff,
                                        0, 0, hsi_grss2013.shape[0], hsi_grss2013.shape[1],
                                        ceil(hsi_2013_scale_diff / 2)), tensor_type_info,
        (tensor_output_shape, tensor_output_shape))
    data_set = data_set.apply(shuffle_and_repeat(buffer_size=10000, count=epoch))
    data_set = data_set.batch(batch_size)
    data_set_itr = data_set.make_initializable_iterator()

    return InitializerHook(data_set_itr)


def test_match(band_size, hsi_2013_scale_diff, hsi_2013_spatial_repeat, hsi_2018_spectral_repeat, hsi_grss2013,
               hsi_grss2018):
    sample_width = 250
    sample_height = 250
    start_x = 10
    start_y = 10
    grss2013_sample_image_output = numpy.zeros(
        [sample_height * hsi_2013_spatial_repeat, sample_width * hsi_2013_spatial_repeat, band_size],
        dtype=numpy.uint16)
    grss2018_sample_image_output = numpy.zeros(
        [sample_height * hsi_2013_spatial_repeat, sample_width * hsi_2013_spatial_repeat, band_size],
        dtype=numpy.uint16)
    cur_sample_index = 0
    for grss2013, grss2018 in _matched_data_generator(hsi_grss2013, hsi_grss2018, hsi_2013_spatial_repeat,
                                                      hsi_2018_spectral_repeat, hsi_2013_scale_diff,
                                                      start_x, start_y,
                                                      start_x + sample_width,
                                                      start_y + sample_height, 0):
        cur_start_x = (cur_sample_index % sample_width) * hsi_2013_spatial_repeat
        cur_start_y = floor(cur_sample_index / sample_width) * hsi_2013_spatial_repeat
        cur_end_x = cur_start_x + hsi_2013_spatial_repeat
        cur_end_y = cur_start_y + hsi_2013_spatial_repeat
        grss2013_sample_image_output[cur_start_x:cur_end_x, cur_start_y:cur_end_y, :] = grss2013.astype(numpy.uint16)
        grss2018_sample_image_output[cur_start_x:cur_end_x, cur_start_y:cur_end_y, :] = grss2018.astype(numpy.uint16)
        cur_sample_index = cur_sample_index + 1

    grss2013_sample_image_output = cv2.resize(grss2013_sample_image_output,
                                              (int(grss2013_sample_image_output.shape[0] / 2),
                                               int(grss2013_sample_image_output.shape[1] / 2)),
                                              interpolation=cv2.INTER_LINEAR)
    grss2018_sample_image_output = cv2.resize(grss2018_sample_image_output,
                                              (int(grss2018_sample_image_output.shape[0] / 2),
                                               int(grss2018_sample_image_output.shape[1] / 2)),
                                              interpolation=cv2.INTER_LINEAR)

    imwrite("grss2013_sample_image.tif", grss2013_sample_image_output, planarconfig='contig')
    imwrite("grss2018_sample_image.tif", grss2018_sample_image_output, planarconfig='contig')


def _define_model(images_x, images_y):
    """Defines a CycleGAN model that maps between images_x and images_y.

    Args:
      images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
      images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.

    Returns:
      A `CycleGANModel` namedtuple.
    """
    cyclegan_model = tfgan.cyclegan_model(
        generator_fn=lambda netinput: _srdata_generator_model(netinput, True),
        discriminator_fn=lambda gendata, geninput: _srdata_discriminator_model(gendata, geninput, True),
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
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        with tf.name_scope('inputs'):
            initializer_hook = load_op(FLAGS.batch_size, FLAGS.max_number_of_steps)
            training_input_iter = initializer_hook.input_itr
            images_x, images_y = training_input_iter.get_next()
            # Set batch size for summaries.
            # images_x.set_shape([FLAGS.batch_size, None, None, None])
            # images_y.set_shape([FLAGS.batch_size, None, None, None])

        # Define CycleGAN model.
        cyclegan_model = _define_model(images_x, images_y)

        # Define CycleGAN loss.
        cyclegan_loss = tfgan.cyclegan_loss(
            cyclegan_model,
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
        tfgan.gan_train(
            train_ops,
            FLAGS.train_log_dir,
            save_checkpoint_secs=60*10,
            get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
            hooks=[
                initializer_hook,
                tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
                tf.train.LoggingTensorHook([status_message], every_n_iter=10)
            ],
            master=FLAGS.master,
            is_chief=FLAGS.task == 0)


if __name__ == '__main__':
    # tf.flags.mark_flag_as_required('image_set_x_file_pattern')
    # tf.flags.mark_flag_as_required('image_set_y_file_pattern')
    tf.app.run()
