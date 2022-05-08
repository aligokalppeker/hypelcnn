from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils

import numpy
import tensorflow as tf
from absl import flags
from tensorflow_core.python.training.session_run_hook import SessionRunContext

from common_nn_operations import get_class
from cycle_gan_wrapper import CycleGANInferenceWrapper
from gan_wrapper import GANInferenceWrapper

required_tensorflow_version = "1.14.0"
if distutils.version.LooseVersion(tf.__version__) < distutils.version.LooseVersion(required_tensorflow_version):
    tfgan = tf.contrib.gan
else:
    pass

flags.DEFINE_integer('neighborhood', 0, 'Neighborhood of samples.')
flags.DEFINE_integer('number_of_samples', 6000, 'Number of samples.')
flags.DEFINE_string('checkpoint_path', '',
                    'GAN checkpoint path created by gan_train_for_shadow.py. '
                    '(e.g. "/mylogdir/model.ckpt-18442")')

flags.DEFINE_string('loader_name', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

flags.DEFINE_string('path', "C:/GoogleDriveBack/PHD/Tez/Source",
                    'Directory where to read the image inputs.')

flags.DEFINE_string('gan_type', "cycle_gan",
                    'Gan type to train, one of the values can be selected for it; cycle_gan, gan_x2y and gan_y2x')

FLAGS = flags.FLAGS


def _validate_flags():
    flags.register_validator('checkpoint_path', bool,
                             'Must provide `checkpoint_path`.')


def main(_):
    numpy.set_printoptions(precision=5, suppress=True)
    neighborhood = FLAGS.neighborhood

    _validate_flags()

    loader_name = FLAGS.loader_name
    loader = get_class(loader_name + '.' + loader_name)(FLAGS.path)
    data_set = loader.load_data(neighborhood, True)
    log_dir = "./"

    # test_x_data = numpy.full([1, 1, band_size], fill_value=1.0, dtype=float)
    # print_tensors_in_checkpoint_file(FLAGS.checkpoint_path, tensor_name='ModelX2Y', all_tensors=True)
    shadow_map, shadow_ratio = loader.load_shadow_map(neighborhood, data_set)

    gan_inference_wrapper_dict = {"cycle_gan": CycleGANInferenceWrapper(),
                                  "gan_x2y": GANInferenceWrapper(fetch_shadows=False),
                                  "gan_y2x": GANInferenceWrapper(fetch_shadows=True)}

    hook = gan_inference_wrapper_dict[FLAGS.gan_type].create_inference_hook(data_set=data_set, loader=loader,
                                                                            log_dir=log_dir,
                                                                            neighborhood=neighborhood,
                                                                            shadow_map=shadow_map,
                                                                            shadow_ratio=shadow_ratio,
                                                                            validation_sample_count=FLAGS.number_of_samples)

    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    with tf.Session() as sess:
        gan_inference_wrapper_dict[FLAGS.gan_type].create_generator_restorer().restore(sess, FLAGS.checkpoint_path)
        hook.after_create_session(sess, None)
        run_context = SessionRunContext(original_args=None, session=sess)
        hook.after_run(run_context=run_context, run_values=None)

        # normal_data_as_matrix, shadow_data_as_matrix = loader.get_targetbased_shadowed_normal_data(data_set,
        #                                                                                            loader,
        #                                                                                            shadow_map,
        #                                                                                            loader.load_samples(
        #                                                                                                0.1))
        # # normal_data_as_matrix, shadow_data_as_matrix = loader.get_all_shadowed_normal_data(data_set,
        # #                                                                             loader,
        # #                                                                             shadow_map)
        # print("Target based shadow index")
        # print(1 / numpy.squeeze(numpy.mean(normal_data_as_matrix, axis=0) / numpy.mean(shadow_data_as_matrix, axis=0)))


if __name__ == '__main__':
    tf.app.run()
