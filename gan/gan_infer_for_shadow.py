import argparse

import numpy
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunContext

from common.cmd_parser import add_parse_cmds_for_loaders, add_parse_cmds_for_loggers
from common.common_nn_ops import set_all_gpu_config, get_loader_from_name
from gan.wrapper_registry import get_infer_wrapper_dict


def main(_):
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loaders(parser)
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_app(parser)
    flags, unparsed = parser.parse_known_args()

    numpy.set_printoptions(precision=5, suppress=True)
    loader = get_loader_from_name(flags.loader_name, flags.path)
    data_set = loader.load_data(flags.neighborhood, True)
    shadow_map, shadow_ratio = loader.load_shadow_map(flags.neighborhood, data_set)

    gan_inf_wrapper_dict = get_infer_wrapper_dict()

    hook = gan_inf_wrapper_dict[flags.gan_type].create_inference_hook(data_set=data_set, loader=loader,
                                                                      log_dir=flags.base_log_path,
                                                                      neighborhood=flags.neighborhood,
                                                                      shadow_map=shadow_map,
                                                                      shadow_ratio=shadow_ratio,
                                                                      validation_iteration_count=0,
                                                                      validation_sample_count=flags.number_of_samples)

    set_all_gpu_config()
    with tf.compat.v1.Session() as sess:
        gan_inf_wrapper_dict[flags.gan_type].create_generator_restorer().restore(sess, flags.base_log_path)
        hook.after_create_session(sess, None)
        run_context = SessionRunContext(original_args=None, session=sess)
        hook.after_run(run_context=run_context, run_values=None)


def add_parse_cmds_for_app(parser):
    parser.add_argument("--number_of_samples", nargs="?", type=int,
                        default=6000,
                        help="Number of samples.")
    parser.add_argument("--gan_type", nargs="?", type=str,
                        default="cycle_gan",
                        help="Gan type to train, possible values; cycle_gan, gan_x2y and gan_y2x")


if __name__ == '__main__':
    tf.compat.v1.app.run()
