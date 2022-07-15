from collections import namedtuple

import numpy
import tensorflow as tf

ShadowOperationStruct = namedtuple('ShadowOperationStruct', ['shadow_op', 'shadow_op_creater', 'shadow_op_initializer'])


def _construct_gan_inference_graph(input_data, wrapper):
    with tf.device('/cpu:0'):
        hs_converted = wrapper.construct_inference_graph(tf.expand_dims(input_data[:, :, :-1], axis=0),
                                                         is_shadow_graph=True,
                                                         clip_invalid_values=False)
    axis_id = 2
    return tf.concat(axis=axis_id, values=[hs_converted[0], input_data[:, :, -1, numpy.newaxis]])


def create_simple_shadow_struct(shadow_ratio):
    simple_shadow_func = lambda inp: (inp / numpy.append(shadow_ratio, 1))
    simple_shadow_struct = ShadowOperationStruct(shadow_op=simple_shadow_func, shadow_op_creater=lambda: None,
                                                 shadow_op_initializer=lambda restorer, session: None)
    return simple_shadow_struct


def create_gan_struct(gan_inference_wrapper, model_base_dir, ckpt_relative_path):
    gan_shadow_func = lambda inp: (_construct_gan_inference_graph(inp, gan_inference_wrapper))
    gan_shadow_op_creater = gan_inference_wrapper.create_generator_restorer
    gan_shadow_op_initializer = lambda restorer, session: (
        restorer.restore(session, model_base_dir + ckpt_relative_path))
    gan_struct = ShadowOperationStruct(shadow_op=gan_shadow_func, shadow_op_creater=gan_shadow_op_creater,
                                       shadow_op_initializer=gan_shadow_op_initializer)
    return gan_struct
