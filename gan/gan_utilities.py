from functools import partial

import numpy
import tensorflow as tf


class ShadowOpHolder:

    def __init__(self, shadow_op, deshadow_op, shadow_op_creater, shadow_op_initializer) -> None:
        super().__init__()
        self.shadow_op_initializer = shadow_op_initializer
        self.shadow_op_creater = shadow_op_creater
        self.shadow_op = shadow_op
        self.deshadow_op = deshadow_op


def create_simple_shadow_struct(shadow_ratio):
    def simple_shadow_func(inp):
        return inp / numpy.append(shadow_ratio, 1)

    def simple_deshadow_func(inp):
        return inp * numpy.append(shadow_ratio, 1)

    return ShadowOpHolder(shadow_op=simple_shadow_func,
                          deshadow_op=simple_deshadow_func,
                          shadow_op_creater=lambda: None,
                          shadow_op_initializer=lambda restorer, session: None)


def create_gan_struct(gan_inference_wrapper, model_base_dir, ckpt_relative_path):
    def _build_shadowed_inference_graph(input_data, is_shadow_graph):
        hs_converted = gan_inference_wrapper.construct_inference_graph(tf.expand_dims(input_data[:, :, :-1], axis=0),
                                                                       is_shadow_graph=is_shadow_graph,
                                                                       clip_invalid_values=False)
        return tf.concat(axis=2, values=[hs_converted[0], input_data[:, :, -1, numpy.newaxis]])

    def _initializer(restorer, session):
        restorer.restore(session, model_base_dir + ckpt_relative_path)

    return ShadowOpHolder(shadow_op=partial(_build_shadowed_inference_graph, is_shadow_graph=True),
                          deshadow_op=partial(_build_shadowed_inference_graph, is_shadow_graph=False),
                          shadow_op_creater=gan_inference_wrapper.create_generator_restorer,
                          shadow_op_initializer=_initializer)
