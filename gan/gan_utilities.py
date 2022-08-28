import numpy
import tensorflow as tf


class ShadowOpHolder:

    def __init__(self, shadow_op, shadow_op_creater, shadow_op_initializer) -> None:
        super().__init__()
        self.shadow_op_initializer = shadow_op_initializer
        self.shadow_op_creater = shadow_op_creater
        self.shadow_op = shadow_op


def create_simple_shadow_struct(shadow_ratio):
    simple_shadow_func = lambda inp: (inp / numpy.append(shadow_ratio, 1))
    return ShadowOpHolder(shadow_op=simple_shadow_func, shadow_op_creater=lambda: None,
                          shadow_op_initializer=lambda restorer, session: None)


def create_gan_struct(gan_inference_wrapper, model_base_dir, ckpt_relative_path):
    def _construct_gan_inference_graph(input_data):
        hs_converted = gan_inference_wrapper.construct_inference_graph(tf.expand_dims(input_data[:, :, :-1], axis=0),
                                                                       is_shadow_graph=True,
                                                                       clip_invalid_values=False)
        return tf.concat(axis=2, values=[hs_converted[0], input_data[:, :, -1, numpy.newaxis]])

    gan_shadow_func = _construct_gan_inference_graph
    gan_shadow_op_creater = gan_inference_wrapper.create_generator_restorer
    gan_shadow_op_initializer = lambda restorer, session: (
        restorer.restore(session, model_base_dir + ckpt_relative_path))
    return ShadowOpHolder(shadow_op=gan_shadow_func, shadow_op_creater=gan_shadow_op_creater,
                          shadow_op_initializer=gan_shadow_op_initializer)
