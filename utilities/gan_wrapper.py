from __future__ import division, absolute_import, print_function

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan import gan_loss
from tensorflow import reduce_mean
from tensorflow_core.contrib import slim

from shadow_data_generator import _shadowdata_generator_model, _shadowdata_discriminator_model
from gan_common import ValidationHook

model_forward_generator_name = 'ModelX2Y'


class GANWrapper:

    def __init__(self, identity_loss_weight, use_identity_loss, swap_inputs) -> None:
        super().__init__()
        self._identity_loss_weight = identity_loss_weight
        self._use_identity_loss = use_identity_loss
        self._swap_inputs = swap_inputs

    def define_model(self, images_x, images_y):
        """Defines a CycleGAN model that maps between images_x and images_y.

        Args:
          images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
          images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.
          use_identity_loss: Whether to use identity loss or not

        Returns:
          A `CycleGANModel` namedtuple.
        """
        if self._swap_inputs:
            generator_inputs = images_y
            real_data = images_x
        else:
            generator_inputs = images_x
            real_data = images_y

        gan_model = tfgan.gan_model(
            generator_fn=_shadowdata_generator_model,
            discriminator_fn=_shadowdata_discriminator_model,
            generator_inputs=generator_inputs,
            real_data=real_data)

        # Add summaries for generated images.
        # tfgan.eval.add_cyclegan_image_summaries(gan_model)

        return gan_model

    def define_loss(self, model):
        # Define CycleGAN loss.
        loss = gan_loss(
            model,
            # generator_loss_fn=wasserstein_generator_loss,
            # discriminator_loss_fn=wasserstein_discriminator_loss,
            tensor_pool_fn=tfgan.features.tensor_pool)
        return loss

    def create_validation_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                               validation_iteration_count, validation_sample_count):
        element_size = loader.get_data_shape(data_set)
        element_size = [1, element_size[0], element_size[1], element_size[2] - 1]

        x_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='x')
        y_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='y')
        model_for_validation = self.define_model(x_input_tensor, y_input_tensor)
        shadowed_validation_hook = ValidationHook(iteration_freq=validation_iteration_count,
                                                  sample_count=validation_sample_count,
                                                  log_dir=log_dir,
                                                  loader=loader, data_set=data_set, neighborhood=neighborhood,
                                                  shadow_map=shadow_map,
                                                  shadow_ratio=shadow_ratio,
                                                  input_tensor=x_input_tensor,
                                                  model=model_for_validation.generated_data,
                                                  fetch_shadows=False, name_suffix="shadowed")
        return shadowed_validation_hook


class GANInferenceWrapper:
    def __init__(self, fetch_shadows):
        self.fetch_shadows = fetch_shadows

    def construct_inference_graph(self, input_tensor, is_shadow_graph, clip_invalid_values=False):
        shp = input_tensor.get_shape()

        output_tensor_in_col = []
        for first_dim in range(shp[1]):
            output_tensor_in_row = []
            for second_dim in range(shp[2]):
                input_cell = tf.expand_dims(tf.expand_dims(input_tensor[first_dim][second_dim], 0), 0)
                with tf.variable_scope('Model'):
                    with tf.variable_scope(model_forward_generator_name):
                        with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
                            generated_tensor = _shadowdata_generator_model(input_cell, False)
                            if clip_invalid_values:
                                input_mean = reduce_mean(input_cell)
                                generated_mean = reduce_mean(generated_tensor)

                if clip_invalid_values:
                    result_tensor = tf.cond(tf.less(generated_mean, input_mean),
                                            lambda: generated_tensor,
                                            lambda: input_cell)
                else:
                    result_tensor = generated_tensor

                output_tensor_in_row.append(tf.squeeze(result_tensor, [0, 1]))
            image_output_row = tf.concat(output_tensor_in_row, axis=0)
            output_tensor_in_col.append(image_output_row)

        image_output_row = tf.stack(output_tensor_in_col)

        return image_output_row

    def make_inference_graph(self, data_set, loader, is_shadow_graph, clip_invalid_values):
        element_size = loader.get_data_shape(data_set)
        element_size = [1, element_size[0], element_size[1], element_size[2] - 1]
        input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='x')
        generated = self.construct_inference_graph(input_tensor, is_shadow_graph, clip_invalid_values)
        return input_tensor, generated

    def create_generator_restorer(self):
        # Restore all the variables that were saved in the checkpoint.
        gan_restorer = tf.train.Saver(
            slim.get_variables_to_restore(include=["Model/" + model_forward_generator_name]),
            name='GeneratorRestoreHandler'
        )
        return gan_restorer

    def create_inference_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                              validation_sample_count):
        element_size = loader.get_data_shape(data_set)
        element_size = [1, element_size[0], element_size[1], element_size[2] - 1]

        if self.fetch_shadows:
            shadow_ratio = 1. / shadow_ratio
        else:
            shadow_ratio = shadow_ratio

        x_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='x')
        return ValidationHook(iteration_freq=0,
                              sample_count=validation_sample_count,
                              log_dir=log_dir,
                              loader=loader, data_set=data_set, neighborhood=neighborhood,
                              shadow_map=shadow_map,
                              shadow_ratio=shadow_ratio,
                              input_tensor=x_input_tensor,
                              model=self.construct_inference_graph(x_input_tensor),
                              fetch_shadows=self.fetch_shadows, name_suffix="shadowed")
