from __future__ import division, absolute_import, print_function

import collections

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan import gan_loss
from tensorflow import reduce_mean
from tensorflow_core.contrib import slim
from tensorflow_gan.python.contrib_utils import get_trainable_variables
from tensorflow_gan.python.train import _convert_tensor_or_l_or_d

from shadow_data_generator import _shadowdata_generator_model, _shadowdata_discriminator_model
from gan_common import ValidationHook, input_x_tensor_name, input_y_tensor_name, model_base_name, model_generator_name, \
    adj_shadow_ratio


class CUTModel(
    collections.namedtuple('CUTModel', (
            'generator_inputs',
            'generated_data',
            'generator_variables',
            'generator_scope',
            'generator_fn',
            'real_data',
            'discriminator_real_outputs',
            'discriminator_gen_outputs',
            'discriminator_variables',
            'discriminator_scope',
            'discriminator_fn',
            'feat_discriminator_real_outputs',
            'feat_discriminator_gen_outputs',
            'feat_discriminator_variables',
            'feat_discriminator_scope',
            'feat_discriminator_fn',
    ))):
    """A CUTModel contains all the pieces needed for GAN training.

    Generative Adversarial Networks (https://arxiv.org/abs/1406.2661) attempt
    to create an implicit generative model of data by solving a two agent game.
    The generator generates candidate examples that are supposed to match the
    data distribution, and the discriminator aims to tell the real examples
    apart from the generated samples.

    Args:
      generator_inputs: The random noise source that acts as input to the
        generator.
      generated_data: The generated output data of the GAN.
      generator_variables: A list of all generator variables.
      generator_scope: Variable scope all generator variables live in.
      generator_fn: The generator function.
      real_data: A tensor or real data.
      discriminator_real_outputs: The discriminator's output on real data.
      discriminator_gen_outputs: The discriminator's output on generated data.
      discriminator_variables: A list of all discriminator variables.
      discriminator_scope: Variable scope all discriminator variables live in.
      discriminator_fn: The discriminator function.
    """


def cut_model(
        # Lambdas defining models.
        generator_fn,
        discriminator_fn,
        feat_discriminator_fn,
        # Real data and conditioning.
        real_data,
        generator_inputs,
        # Optional scopes.
        generator_scope='Generator',
        discriminator_scope='Discriminator',
        feat_discriminator_scope='FeatDiscriminator',
        # Options.
        check_shapes=True):
    """Returns GAN model outputs and variables.

    Args:
      generator_fn: A python lambda that takes `generator_inputs` as inputs and
        returns the outputs of the GAN generator.
      discriminator_fn: A python lambda that takes `real_data`/`generated data`
        and `generator_inputs`. Outputs a Tensor in the range [-inf, inf].
      real_data: A Tensor representing the real data.
      generator_inputs: A Tensor or list of Tensors to the generator. In the
        vanilla GAN case, this might be a single noise Tensor. In the conditional
        GAN case, this might be the generator's conditioning.
      generator_scope: Optional generator variable scope. Useful if you want to
        reuse a subgraph that has already been created.
      discriminator_scope: Optional discriminator variable scope. Useful if you
        want to reuse a subgraph that has already been created.
      check_shapes: If `True`, check that generator produces Tensors that are the
        same shape as real data. Otherwise, skip this check.

    Returns:
      A GANModel namedtuple.

    Raises:
      ValueError: If the generator outputs a Tensor that isn't the same shape as
        `real_data`.
      ValueError: If TF is executing eagerly.
    """
    if tf.executing_eagerly():
        raise ValueError('`vut_model` doesn\'t work when executing eagerly.')
    # Create models
    with tf.compat.v1.variable_scope(
            generator_scope, reuse=tf.compat.v1.AUTO_REUSE) as gen_scope:
        generator_inputs = _convert_tensor_or_l_or_d(generator_inputs)
        generated_data = generator_fn(generator_inputs)
    with tf.compat.v1.variable_scope(
            discriminator_scope, reuse=tf.compat.v1.AUTO_REUSE) as dis_scope:
        discriminator_gen_outputs = discriminator_fn(generated_data,
                                                     generator_inputs)
    with tf.compat.v1.variable_scope(dis_scope, reuse=True):
        real_data = _convert_tensor_or_l_or_d(real_data)
        discriminator_real_outputs = discriminator_fn(real_data, generator_inputs)

    if check_shapes:
        if not generated_data.shape.is_compatible_with(real_data.shape):
            raise ValueError(
                'Generator output shape (%s) must be the same shape as real data '
                '(%s).' % (generated_data.shape, real_data.shape))

    # Get model-specific variables.
    generator_variables = get_trainable_variables(gen_scope)
    discriminator_variables = get_trainable_variables(dis_scope)

    return CUTModel(
        generator_inputs, generated_data, generator_variables, gen_scope, generator_fn,
        real_data,
        discriminator_real_outputs, discriminator_gen_outputs, discriminator_variables, dis_scope, discriminator_fn)


class CUTWrapper:

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

        x_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name=input_x_tensor_name)
        y_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name=input_y_tensor_name)
        model_for_validation = self.define_model(x_input_tensor, y_input_tensor)
        shadowed_validation_hook = ValidationHook(iteration_freq=validation_iteration_count,
                                                  sample_count=validation_sample_count,
                                                  log_dir=log_dir,
                                                  loader=loader, data_set=data_set, neighborhood=neighborhood,
                                                  shadow_map=shadow_map,
                                                  shadow_ratio=adj_shadow_ratio(shadow_ratio, self._swap_inputs),
                                                  input_tensor=y_input_tensor if self._swap_inputs else x_input_tensor,
                                                  model=model_for_validation.generated_data,
                                                  fetch_shadows=False,
                                                  name_suffix="deshadowed" if self._swap_inputs else "shadowed")
        return shadowed_validation_hook


class CUTInferenceWrapper:
    def __init__(self, fetch_shadows):
        self.fetch_shadows = fetch_shadows

    def construct_inference_graph(self, input_tensor, is_shadow_graph, clip_invalid_values=False):
        shp = input_tensor.get_shape()

        output_tensor_in_col = []
        for first_dim in range(shp[1]):
            output_tensor_in_row = []
            for second_dim in range(shp[2]):
                input_cell = tf.expand_dims(tf.expand_dims(input_tensor[:, first_dim, second_dim], 0), 0)
                with tf.variable_scope(model_base_name):
                    with tf.variable_scope(model_generator_name, reuse=tf.AUTO_REUSE):
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

    @staticmethod
    def __create_input_tensor(data_set, loader, is_shadow_graph):
        element_size = loader.get_data_shape(data_set)
        element_size = [1, element_size[0], element_size[1], element_size[2] - 1]
        input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size,
                                      name=input_x_tensor_name if is_shadow_graph else input_y_tensor_name)
        return input_tensor

    def make_inference_graph(self, data_set, loader, is_shadow_graph, clip_invalid_values):
        input_tensor = self.__create_input_tensor(data_set, loader, is_shadow_graph)
        generated = self.construct_inference_graph(input_tensor, is_shadow_graph, clip_invalid_values)
        return input_tensor, generated

    def create_generator_restorer(self):
        # Restore all the variables that were saved in the checkpoint.
        gan_restorer = tf.train.Saver(
            slim.get_variables_to_restore(include=[model_base_name]),
            name='GeneratorRestoreHandler'
        )
        return gan_restorer

    def create_inference_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                              validation_sample_count):
        input_tensor = self.__create_input_tensor(data_set, loader, self.fetch_shadows)
        return ValidationHook(iteration_freq=0,
                              sample_count=validation_sample_count,
                              log_dir=log_dir,
                              loader=loader, data_set=data_set, neighborhood=neighborhood,
                              shadow_map=shadow_map,
                              shadow_ratio=adj_shadow_ratio(shadow_ratio, self.fetch_shadows),
                              input_tensor=input_tensor,
                              model=self.construct_inference_graph(input_tensor, None),
                              fetch_shadows=self.fetch_shadows, name_suffix="")
